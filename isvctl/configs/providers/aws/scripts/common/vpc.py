# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared VPC test helpers.

Provides common VPC operations used across network test scripts:
- VPC creation with tagging and optional DNS
- VPC cleanup / deletion
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from botocore.exceptions import ClientError

from common.errors import delete_with_retry

logger = logging.getLogger(__name__)

# A VPC's CoreDNS service VM is provisioned asynchronously (nova + vm-manager) after the VPC
# reports available. Tests that create and then quickly delete a VPC can race that provisioning
# and orphan the service VM (or kill its boot volume mid-boot). Wait this long after a VPC is
# available so the service VM is fully created before the caller proceeds to delete it.
_SERVICE_VM_SETTLE_SECONDS = float(os.environ.get("ZCOMPUTE_SERVICE_VM_SETTLE_SECONDS", "30"))


def settle_after_vpc_available() -> None:
    """Wait for a VPC's async CoreDNS service VM (nova + vm-manager) to finish provisioning.

    Every VPC-creation path should call this once the VPC is available, before the VPC can be
    deleted - otherwise a fast create/delete races the provisioning and orphans the service VM.
    """
    time.sleep(_SERVICE_VM_SETTLE_SECONDS)


def create_test_vpc(
    ec2: Any,
    cidr: str,
    name: str,
    *,
    enable_dns: bool = False,
) -> dict[str, Any]:
    """Create a tagged test VPC and wait for it to become available.

    Args:
        ec2: Boto3 EC2 client.
        cidr: CIDR block for the VPC (e.g., "10.94.0.0/16").
        name: Name tag for the VPC.
        enable_dns: If True, enable DNS support and hostnames on the VPC.

    Returns:
        Dict with keys: passed, vpc_id, cidr, message/error.
    """
    result: dict[str, Any] = {"passed": False}
    try:
        vpc = ec2.create_vpc(CidrBlock=cidr)
        vpc_id = vpc["Vpc"]["VpcId"]
        result["vpc_id"] = vpc_id  # Set early so finally-block cleanup can find it on partial failure

        ec2.create_tags(
            Resources=[vpc_id],
            Tags=[
                {"Key": "Name", "Value": name},
                {"Key": "CreatedBy", "Value": "isvtest"},
            ],
        )

        waiter = ec2.get_waiter("vpc_available")
        waiter.wait(VpcIds=[vpc_id])

        if enable_dns:
            ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={"Value": True})
            ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})

        # Let the asynchronous CoreDNS service VM finish provisioning before the caller can
        # delete this VPC, otherwise a fast create/delete races and orphans the service VM.
        settle_after_vpc_available()

        result["passed"] = True
        result["cidr"] = cidr
        result["message"] = f"Created VPC {vpc_id}"
    except ClientError as e:
        result["error"] = str(e)

    return result


def delete_vpc(ec2: Any, vpc_id: str) -> None:
    """Delete a VPC with transient-error retry.

    Routes through ``delete_with_retry`` so a transient throttling or
    endpoint-reset does not leak the VPC on the finally-block path.

    Args:
        ec2: Boto3 EC2 client.
        vpc_id: VPC ID to delete.
    """
    delete_with_retry(
        ec2.delete_vpc,
        VpcId=vpc_id,
        resource_desc=f"VPC {vpc_id}",
    )


def delete_peering_connections_for_vpc(
    ec2: Any,
    vpc_id: str,
    *,
    wait_timeout: float = 60.0,
    poll_seconds: float = 2.0,
) -> list[str]:
    """Delete every VPC peering connection ``vpc_id`` is part of, waiting until each is gone.

    A peering connection blocks ``delete_vpc`` until it is fully removed, and the backend
    tears it down asynchronously - so we request deletion and then poll until each
    connection is actually gone before returning, otherwise the subsequent VPC delete races
    the teardown and fails with ``DependencyViolation``.

    Connections are discovered with the ``requester/accepter-vpc-info.vpc-id`` filters (the
    unfiltered describe returns ``InternalFailure`` on zCompute). This is best-effort cleanup
    invoked from ``finally`` blocks, so it never raises - a discovery/delete failure is logged
    and the caller proceeds (the VPC delete still retries ``DependencyViolation``).

    Args:
        ec2: Boto3 EC2 client.
        vpc_id: VPC whose peering connections should be removed.
        wait_timeout: Max seconds to wait for all deletions to complete.
        poll_seconds: Delay between status polls.

    Returns:
        The peering connection IDs that were waited on (delete requested, or already deleting).
    """
    waiting_on: list[str] = []
    try:
        # Discover every connection this VPC is part of that isn't already fully gone. A connection
        # already in "deleting" still blocks delete_vpc, so we must wait for it too (not skip it).
        statuses: dict[str, str] = {}
        for role in ("requester-vpc-info.vpc-id", "accepter-vpc-info.vpc-id"):
            response = ec2.describe_vpc_peering_connections(Filters=[{"Name": role, "Values": [vpc_id]}])
            for pc in response.get("VpcPeeringConnections", []):
                code = pc.get("Status", {}).get("Code")
                if code != "deleted":
                    statuses[pc["VpcPeeringConnectionId"]] = code

        for pcid, code in statuses.items():
            if code == "deleting":
                waiting_on.append(pcid)  # already tearing down - just wait for it to finish
            elif delete_with_retry(
                ec2.delete_vpc_peering_connection,
                VpcPeeringConnectionId=pcid,
                resource_desc=f"VPC peering connection {pcid}",
            ):
                waiting_on.append(pcid)
            else:
                logger.warning("Could not request deletion of peering connection %s for VPC %s", pcid, vpc_id)

        for pcid in waiting_on:
            deadline = time.monotonic() + wait_timeout  # per-connection budget, not shared
            confirmed_gone = False
            while time.monotonic() < deadline:
                try:
                    conns = ec2.describe_vpc_peering_connections(VpcPeeringConnectionIds=[pcid]).get(
                        "VpcPeeringConnections", []
                    )
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") == "InvalidVpcPeeringConnectionID.NotFound":
                        confirmed_gone = True
                        break
                    # A transient error polling THIS connection must not abort the others; log,
                    # stop waiting on this one, and move on.
                    logger.warning("Error polling peering connection %s for VPC %s: %s", pcid, vpc_id, e)
                    break
                if not conns or conns[0].get("Status", {}).get("Code") == "deleted":
                    confirmed_gone = True
                    break
                time.sleep(poll_seconds)
            if not confirmed_gone:
                # Don't fail silently: the VPC delete that follows may hit DependencyViolation.
                logger.warning(
                    "Peering connection %s for VPC %s not confirmed deleted within %ss",
                    pcid,
                    vpc_id,
                    wait_timeout,
                )
    except Exception:  # best-effort cleanup must never raise into a finally block
        logger.warning("Best-effort peering-connection cleanup failed for VPC %s", vpc_id, exc_info=True)
    return waiting_on


def cleanup_vpc_resources(
    ec2: Any,
    vpc_id: str,
    *,
    subnet_ids: list[str] | None = None,
    sg_ids: list[str] | None = None,
    nacl_ids: list[str] | None = None,
) -> None:
    """Clean up VPC and associated resources with transient-error retry.

    Deletes resources in dependency order: SGs -> NACLs -> subnets -> VPC.
    Every delete goes through ``delete_with_retry``, so a transient
    failure on one resource does not orphan the rest of the dependency tree.

    Args:
        ec2: Boto3 EC2 client.
        vpc_id: VPC ID to clean up.
        subnet_ids: Subnet IDs to delete.
        sg_ids: Security group IDs to delete.
        nacl_ids: Network ACL IDs to delete.
    """
    for sg_id in sg_ids or []:
        delete_with_retry(
            ec2.delete_security_group,
            GroupId=sg_id,
            resource_desc=f"security group {sg_id}",
        )

    for nacl_id in nacl_ids or []:
        delete_with_retry(
            ec2.delete_network_acl,
            NetworkAclId=nacl_id,
            resource_desc=f"network ACL {nacl_id}",
        )

    for subnet_id in subnet_ids or []:
        delete_with_retry(
            ec2.delete_subnet,
            SubnetId=subnet_id,
            resource_desc=f"subnet {subnet_id}",
        )

    delete_vpc(ec2, vpc_id)
