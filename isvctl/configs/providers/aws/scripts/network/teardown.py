#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Teardown AWS VPC and all associated resources.

Usage:
    python teardown.py --vpc-id vpc-xxx --region us-west-2

Output JSON:
{
    "success": true,
    "resources_destroyed": true,
    "deleted": {
        "instances": ["i-xxx"],
        "security_groups": ["sg-xxx"],
        "subnets": ["subnet-xxx"],
        "route_tables": ["rtb-xxx"],
        "internet_gateways": ["igw-xxx"],
        "vpc": "vpc-xxx"
    }
}
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import boto3
from botocore.exceptions import ClientError
from common.ec2 import sanitize_key_name
from common.errors import (
    SERVICE_VM_PROVISIONING_MSG,
    SERVICE_VM_PROVISIONING_RETRY_SECONDS,
    SERVICE_VM_PROVISIONING_TIMEOUT_SECONDS,
    handle_aws_errors,
)
from common.vpc import delete_peering_connections_for_vpc

logger = logging.getLogger(__name__)


def delete_with_retry(func, resource_type: str, max_retries: int = 5, **kwargs) -> bool:
    """Delete resource with retry for dependency errors."""
    provisioning_deadline = time.monotonic() + SERVICE_VM_PROVISIONING_TIMEOUT_SECONDS
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            func(**kwargs)
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if (
                error_code == "InvalidParameterValue"
                and SERVICE_VM_PROVISIONING_MSG in str(e)
                and time.monotonic() < provisioning_deadline
            ):
                # zCompute rejects VPC deletion while the CoreDNS service VM is still being
                # provisioned; the condition clears on its own, so wait it out without
                # consuming the retry budget (an early teardown would otherwise leak the VPC).
                attempt -= 1
                logger.warning(
                    "%s is still provisioning its service VM; retrying delete in %.0fs",
                    resource_type,
                    SERVICE_VM_PROVISIONING_RETRY_SECONDS,
                )
                time.sleep(SERVICE_VM_PROVISIONING_RETRY_SECONDS)
                continue
            if error_code == "DependencyViolation":
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
            elif error_code in [
                "InvalidGroup.NotFound",
                "InvalidSubnetID.NotFound",
                "InvalidRouteTableID.NotFound",
                "InvalidInternetGatewayID.NotFound",
                "InvalidVpcID.NotFound",
            ]:
                return True  # Already deleted
            raise
    return False


def cleanup_key_pairs(ec2: Any, key_names: list[str]) -> list[str]:
    """Delete key pairs by exact name (AWS + local PEM files).

    AWS key pair names allow any printable ASCII up to 255 chars, so pass
    the raw name to EC2. Sanitization is only needed when composing the
    local ``/tmp/<name>.pem`` path, where a crafted name could traverse.
    """
    deleted = []
    for raw_name in key_names:
        try:
            ec2.describe_key_pairs(KeyNames=[raw_name])
            ec2.delete_key_pair(KeyName=raw_name)
            deleted.append(raw_name)
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidKeyPair.NotFound":
                raise

        # Clean up local PEM file (0400 permissions require chmod first)
        try:
            safe_name = sanitize_key_name(raw_name)
        except ValueError as e:
            logger.warning("Key pair %r deleted but local PEM filename is unsafe: %s", raw_name, e)
            continue
        pem_path = Path(f"/tmp/{safe_name}.pem")
        if pem_path.exists():
            pem_path.chmod(0o600)
            pem_path.unlink()
    return deleted


def teardown_vpc(ec2: Any, vpc_id: str) -> dict[str, Any]:
    """Delete VPC and all associated resources."""
    deleted = {
        "instances": [],
        "key_pairs": [],
        "security_groups": [],
        "subnets": [],
        "route_tables": [],
        "internet_gateways": [],
        "peering_connections": [],
        "vpc": None,
    }

    # Terminate all instances in VPC and collect their key names
    instances = ec2.describe_instances(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "instance-state-name", "Values": ["running", "stopped", "pending"]},
        ]
    )
    instance_ids = []
    instance_key_names: set[str] = set()
    for reservation in instances["Reservations"]:
        for instance in reservation["Instances"]:
            tag_keys = {t["Key"] for t in instance.get("Tags", [])}
            if "managed_by_VPC" in tag_keys:
                # zCompute service VMs (e.g. the per-VPC CoreDNS) are managed by the VPC
                # service and can only be removed by deleting the VPC itself; calling
                # TerminateInstances on them is rejected (403, surfaced as
                # InternalServerError) and would abort the whole teardown.
                continue
            instance_ids.append(instance["InstanceId"])
            if instance.get("KeyName"):
                instance_key_names.add(instance["KeyName"])

    if instance_ids:
        ec2.terminate_instances(InstanceIds=instance_ids)
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=instance_ids)
        deleted["instances"] = instance_ids

    # Clean up key pairs that were used by terminated instances
    if instance_key_names:
        deleted["key_pairs"] = cleanup_key_pairs(ec2, list(instance_key_names))

    # Delete security groups (except default)
    sgs = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    for sg in sgs["SecurityGroups"]:
        if sg["GroupName"] != "default":
            delete_with_retry(
                ec2.delete_security_group,
                "security_group",
                GroupId=sg["GroupId"],
            )
            deleted["security_groups"].append(sg["GroupId"])

    # Delete subnets
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    for subnet in subnets["Subnets"]:
        delete_with_retry(
            ec2.delete_subnet,
            "subnet",
            SubnetId=subnet["SubnetId"],
        )
        deleted["subnets"].append(subnet["SubnetId"])

    # Delete route tables (except main)
    rtbs = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    for rtb in rtbs["RouteTables"]:
        # Skip main route table
        is_main = any(assoc.get("Main", False) for assoc in rtb.get("Associations", []))
        if not is_main:
            # Delete associations first
            for assoc in rtb.get("Associations", []):
                if not assoc.get("Main", False) and assoc.get("RouteTableAssociationId"):
                    try:
                        ec2.disassociate_route_table(AssociationId=assoc["RouteTableAssociationId"])
                    except ClientError:
                        pass

            delete_with_retry(
                ec2.delete_route_table,
                "route_table",
                RouteTableId=rtb["RouteTableId"],
            )
            deleted["route_tables"].append(rtb["RouteTableId"])

    # Detach and delete internet gateways
    igws = ec2.describe_internet_gateways(Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}])
    for igw in igws["InternetGateways"]:
        try:
            ec2.detach_internet_gateway(
                InternetGatewayId=igw["InternetGatewayId"],
                VpcId=vpc_id,
            )
        except ClientError:
            pass

        delete_with_retry(
            ec2.delete_internet_gateway,
            "internet_gateway",
            InternetGatewayId=igw["InternetGatewayId"],
        )
        deleted["internet_gateways"].append(igw["InternetGatewayId"])

    # Delete VPC peering connections (otherwise the VPC delete fails with DependencyViolation)
    deleted["peering_connections"] = delete_peering_connections_for_vpc(ec2, vpc_id)

    # Delete VPC
    delete_with_retry(ec2.delete_vpc, "vpc", VpcId=vpc_id)
    deleted["vpc"] = vpc_id

    return deleted


@handle_aws_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Teardown VPC")
    parser.add_argument("--vpc-id", required=True, help="VPC ID to delete")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--skip-destroy", action="store_true", help="Skip actual destroy")
    args = parser.parse_args()

    result = {
        "success": False,
        "platform": "network",
        "resources_destroyed": False,
        "network_id": args.vpc_id,
        "deleted": {},
    }

    # Skip only if explicitly requested via flag or env var
    skip_destroy = args.skip_destroy or os.environ.get("AWS_NETWORK_SKIP_TEARDOWN", "").lower() == "true"
    if skip_destroy:
        result["success"] = True
        result["message"] = "Destroy skipped (--skip-destroy flag or AWS_NETWORK_SKIP_TEARDOWN=true)"
        print(json.dumps(result, indent=2))
        return 0

    ec2 = boto3.client("ec2", region_name=args.region)

    try:
        deleted = teardown_vpc(ec2, args.vpc_id)
        result["deleted"] = deleted
        result["success"] = True
        result["resources_destroyed"] = True
        result["message"] = "VPC and all resources destroyed successfully"
    except ClientError as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
