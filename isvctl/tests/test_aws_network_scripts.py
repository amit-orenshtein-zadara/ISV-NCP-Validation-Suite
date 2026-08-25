# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for AWS network reference scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from botocore.exceptions import ClientError

ISVCTL_ROOT = Path(__file__).resolve().parents[1]
AWS_NETWORK_SCRIPTS = ISVCTL_ROOT / "configs" / "providers" / "aws" / "scripts" / "network"
MY_ISV_NETWORK_SCRIPTS = ISVCTL_ROOT / "configs" / "providers" / "my-isv" / "scripts" / "network"


def _load_network_script(script_name: str) -> ModuleType:
    """Load an AWS network script as a module for direct helper testing."""
    script_path = AWS_NETWORK_SCRIPTS / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_path.stem}", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client_error(operation_name: str, code: str = "AccessDenied", message: str = "denied") -> ClientError:
    """Create a botocore ClientError for fake AWS client failures."""
    return ClientError({"Error": {"Code": code, "Message": message}}, operation_name)


class FakeServiceScopingEc2:
    """Fake EC2 client covering the calls used by test_service_scoping."""

    def __init__(
        self,
        endpoint_eni_ids: list[str] | None = None,
        delete_endpoint_error: ClientError | None = None,
        *,
        endpoint_deleted_after_delete: bool = True,
        delete_endpoint_unsuccessful: list[dict[str, Any]] | None = None,
        subnet_dependency_failures: int = 0,
        sg_dependency_failures: int = 0,
    ) -> None:
        """Configure ENIs returned by the endpoint and optional delete failure."""
        self.endpoint_eni_ids = endpoint_eni_ids if endpoint_eni_ids is not None else ["eni-endpoint-1"]
        self.delete_endpoint_error = delete_endpoint_error
        self.endpoint_deleted_after_delete = endpoint_deleted_after_delete
        self.delete_endpoint_unsuccessful = delete_endpoint_unsuccessful or []
        self.subnet_dependency_failures = subnet_dependency_failures
        self.sg_dependency_failures = sg_dependency_failures
        self.delete_subnet_attempts = 0
        self.delete_sg_attempts = 0
        self.created_sg_ingress: list[dict[str, Any]] = []
        self.deleted_endpoints: list[str] = []
        self.deleted_subnets: list[str] = []
        self.deleted_sgs: list[str] = []
        self.deleted_enis: list[str] = []

    def create_subnet(self, VpcId: str, CidrBlock: str, AvailabilityZone: str) -> dict[str, Any]:
        """Return a fake subnet."""
        return {"Subnet": {"SubnetId": "subnet-aaa", "VpcId": VpcId, "CidrBlock": CidrBlock}}

    def create_security_group(
        self,
        GroupName: str,
        Description: str,
        VpcId: str,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a fake SG ID."""
        return {"GroupId": "sg-svc"}

    def authorize_security_group_ingress(self, GroupId: str, IpPermissions: list[dict[str, Any]]) -> dict[str, Any]:
        """Record the SG rule that was authorized."""
        self.created_sg_ingress.append({"GroupId": GroupId, "IpPermissions": IpPermissions})
        return {}

    def create_vpc_endpoint(
        self,
        VpcId: str,
        ServiceName: str,
        VpcEndpointType: str,
        SubnetIds: list[str],
        SecurityGroupIds: list[str],
        PrivateDnsEnabled: bool,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a fake VPC interface endpoint."""
        assert VpcEndpointType == "Interface"
        assert ServiceName.startswith("com.amazonaws.")
        assert PrivateDnsEnabled is False
        return {"VpcEndpoint": {"VpcEndpointId": "vpce-svc"}}

    def create_network_interface(self, SubnetId: str, **kwargs: Any) -> dict[str, Any]:
        """Return a fake unrelated ENI without an SG."""
        return {"NetworkInterface": {"NetworkInterfaceId": "eni-other"}}

    def describe_vpc_endpoints(self, VpcEndpointIds: list[str]) -> dict[str, Any]:
        """Report the endpoint with its ENI IDs (or absence after deletion)."""
        if VpcEndpointIds[0] in self.deleted_endpoints and self.endpoint_deleted_after_delete:
            return {"VpcEndpoints": []}
        return {
            "VpcEndpoints": [
                {
                    "VpcEndpointId": VpcEndpointIds[0],
                    "NetworkInterfaceIds": list(self.endpoint_eni_ids),
                    "State": "available",
                }
            ]
        }

    def describe_network_interfaces(self, NetworkInterfaceIds: list[str]) -> dict[str, Any]:
        """Report SG attachment: SG attached to endpoint ENIs, none on the unrelated ENI."""
        nics = []
        for nic_id in NetworkInterfaceIds:
            if nic_id in self.endpoint_eni_ids:
                nics.append({"NetworkInterfaceId": nic_id, "Groups": [{"GroupId": "sg-svc"}]})
            else:
                nics.append({"NetworkInterfaceId": nic_id, "Groups": []})
        return {"NetworkInterfaces": nics}

    def delete_vpc_endpoints(self, VpcEndpointIds: list[str]) -> dict[str, Any]:
        """Delete the endpoint, optionally raising a configured error."""
        if self.delete_endpoint_error:
            raise self.delete_endpoint_error
        self.deleted_endpoints.extend(VpcEndpointIds)
        return {"Unsuccessful": self.delete_endpoint_unsuccessful}

    def delete_network_interface(self, NetworkInterfaceId: str) -> None:
        """Delete a fake ENI."""
        self.deleted_enis.append(NetworkInterfaceId)

    def delete_subnet(self, SubnetId: str) -> None:
        """Delete a fake subnet."""
        self.delete_subnet_attempts += 1
        if self.delete_subnet_attempts <= self.subnet_dependency_failures:
            raise _client_error("DeleteSubnet", "DependencyViolation", "subnet has dependencies")
        self.deleted_subnets.append(SubnetId)

    def delete_security_group(self, GroupId: str) -> None:
        """Delete a fake SG."""
        self.delete_sg_attempts += 1
        if self.delete_sg_attempts <= self.sg_dependency_failures:
            raise _client_error("DeleteSecurityGroup", "DependencyViolation", "SG has dependencies")
        self.deleted_sgs.append(GroupId)


def test_service_scoping_happy_path_attaches_sg_only_to_endpoint_eni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SG must attach to the endpoint's ENIs and not to the unrelated ENI."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeServiceScopingEc2(endpoint_eni_ids=["eni-endpoint-1", "eni-endpoint-2"])

    result = module.test_service_scoping(ec2, "vpc-test", "us-west-2a", "us-west-2")

    assert result["create_sg"]["passed"] is True
    assert result["apply_service_rule"]["passed"] is True
    assert result["service_endpoint_allowed"]["passed"] is True
    assert result["other_endpoint_blocked"]["passed"] is True
    assert result["cleanup"]["passed"] is True
    assert ec2.created_sg_ingress[0]["IpPermissions"][0]["FromPort"] == 443
    assert ec2.created_sg_ingress[0]["IpPermissions"][0]["ToPort"] == 443
    assert ec2.deleted_endpoints == ["vpce-svc"]
    assert ec2.deleted_enis == ["eni-other"]
    assert ec2.deleted_subnets == ["subnet-aaa"]
    assert ec2.deleted_sgs == ["sg-svc"]


def test_service_scoping_records_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed VPC endpoint deletion is reported via the cleanup result."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeServiceScopingEc2(
        endpoint_eni_ids=["eni-endpoint-1"],
        delete_endpoint_error=_client_error("DeleteVpcEndpoints"),
    )

    result = module.test_service_scoping(ec2, "vpc-test", "us-west-2a", "us-west-2")

    assert result["service_endpoint_allowed"]["passed"] is True
    assert result["cleanup"]["passed"] is False
    assert "delete VPC endpoint vpce-svc" in result["cleanup"]["error"]


def test_service_scoping_records_endpoint_delete_unsuccessful(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsuccessful delete_vpc_endpoints entries should fail cleanup."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeServiceScopingEc2(
        endpoint_eni_ids=["eni-endpoint-1"],
        delete_endpoint_unsuccessful=[{"ResourceId": "vpce-svc", "Error": {"Code": "UnauthorizedOperation"}}],
    )

    result = module.test_service_scoping(ec2, "vpc-test", "us-west-2a", "us-west-2")

    assert result["cleanup"]["passed"] is False
    assert "delete_vpc_endpoints reported unsuccessful entries" in result["cleanup"]["error"]
    assert ec2.deleted_enis == ["eni-other"]
    assert ec2.deleted_subnets == ["subnet-aaa"]
    assert ec2.deleted_sgs == ["sg-svc"]


def test_service_scoping_records_endpoint_wait_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint deletion wait timeouts should be the visible cleanup cause."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeServiceScopingEc2(
        endpoint_eni_ids=["eni-endpoint-1"],
        endpoint_deleted_after_delete=False,
    )

    result = module.test_service_scoping(ec2, "vpc-test", "us-west-2a", "us-west-2")

    assert result["cleanup"]["passed"] is False
    assert result["cleanup"]["error"].startswith("delete VPC endpoint vpce-svc: Timed out waiting")
    assert ec2.deleted_enis == ["eni-other"]
    assert ec2.deleted_subnets == ["subnet-aaa"]
    assert ec2.deleted_sgs == ["sg-svc"]


def test_service_scoping_retries_dependency_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subnet and SG cleanup should retry brief dependency lag after endpoint deletion."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeServiceScopingEc2(
        endpoint_eni_ids=["eni-endpoint-1"],
        subnet_dependency_failures=2,
        sg_dependency_failures=1,
    )

    result = module.test_service_scoping(ec2, "vpc-test", "us-west-2a", "us-west-2")

    assert result["cleanup"]["passed"] is True
    assert ec2.delete_subnet_attempts == 3
    assert ec2.delete_sg_attempts == 2
    assert ec2.deleted_subnets == ["subnet-aaa"]
    assert ec2.deleted_sgs == ["sg-svc"]


class FakeEndpointDeletionWaitEc2:
    """Fake EC2 client for endpoint deletion polling."""

    def __init__(self, error: ClientError) -> None:
        """Configure the error raised by describe_vpc_endpoints."""
        self.error = error

    def describe_vpc_endpoints(self, VpcEndpointIds: list[str]) -> dict[str, Any]:
        """Raise the configured describe error."""
        raise self.error


def test_wait_for_endpoint_deletion_treats_not_found_as_success() -> None:
    """AWS NotFound during endpoint deletion means the endpoint is already gone."""
    module = _load_network_script("sg_scoping_test.py")
    ec2 = FakeEndpointDeletionWaitEc2(
        _client_error("DescribeVpcEndpoints", "InvalidVpcEndpointId.NotFound", "endpoint not found")
    )

    module._wait_for_endpoint_deletion(ec2, "vpce-svc", attempts=1, delay=0)


def test_wait_for_endpoint_deletion_reraises_unexpected_client_error() -> None:
    """Unexpected describe errors should still fail cleanup."""
    module = _load_network_script("sg_scoping_test.py")
    ec2 = FakeEndpointDeletionWaitEc2(_client_error("DescribeVpcEndpoints", "RequestLimitExceeded", "throttled"))

    with pytest.raises(ClientError):
        module._wait_for_endpoint_deletion(ec2, "vpce-svc", attempts=1, delay=0)


class FakeNeverDeletedEndpointEc2:
    """Fake EC2 client that keeps reporting the endpoint as present."""

    def describe_vpc_endpoints(self, VpcEndpointIds: list[str]) -> dict[str, Any]:
        """Report a still-present endpoint."""
        return {
            "VpcEndpoints": [
                {
                    "VpcEndpointId": VpcEndpointIds[0],
                    "NetworkInterfaceIds": ["eni-endpoint-1"],
                    "State": "deleting",
                }
            ]
        }


def test_wait_for_endpoint_deletion_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Polling exhaustion must surface as a timeout instead of returning success."""
    module = _load_network_script("sg_scoping_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeNeverDeletedEndpointEc2()

    with pytest.raises(TimeoutError, match="Timed out waiting for VPC endpoint vpce-svc deletion"):
        module._wait_for_endpoint_deletion(ec2, "vpce-svc", attempts=2, delay=0)


class FakeSdnLoggingEc2:
    """Fake EC2 client for SDN logging script tests."""

    def __init__(
        self,
        flow_logs: list[dict[str, Any]] | None = None,
        flow_log_error: ClientError | None = None,
        delete_sg_error: ClientError | None = None,
        instances: list[dict[str, Any]] | None = None,
        network_interfaces: list[dict[str, Any]] | None = None,
        authorize_error: ClientError | None = None,
        revoke_error: ClientError | None = None,
    ) -> None:
        """Configure Flow Logs, metric resources, and optional SG failures."""
        self.flow_logs = flow_logs or []
        self.flow_log_error = flow_log_error
        self.delete_sg_error = delete_sg_error
        self.instances = instances or []
        self.network_interfaces = network_interfaces or []
        self.authorize_error = authorize_error
        self.revoke_error = revoke_error
        self.authorized_rules: list[dict[str, Any]] = []
        self.revoked_rules: list[dict[str, Any]] = []
        self.deleted_sgs: list[str] = []

    def describe_flow_logs(self, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        """Return configured VPC Flow Logs."""
        assert Filters[0]["Name"] == "resource-id"
        if self.flow_log_error:
            raise self.flow_log_error
        return {"FlowLogs": list(self.flow_logs)}

    def describe_instances(self, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        """Return configured target VPC instances for metric scoping."""
        assert {"Name": "vpc-id", "Values": ["vpc-test"]} in Filters
        return {"Reservations": [{"Instances": list(self.instances)}] if self.instances else []}

    def describe_network_interfaces(self, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        """Return configured target VPC network interfaces for metric scoping."""
        assert Filters == [{"Name": "vpc-id", "Values": ["vpc-test"]}]
        return {"NetworkInterfaces": list(self.network_interfaces)}

    def create_security_group(
        self,
        GroupName: str,
        Description: str,
        VpcId: str,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a fake audit probe security group."""
        assert GroupName.startswith("isv-sdn-audit-")
        assert Description == "ISV SDN09 audit trail probe"
        assert VpcId == "vpc-test"
        assert TagSpecifications
        return {"GroupId": "sg-audit"}

    def authorize_security_group_ingress(self, GroupId: str, IpPermissions: list[dict[str, Any]]) -> dict[str, Any]:
        """Record authorized ingress rules."""
        if self.authorize_error:
            raise self.authorize_error
        self.authorized_rules.append({"GroupId": GroupId, "IpPermissions": IpPermissions})
        return {}

    def revoke_security_group_ingress(self, GroupId: str, IpPermissions: list[dict[str, Any]]) -> dict[str, Any]:
        """Record revoked ingress rules."""
        if self.revoke_error:
            raise self.revoke_error
        self.revoked_rules.append({"GroupId": GroupId, "IpPermissions": IpPermissions})
        return {}

    def delete_security_group(self, GroupId: str) -> dict[str, Any]:
        """Delete a fake security group, optionally raising a configured error."""
        if self.delete_sg_error:
            raise self.delete_sg_error
        self.deleted_sgs.append(GroupId)
        return {}


class FakeHealth:
    """Fake AWS Health client for SDN hardware-fault logging tests."""

    def __init__(self, events: list[dict[str, Any]] | None = None, error: ClientError | None = None) -> None:
        """Configure events or a describe_events error."""
        self.events = events or []
        self.error = error

    def describe_events(self, **kwargs: Any) -> dict[str, Any]:
        """Return configured Health events."""
        assert "EC2" in kwargs["filter"]["services"]
        if self.error:
            raise self.error
        return {"events": list(self.events)}


class FakeCloudWatch:
    """Fake CloudWatch client for latency/performance telemetry tests."""

    def __init__(
        self,
        metrics: list[dict[str, Any]] | None = None,
        datapoints: list[dict[str, Any]] | None = None,
        list_error: ClientError | None = None,
    ) -> None:
        """Configure metrics, datapoints, and optional list_metrics failure."""
        self.metrics = metrics or []
        self.datapoints = datapoints or []
        self.list_error = list_error
        self.list_metric_dimensions: list[list[dict[str, str]] | None] = []

    def list_metrics(
        self,
        Namespace: str,
        MetricName: str,
        Dimensions: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Return configured metrics."""
        assert Namespace == "AWS/EC2"
        assert MetricName == "NetworkPacketsIn"
        self.list_metric_dimensions.append(Dimensions)
        if self.list_error:
            raise self.list_error
        if not Dimensions:
            return {"Metrics": list(self.metrics)}

        requested = {(dimension["Name"], dimension["Value"]) for dimension in Dimensions}
        scoped_metrics = []
        for metric in self.metrics:
            metric_dimensions = {(dimension["Name"], dimension["Value"]) for dimension in metric.get("Dimensions", [])}
            if requested <= metric_dimensions:
                scoped_metrics.append(metric)
        return {"Metrics": scoped_metrics}

    def get_metric_statistics(
        self,
        Namespace: str,
        MetricName: str,
        Dimensions: list[dict[str, str]],
        StartTime: Any,
        EndTime: Any,
        Period: int,
        Statistics: list[str],
    ) -> dict[str, Any]:
        """Return configured datapoints."""
        assert Namespace == "AWS/EC2"
        assert MetricName == "NetworkPacketsIn"
        assert Period == 60
        assert Statistics == ["Sum"]
        assert StartTime < EndTime
        assert Dimensions
        return {"Datapoints": list(self.datapoints)}


class FakeLogs:
    """Fake CloudWatch Logs client for VPC Flow Log samples."""

    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        """Configure log events."""
        self.events = events or []
        self.calls: list[str] = []

    def filter_log_events(
        self,
        logGroupName: str,
        startTime: int,
        endTime: int,
        limit: int,
    ) -> dict[str, Any]:
        """Return configured Flow Log events."""
        assert logGroupName == "/aws/vpc/flow-logs"
        assert startTime < endTime
        assert limit == 10
        self.calls.append(logGroupName)
        return {"events": list(self.events)}


class FakeCloudTrail:
    """Fake CloudTrail client for audit trail tests."""

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        error: ClientError | None = None,
        event_batches: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        """Configure parsed CloudTrail events or a lookup error."""
        self.events = events or []
        self.error = error
        self.event_batches = event_batches
        self.lookup_calls = 0

    def lookup_events(
        self,
        LookupAttributes: list[dict[str, str]],
        StartTime: Any,
        EndTime: Any,
    ) -> dict[str, Any]:
        """Return configured events as CloudTrail LookupEvents entries."""
        assert LookupAttributes == [{"AttributeKey": "ResourceName", "AttributeValue": "sg-audit"}]
        assert StartTime < EndTime
        if self.error:
            raise self.error
        events = self.events
        if self.event_batches is not None:
            batch_index = min(self.lookup_calls, len(self.event_batches) - 1)
            events = self.event_batches[batch_index]
        self.lookup_calls += 1
        return {"Events": [{"CloudTrailEvent": json.dumps(event)} for event in events]}


def _active_flow_log() -> dict[str, Any]:
    """Return a fake active VPC Flow Log."""
    return {
        "FlowLogId": "fl-123",
        "FlowLogStatus": "ACTIVE",
        "LogDestinationType": "cloud-watch-logs",
        "LogGroupName": "/aws/vpc/flow-logs",
        "LogDestination": "arn:aws:logs:us-west-2:123456789012:log-group:/aws/vpc/flow-logs",
    }


def _active_s3_flow_log() -> dict[str, Any]:
    """Return a fake active S3-backed VPC Flow Log."""
    return {
        "FlowLogId": "fl-s3",
        "FlowLogStatus": "ACTIVE",
        "LogDestinationType": "s3",
        "LogDestination": "arn:aws:s3:::isv-flow-logs",
    }


@pytest.mark.parametrize(
    ("aspect", "step_name"),
    [
        ("hardware_faults", "sdn_hardware_fault_logging"),
        ("latency_perf", "sdn_latency_perf_logging"),
        ("audit_trail", "sdn_filter_audit_trail"),
    ],
)
def test_aws_sdn_logging_result_test_names_match_suite_steps(aspect: str, step_name: str) -> None:
    """AWS SDN logging output names must match suite step IDs."""
    module = _load_network_script("sdn_logging_test.py")

    result = module._base_result(aspect, "vpc-test", "us-west-2")

    assert result["test_name"] == step_name


@pytest.mark.parametrize(
    ("aspect", "step_name"),
    [
        ("hardware_faults", "sdn_hardware_fault_logging"),
        ("latency_perf", "sdn_latency_perf_logging"),
        ("audit_trail", "sdn_filter_audit_trail"),
    ],
)
def test_my_isv_sdn_logging_demo_test_names_match_suite_steps(aspect: str, step_name: str) -> None:
    """my-isv SDN logging template output names must match suite step IDs."""
    script = MY_ISV_NETWORK_SCRIPTS / "sdn_logging_test.py"
    env = os.environ | {"ISVCTL_DEMO_MODE": "1"}

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--region",
                "demo-region",
                "--vpc-id",
                "vpc-demo",
                "--aspect",
                aspect,
            ],
            capture_output=True,
            env=env,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"{script} timed out after {exc.timeout} seconds\nstdout: {exc.stdout!r}\nstderr: {exc.stderr!r}")

    assert completed.returncode == 0, completed.stderr
    result: dict[str, Any] = json.loads(completed.stdout)
    assert result["test_name"] == step_name


def test_sdn_hardware_fault_logging_happy_path() -> None:
    """Hardware-fault logging passes with Flow Logs and queryable Health events."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[_active_flow_log()])
    health = FakeHealth(
        events=[
            {
                "arn": "arn:aws:health:global::event/EC2/test",
                "service": "EC2",
                "eventTypeCategory": "issue",
                "startTime": "2026-05-05T00:00:00Z",
            }
        ]
    )

    result = module.check_hardware_fault_logging(ec2, health, "vpc-test", "us-west-2")

    assert result["success"] is True
    assert result["tests"]["logging_endpoint_reachable"]["passed"] is True
    assert result["tests"]["fault_event_source_queryable"]["passed"] is True
    assert result["tests"]["event_schema_valid"]["passed"] is True
    assert result["log_destination"].endswith("/aws/vpc/flow-logs")
    assert result["recent_event_count"] == 1


def test_sdn_hardware_fault_logging_marks_health_subscription_provider_hidden() -> None:
    """AWS Health subscription gating should not fail the hardware-fault check."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[_active_flow_log()])
    health = FakeHealth(
        error=_client_error("DescribeEvents", "SubscriptionRequiredException", "AWS Health subscription required")
    )

    result = module.check_hardware_fault_logging(ec2, health, "vpc-test", "us-west-2")

    assert result["success"] is True
    assert result["tests"]["fault_event_source_queryable"]["provider_hidden"] is True
    assert result["tests"]["event_schema_valid"]["provider_hidden"] is True
    assert result["recent_event_count"] == 0


def test_sdn_hardware_fault_logging_marks_absent_flow_logs_provider_hidden() -> None:
    """Hardware-fault logging does not fail the AWS suite when Flow Logs are not configured."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[])
    health = FakeHealth(events=[])

    result = module.check_hardware_fault_logging(ec2, health, "vpc-test", "us-west-2")

    assert result["success"] is True
    assert result["log_destination"] == "aws-vpc-flow-logs:not-configured"
    assert result["tests"]["log_destination_configured"]["provider_hidden"] is True


def test_sdn_hardware_fault_logging_fails_destination_when_flow_log_query_fails() -> None:
    """Hardware-fault logging must not report absent Flow Logs when the query failed."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_log_error=_client_error("DescribeFlowLogs", "UnauthorizedOperation", "denied"))
    health = FakeHealth(events=[])

    result = module.check_hardware_fault_logging(ec2, health, "vpc-test", "us-west-2")

    log_destination = result["tests"]["log_destination_configured"]
    assert result["success"] is False
    assert result["log_destination"] == "aws-vpc-flow-logs:unknown"
    assert log_destination["passed"] is False
    assert "Unable to inspect VPC Flow Logs" in log_destination["error"]
    assert log_destination["flow_log_query"]["passed"] is False
    assert "provider_hidden" not in log_destination


def test_sdn_hardware_fault_logging_fails_event_schema_when_health_query_fails() -> None:
    """A non-hidden Health query failure must not be reported as a passing schema check."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[_active_flow_log()])
    health = FakeHealth(error=_client_error("DescribeEvents", "UnauthorizedOperation", "denied"))

    result = module.check_hardware_fault_logging(ec2, health, "vpc-test", "us-west-2")

    event_schema_valid = result["tests"]["event_schema_valid"]
    assert result["success"] is False
    assert result["tests"]["fault_event_source_queryable"]["passed"] is False
    assert event_schema_valid["passed"] is False
    assert "provider_hidden" not in event_schema_valid
    assert event_schema_valid["health_query"]["passed"] is False


def test_sdn_latency_perf_logging_happy_path_with_cloudwatch_datapoint() -> None:
    """Latency/performance logging passes when CloudWatch has recent packet datapoints."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(
        flow_logs=[],
        instances=[
            {
                "InstanceId": "i-probe",
                "NetworkInterfaces": [{"NetworkInterfaceId": "eni-probe"}],
            }
        ],
        network_interfaces=[{"NetworkInterfaceId": "eni-probe"}],
    )
    cloudwatch = FakeCloudWatch(
        metrics=[
            {
                "MetricName": "NetworkPacketsIn",
                "Dimensions": [{"Name": "InstanceId", "Value": "i-probe"}],
            }
        ],
        datapoints=[{"Sum": 42.0}],
    )
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is True
    assert result["telemetry_namespace"] == "AWS/EC2"
    assert result["probe_resource_id"] == "i-probe"
    assert [{"Name": "InstanceId", "Value": "i-probe"}] in cloudwatch.list_metric_dimensions
    assert result["tests"]["performance_metric_present"]["provider_hidden"] is True
    assert result["tests"]["samples_recent"]["passed"] is True


def test_sdn_logging_list_packet_metrics_paginates_and_filters() -> None:
    """CloudWatch packet metric discovery must include all list_metrics pages."""
    module = _load_network_script("sdn_logging_test.py")
    calls: list[dict[str, Any]] = []
    dimensions = [{"Name": "InstanceId", "Value": "i-probe"}]
    pages = [
        {
            "Metrics": [
                {
                    "MetricName": "NetworkPacketsIn",
                    "Dimensions": dimensions,
                }
            ],
            "NextToken": "page-2",
        },
        {
            "Metrics": [
                {
                    "MetricName": "NetworkBytesIn",
                    "Dimensions": dimensions,
                },
                {
                    "MetricName": "NetworkPacketsIn",
                    "Dimensions": [{"Name": "InstanceId", "Value": "i-next"}],
                },
            ]
        },
    ]

    class PagedCloudWatch:
        """Fake CloudWatch client returning a tokenized list_metrics response."""

        def list_metrics(self, **kwargs: Any) -> dict[str, Any]:
            """Return the next configured metrics page."""
            calls.append(dict(kwargs))
            return pages[len(calls) - 1]

    metrics = module._list_packet_metrics(PagedCloudWatch(), dimensions)

    assert calls == [
        {
            "Namespace": "AWS/EC2",
            "MetricName": "NetworkPacketsIn",
            "Dimensions": dimensions,
        },
        {
            "Namespace": "AWS/EC2",
            "MetricName": "NetworkPacketsIn",
            "Dimensions": dimensions,
            "NextToken": "page-2",
        },
    ]
    assert metrics == [
        {
            "Namespace": "AWS/EC2",
            "MetricName": "NetworkPacketsIn",
            "Dimensions": dimensions,
        },
        {
            "Namespace": "AWS/EC2",
            "MetricName": "NetworkPacketsIn",
            "Dimensions": [{"Name": "InstanceId", "Value": "i-next"}],
        },
    ]


def test_sdn_latency_perf_logging_cloudwatch_metrics_pass_when_flow_log_query_fails() -> None:
    """CloudWatch packet metrics independently satisfy packet telemetry."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(
        flow_log_error=_client_error("DescribeFlowLogs", "UnauthorizedOperation", "denied"),
        instances=[
            {
                "InstanceId": "i-probe",
                "NetworkInterfaces": [{"NetworkInterfaceId": "eni-probe"}],
            }
        ],
        network_interfaces=[{"NetworkInterfaceId": "eni-probe"}],
    )
    cloudwatch = FakeCloudWatch(
        metrics=[
            {
                "MetricName": "NetworkPacketsIn",
                "Dimensions": [{"Name": "InstanceId", "Value": "i-probe"}],
            }
        ],
        datapoints=[{"Sum": 42.0}],
    )
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is True
    assert result["tests"]["packet_metric_present"]["passed"] is True
    assert result["tests"]["samples_recent"]["passed"] is True
    assert result["telemetry_namespace"] == "AWS/EC2"
    assert result["probe_resource_id"] == "i-probe"


def test_sdn_latency_perf_logging_ignores_account_metrics_when_target_vpc_has_no_resources() -> None:
    """Account-wide EC2 metrics must not count as target VPC telemetry."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[])
    cloudwatch = FakeCloudWatch(
        metrics=[
            {
                "MetricName": "NetworkPacketsIn",
                "Dimensions": [{"Name": "InstanceId", "Value": "i-unrelated"}],
            }
        ],
        datapoints=[{"Sum": 42.0}],
    )
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is True
    assert result["telemetry_namespace"] == "provider-hidden"
    assert result["probe_resource_id"] == "vpc-test"
    assert result["tests"]["packet_metric_present"]["provider_hidden"] is True
    assert result["tests"]["samples_recent"]["provider_hidden"] is True


def test_sdn_latency_perf_logging_fails_packet_metric_when_flow_log_query_fails() -> None:
    """A Flow Logs query failure must not be hidden as absent target telemetry."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_log_error=_client_error("DescribeFlowLogs", "UnauthorizedOperation", "denied"))
    cloudwatch = FakeCloudWatch(metrics=[], datapoints=[])
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    packet_metric = result["tests"]["packet_metric_present"]
    assert result["success"] is False
    assert packet_metric["passed"] is False
    assert "Unable to verify VPC Flow Logs" in packet_metric["error"]
    assert "UnauthorizedOperation" in packet_metric["flow_log_error"]
    assert packet_metric["flow_log_query"]["passed"] is False
    assert "provider_hidden" not in packet_metric


def test_sdn_latency_perf_logging_rejects_unrelated_cloudwatch_metrics_for_target_resources() -> None:
    """Metrics for instances outside the target VPC must not satisfy packet telemetry."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[], instances=[{"InstanceId": "i-probe"}])
    cloudwatch = FakeCloudWatch(
        metrics=[
            {
                "MetricName": "NetworkPacketsIn",
                "Dimensions": [{"Name": "InstanceId", "Value": "i-unrelated"}],
            }
        ],
        datapoints=[{"Sum": 42.0}],
    )
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is False
    assert result["tests"]["packet_metric_present"]["passed"] is False
    assert "No target-VPC CloudWatch packet metric" in result["tests"]["packet_metric_present"]["error"]


def test_sdn_latency_perf_logging_fails_without_recent_samples() -> None:
    """Latency/performance logging fails when no packet telemetry sample is recent."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(
        flow_logs=[_active_flow_log()],
        instances=[{"InstanceId": "i-probe"}],
    )
    cloudwatch = FakeCloudWatch(
        metrics=[
            {
                "MetricName": "NetworkPacketsIn",
                "Dimensions": [{"Name": "InstanceId", "Value": "i-probe"}],
            }
        ],
        datapoints=[],
    )
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is False
    assert result["tests"]["packet_metric_present"]["passed"] is True
    assert result["tests"]["samples_recent"]["passed"] is False
    assert "No recent packet telemetry samples" in result["tests"]["samples_recent"]["error"]


def test_sdn_latency_perf_logging_uses_recent_flow_log_samples() -> None:
    """Flow Log records satisfy the recent sample requirement when metrics are absent."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[_active_flow_log()])
    cloudwatch = FakeCloudWatch(metrics=[], datapoints=[])
    logs = FakeLogs(events=[{"message": "2 123 eni-1 10.0.0.1 10.0.0.2 443 443 6 1 52 1 2 ACCEPT OK"}])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    assert result["success"] is True
    assert result["telemetry_namespace"] == "AWS/VPCFlowLogs"
    assert result["probe_resource_id"] == "vpc-test"
    assert result["tests"]["samples_recent"]["sample_count"] == 1


def test_sdn_latency_perf_logging_marks_s3_flow_log_samples_provider_hidden() -> None:
    """S3-backed Flow Logs are valid telemetry but cannot be sampled through CloudWatch Logs."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(flow_logs=[_active_s3_flow_log()])
    cloudwatch = FakeCloudWatch(metrics=[], datapoints=[])
    logs = FakeLogs(events=[])

    result = module.check_latency_perf_logging(
        ec2,
        cloudwatch,
        logs,
        "vpc-test",
        "us-west-2",
        sample_window_seconds=60,
    )

    samples_recent = result["tests"]["samples_recent"]
    assert result["success"] is True
    assert result["telemetry_namespace"] == "AWS/VPCFlowLogs"
    assert result["probe_resource_id"] == "vpc-test"
    assert result["tests"]["packet_metric_present"]["passed"] is True
    assert samples_recent["provider_hidden"] is True
    assert "s3" in samples_recent["message"]
    assert samples_recent["flow_log_destinations"] == ["arn:aws:s3:::isv-flow-logs"]
    assert logs.calls == []


def test_sdn_audit_trail_logging_happy_path() -> None:
    """Audit trail logging passes when CloudTrail has the SG rule lifecycle."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2()
    cloudtrail = FakeCloudTrail(
        events=[
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
            module._audit_event("DeleteSecurityGroup", "sg-audit"),
        ]
    )

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert result["success"] is True
    assert result["target_rule_id"] == "sg-audit"
    assert result["tests"]["create_rule_logged"]["passed"] is True
    assert result["tests"]["modify_rule_logged"]["passed"] is True
    assert result["tests"]["delete_rule_logged"]["passed"] is True
    assert result["tests"]["cleanup"]["passed"] is True
    assert len(ec2.authorized_rules) == 2
    assert len(ec2.revoked_rules) == 2
    assert ec2.deleted_sgs == ["sg-audit"]


def test_sdn_audit_trail_logging_polls_until_full_lifecycle_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CloudTrail polling must not stop after only the first audit event appears."""
    module = _load_network_script("sdn_logging_test.py")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    ec2 = FakeSdnLoggingEc2()
    cloudtrail = FakeCloudTrail(
        event_batches=[
            [module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit")],
            [
                module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
                module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
                module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
                module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
                module._audit_event("DeleteSecurityGroup", "sg-audit"),
            ],
        ]
    )

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert result["success"] is True
    assert cloudtrail.lookup_calls == 2
    assert result["tests"]["audit_endpoint_reachable"]["passed"] is True


def test_sdn_audit_trail_logging_fails_on_cloudtrail_propagation_timeout() -> None:
    """Missing CloudTrail events should fail with a propagation timeout marker."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2()
    cloudtrail = FakeCloudTrail(events=[])

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert result["success"] is False
    assert result["tests"]["audit_endpoint_reachable"]["passed"] is False
    assert result["tests"]["audit_endpoint_reachable"]["propagation_timeout"] is True
    assert result["tests"]["create_rule_logged"]["passed"] is False


def test_sdn_audit_trail_logging_cleans_up_probe_after_partial_create_failure() -> None:
    """If mutation fails after SG creation, the audit probe SG must still be deleted."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(authorize_error=_client_error("AuthorizeSecurityGroupIngress", "InvalidPermission"))
    cloudtrail = FakeCloudTrail(events=[])

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert result["success"] is False
    assert result["target_rule_id"] == "sg-audit"
    assert result["tests"]["cleanup"]["passed"] is True
    assert ec2.deleted_sgs == ["sg-audit"]


def test_sdn_audit_trail_logging_ignores_create_security_group_event_without_group_id() -> None:
    """CreateSecurityGroup events lack groupId in requestParameters and must not fail required-fields."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2()
    create_event = {
        "eventName": "CreateSecurityGroup",
        "userIdentity": {"type": "AssumedRole", "arn": "arn:aws:sts::123456789012:assumed-role/isv/test"},
        "eventTime": datetime.now(UTC).isoformat(),
        "requestParameters": {"groupName": "isv-sdn-audit", "vpcId": "vpc-test"},
    }
    cloudtrail = FakeCloudTrail(
        events=[
            create_event,
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
        ]
    )

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert result["success"] is True
    assert result["tests"]["audit_event_has_required_fields"]["passed"] is True


def test_sdn_audit_trail_logging_records_cleanup_failure() -> None:
    """Audit probe cleanup failures must be visible in the cleanup subtest."""
    module = _load_network_script("sdn_logging_test.py")
    ec2 = FakeSdnLoggingEc2(delete_sg_error=_client_error("DeleteSecurityGroup", "DependencyViolation", "in use"))
    cloudtrail = FakeCloudTrail(
        events=[
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
            module._audit_event("AuthorizeSecurityGroupIngress", "sg-audit"),
            module._audit_event("RevokeSecurityGroupIngress", "sg-audit"),
            module._audit_event("DeleteSecurityGroup", "sg-audit"),
        ]
    )

    result = module.check_audit_trail_logging(
        ec2,
        cloudtrail,
        "vpc-test",
        "us-west-2",
        timeout_seconds=0,
        poll_seconds=0,
    )

    assert result["success"] is False
    assert result["tests"]["cleanup"]["passed"] is False
    assert "Failed to delete audit probe security group sg-audit" in result["tests"]["cleanup"]["error"]


class FakePeeringEc2:
    """Fake EC2 covering describe/delete of VPC peering connections."""

    def __init__(self, connections: list[dict[str, Any]]) -> None:
        """Seed the fake with the peering connections describe should return."""
        self._connections = connections
        self.deleted: list[str] = []

    def describe_vpc_peering_connections(
        self,
        Filters: list[dict[str, Any]] | None = None,
        VpcPeeringConnectionIds: list[str] | None = None,
        **_kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Answer by-ID lookups and requester/accepter-vpc-id filters (the unfiltered describe returns InternalFailure on zCompute)."""
        if VpcPeeringConnectionIds:
            pcid = VpcPeeringConnectionIds[0]
            code = "deleted" if pcid in self.deleted else "active"
            return {"VpcPeeringConnections": [{"VpcPeeringConnectionId": pcid, "Status": {"Code": code}}]}
        if Filters:
            name, values = Filters[0]["Name"], Filters[0]["Values"]
            key = "RequesterVpcInfo" if name.startswith("requester") else "AccepterVpcInfo"
            matched = [pc for pc in self._connections if pc.get(key, {}).get("VpcId") in values]
            return {"VpcPeeringConnections": matched}
        raise AssertionError("unfiltered describe_vpc_peering_connections returns InternalFailure on zCompute")

    def delete_vpc_peering_connection(self, VpcPeeringConnectionId: str) -> dict[str, Any]:
        """Record the deletion request."""
        self.deleted.append(VpcPeeringConnectionId)
        return {}


def test_delete_peering_connections_for_vpc_targets_only_the_vpc() -> None:
    """Active connections where the VPC is requester OR accepter are deleted; others are skipped."""
    module = _load_network_script("teardown.py")
    ec2 = FakePeeringEc2(
        [
            {
                "VpcPeeringConnectionId": "pcx-requester",
                "RequesterVpcInfo": {"VpcId": "vpc-target"},
                "AccepterVpcInfo": {"VpcId": "vpc-other"},
                "Status": {"Code": "active"},
            },
            {
                "VpcPeeringConnectionId": "pcx-accepter",
                "RequesterVpcInfo": {"VpcId": "vpc-z"},
                "AccepterVpcInfo": {"VpcId": "vpc-target"},
                "Status": {"Code": "active"},
            },
            {
                "VpcPeeringConnectionId": "pcx-unrelated",
                "RequesterVpcInfo": {"VpcId": "vpc-x"},
                "AccepterVpcInfo": {"VpcId": "vpc-y"},
                "Status": {"Code": "active"},
            },
            {
                "VpcPeeringConnectionId": "pcx-already-gone",
                "RequesterVpcInfo": {"VpcId": "vpc-target"},
                "AccepterVpcInfo": {"VpcId": "vpc-w"},
                "Status": {"Code": "deleted"},
            },
        ]
    )

    deleted = module.delete_peering_connections_for_vpc(ec2, "vpc-target")

    assert set(deleted) == {"pcx-requester", "pcx-accepter"}
    assert set(ec2.deleted) == {"pcx-requester", "pcx-accepter"}


class _ScriptablePeeringEc2:
    """Peering fake whose filter results, per-ID poll behaviour, and errors are configurable."""

    def __init__(
        self,
        *,
        filter_conns: list[dict[str, Any]] | None = None,
        by_id: dict[str, Any] | None = None,
        filter_error: Exception | None = None,
    ) -> None:
        self._filter_conns = filter_conns or []
        self._by_id = by_id or {}  # pcid -> {"Code": "..."} | Exception
        self._filter_error = filter_error
        self.deleted: list[str] = []

    def describe_vpc_peering_connections(
        self, Filters: list[dict[str, Any]] | None = None, VpcPeeringConnectionIds: list[str] | None = None, **_: Any
    ) -> dict[str, list[dict[str, Any]]]:
        if VpcPeeringConnectionIds:
            outcome = self._by_id.get(VpcPeeringConnectionIds[0], {"Code": "deleted"})
            if isinstance(outcome, Exception):
                raise outcome
            return {"VpcPeeringConnections": [{"VpcPeeringConnectionId": VpcPeeringConnectionIds[0], "Status": outcome}]}
        if Filters:
            if self._filter_error is not None:
                raise self._filter_error
            name = Filters[0]["Name"]
            key = "RequesterVpcInfo" if name.startswith("requester") else "AccepterVpcInfo"
            vpc = Filters[0]["Values"][0]
            return {"VpcPeeringConnections": [c for c in self._filter_conns if c.get(key, {}).get("VpcId") == vpc]}
        raise AssertionError("unfiltered describe is not used")

    def delete_vpc_peering_connection(self, VpcPeeringConnectionId: str) -> dict[str, Any]:
        self.deleted.append(VpcPeeringConnectionId)
        return {}


def _peering_conn(pcid: str, status: str, *, requester: str = "vpc-target", accepter: str = "vpc-other") -> dict[str, Any]:
    return {
        "VpcPeeringConnectionId": pcid,
        "RequesterVpcInfo": {"VpcId": requester},
        "AccepterVpcInfo": {"VpcId": accepter},
        "Status": {"Code": status},
    }


def test_delete_peering_connections_waits_for_already_deleting_without_redeleting() -> None:
    """A connection already in 'deleting' is waited on (until gone) but not re-deleted."""
    module = _load_network_script("teardown.py")
    ec2 = _ScriptablePeeringEc2(
        filter_conns=[_peering_conn("pcx-del", "deleting")],
        by_id={"pcx-del": {"Code": "deleted"}},
    )

    deleted = module.delete_peering_connections_for_vpc(ec2, "vpc-target")

    assert deleted == ["pcx-del"]  # waited on
    assert ec2.deleted == []  # not re-requested for deletion


def test_delete_peering_connections_warns_when_not_confirmed_deleted(caplog: pytest.LogCaptureFixture) -> None:
    """If a connection isn't confirmed gone within the timeout, that is logged (not silent)."""
    module = _load_network_script("teardown.py")
    ec2 = _ScriptablePeeringEc2(
        filter_conns=[_peering_conn("pcx-stuck", "active")],
        by_id={"pcx-stuck": {"Code": "active"}},
    )

    with caplog.at_level("WARNING"):
        deleted = module.delete_peering_connections_for_vpc(ec2, "vpc-target", wait_timeout=0)

    assert deleted == ["pcx-stuck"]
    assert "not confirmed deleted" in caplog.text


def test_delete_peering_connections_poll_error_does_not_abort_other_connections() -> None:
    """A transient poll error on one connection must not skip waiting on the others."""
    module = _load_network_script("teardown.py")
    ec2 = _ScriptablePeeringEc2(
        filter_conns=[_peering_conn("pcx-1", "active"), _peering_conn("pcx-2", "active")],
        by_id={
            "pcx-1": _client_error("DescribeVpcPeeringConnections", "InternalFailure"),
            "pcx-2": {"Code": "deleted"},
        },
    )

    deleted = module.delete_peering_connections_for_vpc(ec2, "vpc-target")

    # Both were requested for deletion; the poll error on pcx-1 did not abort pcx-2.
    assert set(deleted) == {"pcx-1", "pcx-2"}
    assert set(ec2.deleted) == {"pcx-1", "pcx-2"}


def test_delete_peering_connections_discovery_failure_is_swallowed() -> None:
    """A describe (discovery) failure is logged and swallowed - cleanup never raises."""
    module = _load_network_script("teardown.py")
    ec2 = _ScriptablePeeringEc2(filter_error=_client_error("DescribeVpcPeeringConnections", "InternalFailure"))

    deleted = module.delete_peering_connections_for_vpc(ec2, "vpc-target")

    assert deleted == []
    assert ec2.deleted == []


# --- delete_with_retry: service-VM-provisioning rejection (common/errors.py) ---

AWS_COMMON_SCRIPTS = ISVCTL_ROOT / "configs" / "providers" / "aws" / "scripts" / "common"
ZCOMPUTE_COMMON_SCRIPTS = ISVCTL_ROOT / "configs" / "providers" / "zcompute" / "scripts" / "common"

_PROVISIONING_MESSAGE = "Cannot delete vpc x while its service VM is being provisioned"


def _load_common_script(base: Path, script_name: str) -> ModuleType:
    """Load a provider common script (e.g. errors.py) as a module."""
    script_path = base / script_name
    module_name = f"test_common_{base.parents[1].name}_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeClock:
    """Replaces time.sleep/time.monotonic so retry waits run instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize("base", [AWS_COMMON_SCRIPTS, ZCOMPUTE_COMMON_SCRIPTS], ids=["aws", "zcompute"])
def test_delete_with_retry_waits_out_service_vm_provisioning(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provisioning rejection clears once the service VM is up - keep retrying until then."""
    errors_mod = _load_common_script(base, "errors.py")
    clock = _FakeClock()
    monkeypatch.setattr(errors_mod.time, "sleep", clock.sleep)
    monkeypatch.setattr(errors_mod.time, "monotonic", clock.monotonic)

    calls = {"n": 0}

    def fake_delete(**_kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _client_error("DeleteVpc", code="InvalidParameterValue", message=_PROVISIONING_MESSAGE)

    assert errors_mod.delete_with_retry(fake_delete, VpcId="vpc-x", resource_desc="VPC vpc-x") is True
    assert calls["n"] == 3
    # The provisioning waits must not burn the regular (transient) attempt budget.
    assert len(clock.sleeps) == 2


@pytest.mark.parametrize("base", [AWS_COMMON_SCRIPTS, ZCOMPUTE_COMMON_SCRIPTS], ids=["aws", "zcompute"])
def test_delete_with_retry_provisioning_timeout_gives_up(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A service VM stuck in provisioning must not retry forever."""
    errors_mod = _load_common_script(base, "errors.py")
    clock = _FakeClock()
    monkeypatch.setattr(errors_mod.time, "sleep", clock.sleep)
    monkeypatch.setattr(errors_mod.time, "monotonic", clock.monotonic)

    def always_provisioning(**_kwargs: Any) -> None:
        raise _client_error("DeleteVpc", code="InvalidParameterValue", message=_PROVISIONING_MESSAGE)

    assert errors_mod.delete_with_retry(always_provisioning, VpcId="vpc-x", resource_desc="VPC vpc-x") is False
    assert clock.now >= errors_mod.SERVICE_VM_PROVISIONING_TIMEOUT_SECONDS


@pytest.mark.parametrize("base", [AWS_COMMON_SCRIPTS, ZCOMPUTE_COMMON_SCRIPTS], ids=["aws", "zcompute"])
def test_delete_with_retry_other_invalid_parameter_not_retried(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generic InvalidParameterValue (e.g. 'Resources still exist') stays non-retryable."""
    errors_mod = _load_common_script(base, "errors.py")
    clock = _FakeClock()
    monkeypatch.setattr(errors_mod.time, "sleep", clock.sleep)
    monkeypatch.setattr(errors_mod.time, "monotonic", clock.monotonic)

    calls = {"n": 0}

    def resources_exist(**_kwargs: Any) -> None:
        calls["n"] += 1
        raise _client_error(
            "DeleteVpc", code="InvalidParameterValue", message="Resources still exist for vpc x : networks"
        )

    assert errors_mod.delete_with_retry(resources_exist, VpcId="vpc-x", resource_desc="VPC vpc-x") is False
    assert calls["n"] == 1
    assert clock.sleeps == []


def test_teardown_delete_with_retry_waits_out_service_vm_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """teardown's local delete_with_retry must also wait out the provisioning rejection,
    or an early teardown (test phase failed fast) leaks the VPC."""
    module = _load_network_script("teardown.py")
    clock = _FakeClock()
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)

    calls = {"n": 0}

    def fake_delete(**_kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _client_error("DeleteVpc", code="InvalidParameterValue", message=_PROVISIONING_MESSAGE)

    assert module.delete_with_retry(fake_delete, "vpc", VpcId="vpc-x") is True
    assert calls["n"] == 3


@pytest.mark.parametrize("base", [AWS_COMMON_SCRIPTS, ZCOMPUTE_COMMON_SCRIPTS], ids=["aws", "zcompute"])
def test_provisioning_knobs_are_env_overridable(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCOMPUTE_SERVICE_VM_PROVISIONING_RETRY_SECONDS", "1")
    monkeypatch.setenv("ZCOMPUTE_SERVICE_VM_PROVISIONING_TIMEOUT_SECONDS", "7")
    errors_mod = _load_common_script(base, "errors.py")
    assert errors_mod.SERVICE_VM_PROVISIONING_RETRY_SECONDS == 1.0
    assert errors_mod.SERVICE_VM_PROVISIONING_TIMEOUT_SECONDS == 7.0
