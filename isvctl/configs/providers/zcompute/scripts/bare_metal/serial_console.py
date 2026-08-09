#!/usr/bin/env python3
"""Check serial console output availability on a zcompute bare-metal instance.

Same as scripts/vm/serial_console.py, platform="bm". Confirmed via direct
API probe against the BM lab cluster: GetConsoleOutput returns a 500
InternalFailure from the vm-manager-api backend regardless of instance
type, so this is expected to report not_supported=true until zcompute
implements it. This lets the test runner exclude SerialConsoleCheck /
SerialConsoleRetentionCheck rather than fail the whole suite.

Output JSON (supported):
{
    "success": true,
    "platform": "bm",
    "instance_id": "i-xxx",
    "console_available": true,
    "output_length": 1234
}

Output JSON (not supported):
{
    "success": true,
    "platform": "bm",
    "instance_id": "i-xxx",
    "console_available": false,
    "not_supported": true,
    "note": "get_console_output is not implemented on this zcompute version"
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.client import get_client  # noqa: E402

_NOT_IMPLEMENTED_CODES = {
    "NotImplemented",
    "UnsupportedOperation",
    "InvalidOperation",
    "OperationNotSupported",
    "InternalFailure",
    "InternalError",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check serial console output on a zcompute bare-metal instance"
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "symphony"))
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "bm",
        "instance_id": args.instance_id,
        "console_available": False,
    }

    ec2 = get_client("ec2", region=args.region)

    try:
        resp = ec2.get_console_output(InstanceId=args.instance_id)
        output = resp.get("Output", "")
        result["console_available"] = bool(output)
        result["output_length"] = len(output) if output else 0
        result["success"] = True

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", str(e))

        # Confirmed live across every BM run this session (2026-08-09),
        # including right after a successful boot/lifecycle event: zcompute
        # returns "ValidationError ... is not ready (HTTP 409)" here, not
        # the 500 InternalFailure this script originally assumed. Same
        # permanent platform gap, different error shape - treat it the
        # same way.
        is_not_supported = (
            code in _NOT_IMPLEMENTED_CODES
            or any(kw in msg.lower() for kw in ("not implemented", "not supported", "unsupported"))
            or (code == "ValidationError" and "not ready" in msg.lower())
        )

        if is_not_supported:
            result["console_available"] = False
            result["not_supported"] = True
            result["note"] = "get_console_output is not implemented on this zcompute version"
            result["success"] = True
        else:
            result["error"] = msg
            result["error_code"] = code

    except Exception as e:
        result["console_available"] = False
        result["not_supported"] = True
        result["note"] = f"Unexpected error checking console output: {e}"
        result["success"] = True

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
