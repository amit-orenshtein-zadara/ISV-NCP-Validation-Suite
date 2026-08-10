#!/usr/bin/env python3
"""Power-cycle a zcompute bare-metal instance (hard stop + start).

Unlike reboot (OS-level restart), this performs a full hardware
power-cycle: force-stop the instance, wait for 'stopped', then start it
and wait for recovery. Exercises firmware init, BIOS POST, and a cold OS
boot — validating recovery from complete power loss.

zcompute-specific notes:
  - stop_instances(Force=True) is UNVERIFIED on zcompute as of this
    writing — our API probe only exercised Force=True's read-only
    surface, not a real force-stop against bare-metal hardware (no BM
    instance type existed yet to test against). If the service silently
    ignores the flag, this degrades to an ordinary graceful stop, which
    still exercises most of what InstancePowerCycleCheck cares about.
  - Same GPU-resource-release retry loop as start_instance.py.
  - Does NOT reinstall GPU dependencies (Docker, CUDA, NIM image) itself -
    that's install_gpu_dependencies.py, a dedicated step that runs
    immediately after this one (with retries + real verification) since
    zcompute bare-metal does NOT persist post-boot filesystem changes
    across stop/start/reboot/power-cycle (confirmed live 2026-08-09: dpkg
    had zero record of a manually-verified-working Docker install after a
    power-cycle - only what's baked into the base AMI survives).

Output JSON:
{
    "success": true,
    "platform": "bm",
    "instance_id": "i-xxx",
    "state": "running",
    "public_ip": "172.28.x.x",
    "key_file": "/tmp/isv-bm-test-key.pem",
    "power_cycle_initiated": true,
    "power_was_off": true,
    "time_to_stopped_seconds": 842.3,
    "ssh_ready": true,
    "recovery_seconds": 900,
    "nvidia_modules_loaded": true
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.client import get_client  # noqa: E402
from common.ec2 import (  # noqa: E402
    load_nvidia_modules,
    log,
    poll_instance_state,
    wait_for_private_ip,
    wait_for_public_ip,
)
from common.ssh_utils import wait_for_ssh  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Power-cycle a zcompute bare-metal instance")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "symphony"))
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument(
        "--pre-start-delay", type=int, default=600,
        help="Seconds to wait after power-off before issuing start (default: 600)",
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "bm",
        "instance_id": args.instance_id,
        "state": "",
        "region": args.region,
        "key_file": args.key_file,
        "ssh_user": args.ssh_user,
        "power_cycle_initiated": False,
        "power_was_off": False,
        "time_to_stopped_seconds": None,
        "ssh_ready": False,
        "recovery_seconds": None,
    }

    ec2 = get_client("ec2", region=args.region)

    try:
        log("[power-cycle] verifying instance is running before power-cycle ...")
        resp = ec2.describe_instances(InstanceIds=[args.instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        current_state = inst["State"]["Name"]

        if current_state != "running":
            result["error"] = f"Instance is {current_state}, expected running"
            result["state"] = current_state
            print(json.dumps(result, indent=2))
            return 1

        start_time = time.monotonic()

        log(f"[power-cycle] force-stopping instance {args.instance_id} ...")
        try:
            ec2.stop_instances(InstanceIds=[args.instance_id], Force=True)
        except ClientError as e:
            log(
                f"[power-cycle] Force=True rejected ({e.response.get('Error', {}).get('Code')}), "
                "falling back to a plain stop_instances call"
            )
            ec2.stop_instances(InstanceIds=[args.instance_id])
        result["power_cycle_initiated"] = True

        # Bare-metal hardware power-down (BMC/IPMI) can take far longer than
        # the ~30 min originally assumed here - bumped generously (Aviv,
        # 2026-08-06: "be very kind with your timeouts, bump it up").
        stop_start = time.monotonic()
        poll_instance_state(ec2, args.instance_id, ["stopped"], timeout=7200, interval=30)
        time_to_stopped = round(time.monotonic() - stop_start, 1)
        result["power_was_off"] = True
        result["time_to_stopped_seconds"] = time_to_stopped
        log(f"[power-cycle] instance stopped (powered off) {time_to_stopped}s after stop was issued")

        if args.pre_start_delay > 0:
            log(f"[power-cycle] waiting {args.pre_start_delay}s for GPU resource release ...")
            time.sleep(args.pre_start_delay)

        # Retry start_instances on a 4-hour wall-clock budget instead of a
        # fixed attempt count - same rationale/fix as start_instance.py
        # (Aviv/Amit, 2026-08-06): real GB200 hardware can reject
        # start_instances outright for well longer than a handful of
        # 300s-apart retries allowed for.
        RETRY_BUDGET_SECONDS = 14400  # 4 hours total
        RESOURCE_WAIT_SECONDS = 300
        final_state = "stopped"

        retry_deadline = time.monotonic() + RETRY_BUDGET_SECONDS
        attempt = 0
        while True:
            attempt += 1
            remaining = retry_deadline - time.monotonic()
            log(
                f"[power-cycle] starting instance (cold boot, attempt {attempt}, "
                f"~{remaining / 60:.0f}m left in retry budget) ..."
            )
            try:
                ec2.start_instances(InstanceIds=[args.instance_id])
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                log(f"[power-cycle] start_instances call failed ({code}): {e}")
                if time.monotonic() + RESOURCE_WAIT_SECONDS >= retry_deadline:
                    log("[power-cycle] exhausted 4-hour retry budget; instance could not start.")
                    break
                log(f"[power-cycle] retrying in {RESOURCE_WAIT_SECONDS}s ...")
                time.sleep(RESOURCE_WAIT_SECONDS)
                continue

            final_state = poll_instance_state(
                ec2, args.instance_id, ["running", "stopped"], timeout=2700, interval=30
            )
            if final_state == "running":
                log("[power-cycle] instance is running.")
                break

            if time.monotonic() + RESOURCE_WAIT_SECONDS >= retry_deadline:
                log("[power-cycle] exhausted 4-hour retry budget; instance could not start.")
                break
            log(
                f"[power-cycle] instance returned to stopped; waiting "
                f"{RESOURCE_WAIT_SECONDS}s before retry ..."
            )
            time.sleep(RESOURCE_WAIT_SECONDS)

        result["state"] = final_state

        # PrivateIpAddress can still read back empty for a while right after
        # the instance transitions to 'running' - a single un-retried
        # describe_instances call here used to capture it before zcompute
        # finished attaching the network interface, silently sending every
        # downstream SSH call at an empty host (confirmed live 2026-08-09:
        # 160 "Could not resolve hostname" attempts before the whole step
        # failed). Retry for up to 10 minutes (Amit, 2026-08-09) and hard-
        # stop here - never fall through to SSH (public or private) without
        # a real private_ip in hand.
        private_ip = wait_for_private_ip(ec2, args.instance_id, timeout=600, interval=10)
        if not private_ip:
            result["error"] = f"PrivateIpAddress not available after power-cycle for {args.instance_id}"
            log(f"[power-cycle] result so far: {json.dumps(result, indent=2)}")
            print(json.dumps(result, indent=2))
            return 1
        result["private_ip"] = private_ip
        log(f"[power-cycle] private_ip obtained, result so far: {json.dumps(result, indent=2)}")

        resp = ec2.describe_instances(InstanceIds=[args.instance_id])
        inst = resp["Reservations"][0]["Instances"][0]

        # public_ip is informational only, not a gate - the EIP is confirmed
        # unreachable from this run station anyway (2026-07-29), private_ip
        # is the only thing actually used for SSH below. Confirmed live
        # 2026-08-03: this used to hard-fail here if the EIP hadn't
        # reappeared yet, even though the instance was already fully
        # reachable via private_ip at that exact moment (SSH/GPU checks
        # succeeded against it moments later in the same run). Don't repeat
        # that mistake - wait a bit, log what happened, but always proceed
        # to the real check (SSH) regardless of how this turns out.
        fresh_ip = inst.get("PublicIpAddress") or wait_for_public_ip(ec2, args.instance_id, timeout=120, interval=5)
        if fresh_ip and fresh_ip not in ("", "None"):
            result["public_ip"] = fresh_ip
        else:
            log(
                "[power-cycle] no public IP yet after power-cycle - continuing anyway, "
                "private_ip is what actually matters"
            )

        log("[power-cycle] waiting for SSH to be ready ...")
        ssh_ready = wait_for_ssh(result["private_ip"], args.ssh_user, args.key_file, max_attempts=80, interval=15)
        if not ssh_ready:
            # Retry once more before giving up rather than failing on a
            # single SSH-wait window (Aviv, 2026-08-03).
            log("[power-cycle] SSH did not respond; retrying SSH wait once more ...")
            time.sleep(60)
            ssh_ready = wait_for_ssh(result["private_ip"], args.ssh_user, args.key_file, max_attempts=80, interval=15)
        result["ssh_ready"] = ssh_ready
        result["recovery_seconds"] = int(time.monotonic() - start_time)

        if not ssh_ready:
            result["error"] = "SSH not ready after power-cycle"
            print(json.dumps(result, indent=2))
            return 1

        nvidia_ok = load_nvidia_modules(result["private_ip"], args.ssh_user, args.key_file)
        result["nvidia_modules_loaded"] = nvidia_ok

        result["success"] = final_state == "running"
        log(f"[power-cycle] completed successfully! (recovery={result['recovery_seconds']}s)")

    except ClientError as e:
        result["error"] = str(e)
        result["error_code"] = e.response.get("Error", {}).get("Code", "")
    except TimeoutError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
