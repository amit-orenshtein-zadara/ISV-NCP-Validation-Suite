#!/usr/bin/env python3
"""Install/verify GPU dependencies on a zcompute bare-metal instance, with retries.

zcompute bare-metal does NOT persist post-boot filesystem changes across
stop/start/reboot/power-cycle (confirmed live 2026-08-09: dpkg had zero record
of docker-ce ever being installed after a power-cycle, despite it being
manually confirmed working immediately beforehand). power_cycle_instance.py is
the last lifecycle event before the stress/GPU/NCCL/driver/NIM checks run, so
this step runs immediately after it and before describe_instance/deploy_nim -
retrying setup_gpu_dependencies() (idempotent - safe to re-run) against a real
functional verification (gpu_docker_runtime_ok) rather than trusting bare
install-command exit codes, since apt/network/registry operations here are
known to be occasionally flaky (broken Mellanox DOCA repo, transient pulls).

This step never fails the pipeline (Amit, 2026-08-10: "no need to hard fail
anything ... after all dependencies are installed with retries -> continue
with the tests") - it always exits 0 and always continues to
describe_instance/deploy_nim/the GPU test wave regardless of outcome. Image
pre-pulls (NIM, GpuStressCheck's pytorch image, NcclCheck's hpc-benchmarks
image) stay single best-effort attempts, unchanged from before - each
downstream check/step re-pulls itself if its pre-pull didn't happen.

Output JSON:
{
    "success": true,
    "platform": "bm",
    "private_ip": "172.28.x.x",
    "attempts": 1,
    "gpu_deps": {"docker": true, "nvidia_container_toolkit": true, "cuda_toolkit": true, "nvidia_smi_accessible": true, "gpu_docker_runtime_ok": true},
    "gpu_deps_verified": true,
    "nim_image_prepulled": true,
    "gpu_stress_image_prepulled": true,
    "nccl_image_prepulled": true
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.ec2 import log, pull_docker_image, pull_nim_image, setup_gpu_dependencies  # noqa: E402

_REQUIRED_KEYS = ("docker", "nvidia_container_toolkit", "cuda_toolkit", "gpu_docker_runtime_ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install/verify GPU dependencies on a zcompute bare-metal instance, with retries"
    )
    parser.add_argument("--private-ip", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument(
        "--ngc-api-key",
        default=os.environ.get("NGC_API_KEY", "") or os.environ.get("NGC_NIM_API_KEY", ""),
        help="NGC API key for pre-pulling the NIM image (same env fallback as power_cycle_instance.py)",
    )
    parser.add_argument(
        "--nim-model",
        default="meta/llama-3.2-1b-instruct",
        help="NIM model to pre-pull after GPU deps verify, matching deploy_nim.py's default",
    )
    parser.add_argument("--nim-tag", default="latest", help="NIM container image tag to pre-pull")
    parser.add_argument(
        "--gpu-stress-image",
        default="nvcr.io/nvidia/pytorch:25.04-py3",
        help="Image to pre-pull for GpuStressCheck, matching its class default",
    )
    parser.add_argument(
        "--nccl-image",
        default="nvcr.io/nvidia/hpc-benchmarks:25.04",
        help="Image to pre-pull for NcclCheck, matching its class default",
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Max setup_gpu_dependencies() attempts")
    parser.add_argument(
        "--retry-interval",
        type=int,
        default=60,
        help="Seconds to wait between failed attempts (default: 60)",
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": True,
        "platform": "bm",
        "private_ip": args.private_ip,
        "attempts": 0,
        "gpu_deps": {},
        "gpu_deps_verified": False,
        "nim_image_prepulled": False,
        "gpu_stress_image_prepulled": False,
        "nccl_image_prepulled": False,
    }

    try:
        for attempt in range(1, args.max_attempts + 1):
            result["attempts"] = attempt
            log(f"[install-deps] attempt {attempt}/{args.max_attempts}: installing/verifying GPU dependencies ...")
            gpu_deps = setup_gpu_dependencies(args.private_ip, args.ssh_user, args.key_file)
            result["gpu_deps"] = gpu_deps

            verified = all(gpu_deps.get(key) for key in _REQUIRED_KEYS)
            if verified:
                result["gpu_deps_verified"] = True
                log(f"[install-deps] verified working on attempt {attempt}/{args.max_attempts}: {gpu_deps}")
                break

            missing = [key for key in _REQUIRED_KEYS if not gpu_deps.get(key)]
            log(f"[install-deps] attempt {attempt}/{args.max_attempts} not fully verified, missing: {missing}")
            if attempt < args.max_attempts:
                log(f"[install-deps] retrying in {args.retry_interval}s ...")
                time.sleep(args.retry_interval)

        if not result["gpu_deps_verified"]:
            log(
                f"[install-deps] WARNING: GPU dependencies never fully verified after "
                f"{args.max_attempts} attempts - continuing anyway (non-fatal, downstream "
                f"checks will surface the real failure if this is genuinely broken)"
            )

        result["nim_image_prepulled"] = pull_nim_image(
            args.private_ip,
            args.ssh_user,
            args.key_file,
            args.ngc_api_key,
            model=args.nim_model,
            tag=args.nim_tag,
        )

        # GpuStressCheck/NcclCheck cold-pull these every lifecycle cycle
        # otherwise (confirmed live 2026-08-10: ~13min pull against
        # GpuStressCheck's 900s budget, cutting it close) - same public
        # nvcr.io/nvidia/* warmup as the smoke test above, no login needed.
        result["gpu_stress_image_prepulled"] = pull_docker_image(
            args.private_ip, args.ssh_user, args.key_file, args.gpu_stress_image
        )
        result["nccl_image_prepulled"] = pull_docker_image(
            args.private_ip, args.ssh_user, args.key_file, args.nccl_image
        )

    except Exception as e:
        log(f"[install-deps] WARNING: unexpected error (non-fatal, continuing): {e}")
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
