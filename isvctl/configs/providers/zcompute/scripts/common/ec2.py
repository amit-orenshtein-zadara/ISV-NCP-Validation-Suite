#!/usr/bin/env python3
"""EC2 helper utilities for zcompute.

zcompute-specific notes:
  - No boto3 waiters — ec2.get_waiter() will fail. Use custom polling.
  - VPC starts in 'pending' state — poll until 'available'.
  - No auto public IP — must allocate_address + associate_address.
  - PublicIpAddress may be empty string, "None", or None at launch.
  - StartInstances goes: stopped -> pending -> stopped -> pending -> running.
  - Root device is /dev/vda (not /dev/sda).
  - NVIDIA modules not auto-loaded — must modprobe after SSH.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from botocore.exceptions import ClientError


def log(msg: str) -> None:
    """Print a stderr diagnostic line prefixed with a wall-clock timestamp.

    Bare-metal lifecycle operations (stop/start/reboot/power-cycle) can run
    for hours; without a per-line timestamp, correlating a given [poll]/
    [start]/[stop] line to when it actually happened means cross-referencing
    the outer isvctl orchestrator's own log (Aviv/Amit, 2026-08-06).
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)

# If set, load_nvidia_modules/setup_gpu_dependencies use password auth via
# sshpass instead of key_file - same override as ssh_utils.py's
# wait_for_ssh/run_ssh_command, isvtest.core.ssh.get_ssh_client, and
# shared/deploy_nim.py/teardown_nim.py. Never hardcode the password anywhere;
# export it locally in your own shell only.
_SSH_PASSWORD_ENV_VAR = "ISVTEST_SSH_PASSWORD"


def _ssh_command_prefix(user: str, host: str, key_file: str) -> list[str]:
    """Build the `ssh`/`sshpass ssh` argument prefix up to (but not including) the remote command."""
    password = os.environ.get(_SSH_PASSWORD_ENV_VAR)
    if password:
        if not shutil.which("sshpass"):
            raise RuntimeError(
                f"{_SSH_PASSWORD_ENV_VAR} is set but 'sshpass' is not installed. "
                "Install it (e.g. `sudo apt-get install -y sshpass`) or unset "
                f"{_SSH_PASSWORD_ENV_VAR} to use key-based auth instead."
            )
        return [
            "sshpass", "-p", password, "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=no",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            f"{user}@{host}",
        ]
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-i", key_file, f"{user}@{host}",
    ]


def poll_instance_state(
    ec2: Any,
    instance_id: str,
    target_states: list[str],
    timeout: int = 600,
    interval: int = 15,
) -> str:
    """Poll until instance state is in target_states.

    For StartInstances on zcompute the sequence is:
      stopped -> pending -> stopped -> pending -> running
    so we never give up early on 'stopped' when 'running' is the target.

    Args:
        ec2:           Boto3 EC2 client.
        instance_id:   EC2 instance ID.
        target_states: List of acceptable terminal states (e.g. ['running']).
        timeout:       Maximum seconds to wait.
        interval:      Polling interval in seconds.

    Returns:
        Final instance state string.

    Raises:
        TimeoutError: If the instance does not reach a target state in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        log(f"[poll] instance {instance_id} state: {state}")
        if state in target_states:
            return state
        time.sleep(interval)

    raise TimeoutError(
        f"Instance {instance_id} did not reach {target_states} within {timeout}s"
    )


def wait_for_public_ip(
    ec2: Any,
    instance_id: str,
    timeout: int = 120,
    interval: int = 5,
) -> str | None:
    """Poll describe_instances until PublicIpAddress is non-empty/non-None.

    zcompute requires a manual EIP allocation/association — the public IP is
    not assigned automatically. Call allocate_and_associate_eip first, then
    this helper to confirm the IP is visible.

    Args:
        ec2:         Boto3 EC2 client.
        instance_id: EC2 instance ID.
        timeout:     Maximum seconds to wait.
        interval:    Polling interval in seconds.

    Returns:
        Public IP string, or None if not available within timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        ip = inst.get("PublicIpAddress")
        if ip and ip not in ("", "None"):
            return ip
        log(f"[poll] waiting for public IP on {instance_id} ...")
        time.sleep(interval)
    return None


def wait_for_private_ip(
    ec2: Any,
    instance_id: str,
    timeout: int = 60,
    interval: int = 5,
) -> str | None:
    """Poll describe_instances until PrivateIpAddress is non-empty/non-None.

    Confirmed live (2026-08-09): PrivateIpAddress can still read back empty
    for several seconds right after an instance transitions to 'running' -
    a single un-retried describe_instances call right at that transition
    can capture the field before zcompute finishes attaching the network
    interface, silently leaving every downstream SSH call targeting an
    empty host.

    Args:
        ec2:         Boto3 EC2 client.
        instance_id: EC2 instance ID.
        timeout:     Maximum seconds to wait.
        interval:    Polling interval in seconds.

    Returns:
        Private IP string, or None if not available within timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        ip = inst.get("PrivateIpAddress")
        if ip and ip not in ("", "None"):
            return ip
        # Log the instance's actual network-interface state on each attempt
        # (Aviv, 2026-08-09: suspected root cause is the network interface
        # itself not attaching yet) - a bare "waiting" message gives no way
        # to tell a missing NIC apart from a NIC that's attached but just
        # hasn't reported an IP yet.
        nics = inst.get("NetworkInterfaces", [])
        log(
            f"[poll] waiting for private IP on {instance_id} "
            f"(state={inst.get('State', {}).get('Name')}, network_interfaces={len(nics)}) ..."
        )
        time.sleep(interval)
    return None


def _is_valid_private_key(key_file: str) -> bool:
    """Check whether a file on disk is a loadable SSH private key.

    A PEM file can exist on disk but be unusable (truncated by a bad
    copy/paste, accidentally overwritten, etc.) — existence alone doesn't
    mean it will actually authenticate.
    """
    try:
        result = subprocess.run(
            ["ssh-keygen", "-y", "-f", key_file],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def create_key_pair(
    ec2: Any,
    key_name: str,
    key_dir: str | None = None,
) -> str:
    """Create a key pair and save the PEM to disk.

    Idempotent:
      - If the key pair exists in EC2 AND the local PEM file exists, reuse both.
      - If the key pair exists in EC2 but no local PEM, delete and recreate.
      - If the key pair does not exist, create fresh.

    Args:
        ec2:      Boto3 EC2 client.
        key_name: Name of the key pair.
        key_dir:  Directory to save the PEM file. Defaults to /tmp.

    Returns:
        Absolute path to the saved PEM file.
    """
    if key_dir is None:
        key_dir = "/tmp"

    key_file = os.path.join(key_dir, f"{key_name}.pem")

    # Check whether the key pair already exists in EC2.
    exists_in_ec2 = False
    try:
        resp = ec2.describe_key_pairs(KeyNames=[key_name])
        if resp.get("KeyPairs"):
            exists_in_ec2 = True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "InvalidKeyPair.NotFound":
            exists_in_ec2 = False
        else:
            raise

    if exists_in_ec2:
        if os.path.exists(key_file) and _is_valid_private_key(key_file):
            print(
                f"[ec2] reusing existing key pair '{key_name}' and PEM {key_file}",
                file=sys.stderr,
            )
            return key_file
        else:
            # PEM is missing, or present but corrupted (e.g. a bad copy/paste
            # left a truncated file on disk — confirmed to happen in practice,
            # 2026-08-03) — either way, delete the key pair so we can recreate
            # it with new, actually-usable material.
            reason = "PEM not found locally" if not os.path.exists(key_file) else "local PEM is corrupted/unusable"
            print(
                f"[ec2] key pair '{key_name}' exists in EC2 but {reason}; deleting and recreating.",
                file=sys.stderr,
            )
            ec2.delete_key_pair(KeyName=key_name)

    resp = ec2.create_key_pair(KeyName=key_name)
    pem_material = resp["KeyMaterial"]
    os.makedirs(key_dir, exist_ok=True)
    with open(key_file, "w") as fh:
        fh.write(pem_material)
    os.chmod(key_file, 0o600)
    print(f"[ec2] created key pair '{key_name}', saved to {key_file}", file=sys.stderr)
    return key_file


def create_security_group(
    ec2: Any,
    vpc_id: str,
    name: str,
    description: str = "ISV NCP validation",
) -> str:
    """Create a security group with SSH ingress, or reuse an existing one.

    Idempotent — if a SG with the same name already exists in the VPC, it
    is returned without modification.

    Args:
        ec2:         Boto3 EC2 client.
        vpc_id:      VPC ID in which to create the SG.
        name:        Security group name.
        description: Human-readable description.

    Returns:
        Security group ID (sg-xxx).
    """
    # Check whether a SG with this name already exists in the VPC.
    try:
        resp = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )
        if resp.get("SecurityGroups"):
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(
                f"[ec2] reusing existing security group '{name}' ({sg_id})",
                file=sys.stderr,
            )
            return sg_id
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code != "InvalidGroup.NotFound":
            raise

    # Create the security group.
    resp = ec2.create_security_group(
        GroupName=name,
        Description=description,
        VpcId=vpc_id,
    )
    sg_id = resp["GroupId"]
    print(f"[ec2] created security group '{name}' ({sg_id})", file=sys.stderr)

    # Authorize SSH ingress from anywhere.
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
            }
        ],
    )
    print(f"[ec2] authorized SSH ingress on {sg_id}", file=sys.stderr)
    return sg_id


def allocate_and_associate_eip(
    ec2: Any,
    instance_id: str,
) -> tuple[str, str]:
    """Allocate an Elastic IP and associate it with an instance.

    zcompute does not assign public IPs automatically — this step is
    required to make the instance reachable.

    Args:
        ec2:         Boto3 EC2 client.
        instance_id: EC2 instance ID.

    Returns:
        Tuple of (allocation_id, public_ip).
    """
    alloc_resp = ec2.allocate_address(Domain="vpc")
    allocation_id = alloc_resp["AllocationId"]
    public_ip = alloc_resp["PublicIp"]
    print(
        f"[ec2] allocated EIP {public_ip} ({allocation_id})",
        file=sys.stderr,
    )

    ec2.associate_address(InstanceId=instance_id, AllocationId=allocation_id)
    print(
        f"[ec2] associated EIP {public_ip} with {instance_id}",
        file=sys.stderr,
    )
    return allocation_id, public_ip


def load_nvidia_modules(host: str, user: str, key_file: str) -> bool:
    """SSH into instance and load NVIDIA kernel modules.

    zcompute does not auto-load NVIDIA modules at boot; they must be
    loaded explicitly with modprobe.

    Args:
        host:     IP or hostname of the instance.
        user:     SSH username (e.g. 'ubuntu').
        key_file: Path to the private key PEM file.

    Returns:
        True if modprobe succeeded (including 'already loaded'), False otherwise.
    """
    def _ssh(command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*_ssh_command_prefix(user, host, key_file), command],
            capture_output=True, text=True, timeout=timeout,
        )

    log(f"[ec2] loading NVIDIA modules on {host} ...")
    # modprobe loads the kernel modules; nvidia-modprobe -u creates the
    # /dev/nvidia-uvm device node that CUDA (and NIM) require.
    # Without the device node, nvidia-smi works but any CUDA app inside
    # a container fails with CUDA_ERROR_SYSTEM_NOT_READY.
    result = _ssh(
        "sudo modprobe nvidia nvidia-uvm nvidia-modeset && "
        "sudo nvidia-modprobe -u -c=0 2>/dev/null || true"
    )
    loaded = result.returncode == 0 or "already" in result.stderr.lower()

    if not loaded and "not found" in result.stderr.lower():
        # Module not built for the current kernel (kernel upgraded since AMI was built).
        # Search for a pre-built linux-modules-nvidia-*-server-<kernel> package and install it.
        log("[ec2] module not found — searching for pre-built NVIDIA kernel modules ...")
        install_cmd = (
            "KERNEL=$(uname -r) && "
            "PKG=$(apt-cache search \"linux-modules-nvidia.*${KERNEL}\" 2>/dev/null | head -1 | awk '{print $1}') && "
            "if [ -n \"$PKG\" ]; then "
            "  echo \"[ec2] installing $PKG\" && "
            "  sudo apt-get install -y --no-install-recommends $PKG 2>&1 | tail -3; "
            "else "
            "  echo \"[ec2] no pre-built module package found, trying dkms autoinstall\" && "
            "  sudo dkms autoinstall 2>&1 | tail -5; "
            "fi"
        )
        install_result = _ssh(install_cmd, timeout=600)
        log(f"[ec2] install result: {install_result.stdout.strip()}")
        result = _ssh("sudo modprobe nvidia nvidia-uvm nvidia-modeset")
        loaded = result.returncode == 0 or "already" in result.stderr.lower()

    if loaded:
        log("[ec2] NVIDIA modules loaded successfully")
    else:
        log(f"[ec2] modprobe failed (rc={result.returncode}): {result.stderr.strip()}")
        return False

    # Locate nvidia-smi and symlink it to /usr/local/bin which is in PATH
    # for all session types including paramiko non-interactive sessions.
    locate_and_link = (
        "NVSMI=$(find /usr /opt -name nvidia-smi -type f 2>/dev/null | head -1); "
        "echo \"[nvidia] found nvidia-smi at: $NVSMI\"; "
        "if [ -n \"$NVSMI\" ]; then "
        "  sudo ln -sf \"$NVSMI\" /usr/local/bin/nvidia-smi && "
        "  echo \"[nvidia] symlinked $NVSMI -> /usr/local/bin/nvidia-smi\"; "
        "else "
        "  DRVER=$(dpkg -l | awk '/nvidia-kernel-common-[0-9]/{match($2,/[0-9]+/,m);print m[0];exit}'); "
        "  DRVER=${DRVER:-535}; "
        "  echo \"[nvidia] nvidia-smi not found, installing nvidia-utils-${DRVER}-server\"; "
        "  sudo apt-get install -y --no-install-recommends nvidia-utils-${DRVER}-server 2>&1 | tail -3; "
        "  sudo ln -sf /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi 2>/dev/null || true; "
        "fi"
    )
    r = _ssh(locate_and_link, timeout=120)
    log(f"[ec2] nvidia-smi setup: {r.stdout.strip()}")

    # Wait until nvidia-smi actually communicates with the driver.
    # After modprobe the kernel module can take a few seconds to fully
    # initialize before nvidia-smi can query GPUs successfully.
    wait_cmd = (
        "for i in $(seq 1 15); do "
        "  GPU=$(/usr/local/bin/nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); "
        "  if [ -n \"$GPU\" ]; then "
        "    echo \"[nvidia] driver ready after $((i-1)) retries: $GPU\"; exit 0; "
        "  fi; "
        "  echo \"[nvidia] driver not ready yet (attempt $i/15), waiting 3s...\"; "
        "  sleep 3; "
        "done; "
        "echo \"[nvidia] WARNING: driver did not become ready after 45s\"; exit 1"
    )
    r = _ssh(wait_cmd, timeout=90)
    log(f"[ec2] nvidia-smi wait: {r.stdout.strip()}")
    if r.returncode != 0:
        log("[ec2] warning: nvidia-smi did not report GPUs — driver may not be loaded")

    # Persist modules across reboots by adding to /etc/modules.
    # This ensures nvidia-smi is available immediately after every boot
    # without requiring a manual modprobe step.
    persist_cmd = (
        "sudo sh -c '"
        "grep -q nvidia /etc/modules || "
        "printf \"nvidia\\nnvidia-uvm\\nnvidia-modeset\\n\" >> /etc/modules'"
    )
    persist_result = _ssh(persist_cmd)
    if persist_result.returncode == 0:
        log("[ec2] NVIDIA modules persisted in /etc/modules")
    else:
        log(
            f"[ec2] warning: could not persist modules to /etc/modules: "
            f"{persist_result.stderr.strip()}"
        )

    # Restart Docker so the NVIDIA container runtime picks up the freshly
    # loaded kernel modules. Without this restart, `docker run --gpus all`
    # fails with "could not select device driver" after stop/start or reboot.
    r = _ssh("sudo systemctl restart docker 2>/dev/null || true", timeout=30)
    log("[ec2] Docker restarted to pick up NVIDIA modules")

    return True


def setup_gpu_dependencies(host: str, user: str, key_file: str) -> dict[str, bool]:
    """Install Docker, NVIDIA Container Toolkit, and CUDA toolkit via SSH.

    Run once at VM launch time. Takes ~10-15 min on first launch.
    For production, bake these into the base image instead.

    Returns a dict of {component: success} for each install step.
    """

    def _ssh(command: str, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*_ssh_command_prefix(user, host, key_file), command],
            capture_output=True, text=True, timeout=timeout,
        )

    results: dict[str, bool] = {
        "docker": False,
        "nvidia_container_toolkit": False,
        "cuda_toolkit": False,
        "nvidia_smi_accessible": False,
    }

    # ── 0. Fix any pre-existing NVML mismatch BEFORE touching apt ────────────
    # The AMI may already have a kernel module / userspace library mismatch.
    # Fix it immediately using the loaded kernel module version as source of truth,
    # before any other apt operations can make things worse.
    print("[setup] pre-flight: fixing nvidia-utils to match loaded kernel module ...", file=sys.stderr)
    preflight = _ssh(
        "KMOD_VER=$(cat /sys/module/nvidia/version 2>/dev/null || echo '') && "
        "KMOD_MAJOR=$(echo \"$KMOD_VER\" | cut -d. -f1) && "
        "echo \"[setup] KMOD_VER=${KMOD_VER:-not_loaded}, KMOD_MAJOR=${KMOD_MAJOR:-unknown}\" && "
        "if [ -n \"$KMOD_MAJOR\" ]; then "
        "  if ! nvidia-smi --query-gpu=driver_version --format=csv,noheader > /dev/null 2>&1; then "
        "    echo \"[setup] nvidia-smi broken pre-flight, installing nvidia-utils-${KMOD_MAJOR}-server\" && "
        "    ( sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "        nvidia-utils-${KMOD_MAJOR}-server 2>&1 || "
        "      sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "        nvidia-utils-${KMOD_MAJOR} 2>&1 || "
        "      echo \"[setup] WARNING: no nvidia-utils package found for major ${KMOD_MAJOR}\" ) && "
        "    sudo ldconfig && "
        "    echo \"[setup] post-preflight nvidia-smi: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)\"; "
        "  else "
        "    echo \"[setup] nvidia-smi OK pre-flight, no fix needed\"; "
        "  fi; "
        "fi",
        timeout=180,
    )
    print(f"[setup] pre-flight result: {preflight.stdout.strip()}", file=sys.stderr)

    # ── 0.1 Diagnostics: show nvidia version state before we touch anything ──
    diag = _ssh(
        "echo '=== KMOD ===' && (cat /sys/module/nvidia/version 2>/dev/null || echo 'not loaded') && "
        "echo '=== DPKG nvidia-utils ===' && (dpkg -l 'nvidia-utils-*' 2>/dev/null | grep '^ii' || echo 'none') && "
        "echo '=== DPKG linux-modules-nvidia ===' && (dpkg -l 'linux-modules-nvidia-*' 2>/dev/null | grep '^ii' || echo 'none') && "
        "echo '=== nvidia-smi ===' && (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo 'failed')"
    )
    print(f"[setup] pre-install nvidia state:\n{diag.stdout}", file=sys.stderr)

    # ── 0.5. Hold nvidia-utils before CUDA repo ──────────────────────────────
    # The CUDA apt repo ships newer nvidia-utils packages that mismatch the
    # already-loaded kernel module. Hold them now so apt never auto-upgrades.
    print("[setup] holding nvidia-utils packages to prevent CUDA repo upgrade ...", file=sys.stderr)
    _ssh(
        "dpkg -l 'nvidia-utils-*' 2>/dev/null | awk '/^ii/{print $2}' "
        "| xargs -r sudo apt-mark hold 2>/dev/null || true"
    )

    # ── 1. Docker CE ─────────────────────────────────────────────────────────
    # Some AMIs (confirmed on the GB200 image, 2026-07-30) ship containerd.io
    # (Docker Inc's own runtime, from a pre-configured Docker apt repo)
    # already installed and running. docker.io (Ubuntu's own package) bundles
    # a *different* containerd package that directly conflicts with it —
    # apt refuses with "containerd.io : Conflicts: containerd". Detect that
    # case and install docker-ce/docker-ce-cli instead, which is what the
    # pre-existing containerd.io is actually meant to pair with.
    print("[setup] installing Docker ...", file=sys.stderr)
    docker_cmds = (
        "sudo apt-get update -qq && "
        "sudo apt-get install -y --no-install-recommends curl wget gnupg2 ca-certificates && "
        "if dpkg -l containerd.io 2>/dev/null | grep -q '^ii'; then "
        "  echo '[setup] containerd.io already present — installing docker-ce/docker-ce-cli "
        "instead of docker.io to avoid package conflict' && "
        "  sudo apt-get install -y --no-install-recommends docker-ce docker-ce-cli docker-compose-plugin; "
        "else "
        "  sudo apt-get install -y --no-install-recommends docker.io; "
        "fi && "
        "sudo systemctl enable --now docker && "
        "sudo usermod -aG docker ubuntu"
    )
    r = _ssh(docker_cmds, timeout=300)
    results["docker"] = r.returncode == 0
    if results["docker"]:
        print("[setup] Docker installed successfully", file=sys.stderr)
    else:
        print(f"[setup] Docker install failed: {r.stderr[-300:]}", file=sys.stderr)

    # ── 2. CUDA Toolkit (via NVIDIA official apt repo) ───────────────────────
    # Install only nvcc + cuda libraries (not the full cuda-toolkit-12-6 which
    # is ~3GB and times out on slow connections). This satisfies DriverCheck's
    # cuda_toolkit subtest which checks for nvcc availability.
    #
    # The keyring URL is architecture-specific — NVIDIA publishes ARM64
    # server (Grace/GB200) packages under "sbsa", not "arm64" or "x86_64".
    # Some AMIs (confirmed on the GB200 image, 2026-07-30) already have the
    # matching repo configured (cuda-ubuntu2404-sbsa.list) — skip re-adding
    # the keyring in that case, just install straight from it.
    print("[setup] installing nvcc + CUDA libs (architecture-aware repo) ...", file=sys.stderr)
    cuda_cmds = (
        "MACH=$(uname -m) && "
        "if [ \"$MACH\" = \"aarch64\" ] || [ \"$MACH\" = \"arm64\" ]; then ARCH_PATH=sbsa; "
        "else ARCH_PATH=x86_64; fi && "
        "if ls /etc/apt/sources.list.d/cuda*.list >/dev/null 2>&1; then "
        "  echo \"[setup] CUDA apt repo already configured, skipping keyring install\"; "
        "else "
        "  echo \"[setup] adding NVIDIA CUDA apt repo (arch path: ${ARCH_PATH})\" && "
        "  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH_PATH}/"
        "cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && "
        "  sudo dpkg -i /tmp/cuda-keyring.deb; "
        "fi && "
        "sudo apt-get update -qq && "
        # Install nvcc compiler + essential CUDA libraries (much smaller than cuda-toolkit-12-6)
        "sudo apt-get install -y --no-install-recommends "
        "  cuda-nvcc-12-6 cuda-libraries-12-6 libcufft-dev-12-6 libcurand-dev-12-6 && "
        # Add CUDA bin to system-wide PATH for all session types
        "echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh && "
        "sudo ln -sf /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc 2>/dev/null || true"
    )
    r = _ssh(cuda_cmds, timeout=1800)
    check = _ssh(
        "nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null || "
        "/usr/local/bin/nvcc --version 2>/dev/null",
        timeout=30,
    )
    results["cuda_toolkit"] = check.returncode == 0
    if results["cuda_toolkit"]:
        print(f"[setup] CUDA Toolkit installed: {check.stdout.strip()[:80]}", file=sys.stderr)
    else:
        print(f"[setup] CUDA Toolkit install failed: {r.stderr[-300:]}", file=sys.stderr)

    # ── 3. NVIDIA Container Toolkit ──────────────────────────────────────────
    # Some AMIs (confirmed on the GB200 image, 2026-07-30) already have NCT
    # installed and its apt repo configured. Don't blindly re-add the repo
    # and reinstall in that case — that's redundant churn against an
    # already-correct setup and could downgrade a newer pre-installed
    # version. Docker itself didn't exist before this function ran though,
    # so `nvidia-ctk runtime configure --runtime=docker` still needs to run
    # regardless — NCT was never actually wired to a container runtime yet.
    print("[setup] installing NVIDIA Container Toolkit ...", file=sys.stderr)
    nct_cmds = (
        "if command -v nvidia-ctk >/dev/null 2>&1; then "
        "  echo '[setup] nvidia-ctk already present, skipping repo add/reinstall'; "
        "else "
        "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey "
        "    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && "
        "  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list "
        "    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && "
        "  sudo apt-get update -qq && "
        "  sudo apt-get install -y nvidia-container-toolkit; "
        "fi && "
        "sudo nvidia-ctk runtime configure --runtime=docker && "
        "sudo systemctl restart docker"
    )
    r = _ssh(nct_cmds, timeout=300)
    results["nvidia_container_toolkit"] = r.returncode == 0
    if results["nvidia_container_toolkit"]:
        print("[setup] NVIDIA Container Toolkit installed successfully", file=sys.stderr)
    else:
        print(f"[setup] NVIDIA Container Toolkit install failed: {r.stderr[-300:]}", file=sys.stderr)

    # ── 4. Unhold and pin nvidia-utils to the loaded kernel module version ───
    # Only if nvidia-smi is actually broken at this point. Some AMIs
    # (confirmed on the GB200 image, 2026-07-30) ship with nvidia-utils
    # deliberately apt-mark-held by the vendor's own image build, already
    # matching the loaded kernel module — unholding and force-reinstalling
    # unconditionally would strip that protection for no reason and never
    # restored the hold afterward (a real regression against an
    # already-correct AMI). Skip entirely when nvidia-smi already works;
    # only fix + re-hold when it's genuinely mismatched.
    if _ssh("nvidia-smi --query-gpu=driver_version --format=csv,noheader", timeout=30).returncode == 0:
        print("[setup] nvidia-smi already OK post-install — skipping nvidia-utils unhold/pin entirely", file=sys.stderr)
        restore_cmds = None
    else:
        print("[setup] nvidia-smi broken post-install — pinning nvidia-utils to loaded kernel module version ...", file=sys.stderr)
        restore_cmds = (
        # Unhold first so apt can act on these packages again.
        "dpkg -l 'nvidia-utils-*' 2>/dev/null | awk '/^ii/{print $2}' "
        "| xargs -r sudo apt-mark unhold 2>/dev/null || true && "
        # Read the exact version from the already-loaded kernel module.
        "KMOD_VER=$(cat /sys/module/nvidia/version 2>/dev/null || echo '') && "
        "KMOD_MAJOR=$(echo \"$KMOD_VER\" | cut -d. -f1) && "
        "echo \"[setup] kernel module version: ${KMOD_VER:-unknown}, major: ${KMOD_MAJOR:-unknown}\" && "
        # If we know the kernel module version, force-install matching nvidia-utils.
        # Try -server first (Ubuntu HWE repos), fall back to plain name (CUDA repo).
        "if [ -n \"$KMOD_MAJOR\" ]; then "
        "  echo \"[setup] force-installing nvidia-utils-${KMOD_MAJOR} variants\" && "
        "  if sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "       nvidia-utils-${KMOD_MAJOR}-server 2>&1; then "
        "    echo \"[setup] installed nvidia-utils-${KMOD_MAJOR}-server OK\"; "
        "  elif sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "       nvidia-utils-${KMOD_MAJOR} 2>&1; then "
        "    echo \"[setup] installed nvidia-utils-${KMOD_MAJOR} OK\"; "
        "  else "
        "    echo \"[setup] WARNING: could not install nvidia-utils for major ${KMOD_MAJOR}\"; "
        "  fi; "
        "else "
        "  echo '[setup] kernel module not loaded, falling back to dpkg heuristic' && "
        "  DRVER=$(dpkg -l | awk '/nvidia-kernel-common-[0-9]/{match($2,/[0-9]+/,m);print m[0];exit}') && "
        "  DRVER=${DRVER:-535} && "
        "  sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "    nvidia-utils-${DRVER}-server 2>&1 || "
        "  sudo apt-get install -y --allow-downgrades --no-install-recommends "
        "    nvidia-utils-${DRVER} 2>&1 || true; "
        "fi && "
        # Refresh ldconfig so the new library symlinks are picked up immediately.
        "sudo ldconfig && "
        # Symlink nvidia-smi into /usr/local/bin for all SSH session types.
        "NVSMI=$(find /usr /opt -name nvidia-smi -type f 2>/dev/null | head -1) && "
        "echo \"[setup] nvidia-smi at: $NVSMI\" && "
        "[ -n \"$NVSMI\" ] && sudo ln -sf \"$NVSMI\" /usr/local/bin/nvidia-smi || true && "
        # Restart docker so the NVIDIA container runtime picks up the correct library.
        "sudo systemctl restart docker 2>/dev/null || true && "
        "echo '[setup] docker restarted after nvidia-utils pin' && "
        # Re-apply the hold we removed above, now that the matching version
        # is force-installed — restores the same protection the AMI had
        # before we touched it, instead of leaving packages unheld.
        "dpkg -l 'nvidia-utils-*' 2>/dev/null | awk '/^ii/{print $2}' "
        "| xargs -r sudo apt-mark hold 2>/dev/null || true && "
        # Final verification — output will appear in launch_instance JSON via nvidia_diag.
        "echo \"[setup] nvidia-smi final check: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)\""
        )

    if restore_cmds is None:
        results["nvidia_smi_accessible"] = True
    else:
        r = _ssh(restore_cmds, timeout=300)
        results["nvidia_smi_accessible"] = r.returncode == 0
        print(f"[setup] nvidia-smi restore: {r.stdout.strip()}", file=sys.stderr)

    # ── 5. Post-install diagnostics: verify version consistency ─────────────
    diag2 = _ssh(
        "echo '=== KMOD ===' && (cat /sys/module/nvidia/version 2>/dev/null || echo 'not loaded') && "
        "echo '=== DPKG nvidia-utils ===' && (dpkg -l 'nvidia-utils-*' 2>/dev/null | grep '^ii' || echo 'none') && "
        "echo '=== DPKG linux-modules-nvidia ===' && (dpkg -l 'linux-modules-nvidia-*' 2>/dev/null | grep '^ii' | head -3 || echo 'none') && "
        "echo '=== nvidia-smi ===' && (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || echo 'failed')"
    )
    print(f"[setup] post-install nvidia state:\n{diag2.stdout}", file=sys.stderr)

    print(f"[setup] GPU dependencies complete: {results}", file=sys.stderr)
    return results
