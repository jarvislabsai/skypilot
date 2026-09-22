"""JarvisLabs API utilities, wrapping the official `jarvislabs` SDK."""

import os
import re
import time
import typing
from typing import Dict, List, Optional, Tuple

from sky import sky_logging
from sky.adaptors.jarvislabs import jarvislabs
from sky.skylet import constants
from sky.utils import common_utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    from jarvislabs import Client
    from jarvislabs.models import Instance

logger = sky_logging.init_logger(__name__)

# Kept as a plain constant (rather than reaching into the lazily-imported SDK)
# so error messages can reference it without forcing the SDK import.
ENV_API_KEY = 'JL_API_KEY'

# Where the SDK's config file lives on a SkyPilot remote node. NOTE: this is
# deliberately hardcoded rather than derived from `jarvislabs.config
# .config_path()` -- that call resolves via `platformdirs` against whatever
# OS *calls* it, and SkyPilot remote nodes are always Linux regardless of the
# local dev machine's OS (e.g. on macOS it resolves to
# '~/Library/Application Support/jl/config.toml' locally, but the remote
# Linux node's XDG-style path is '~/.config/jl/config.toml').
REMOTE_CREDENTIALS_PATH = '~/.config/jl/config.toml'

# Local staging path for a credentials file materialized from JL_API_KEY when
# no real config file exists locally (see materialize_credentials_for_mount).
_GENERATED_CREDENTIALS_PATH = os.path.join(constants.SKY_USER_FILE_PATH,
                                           'jarvislabs', 'config.toml')

_SSH_PORT_RE = re.compile(r'-p\s+(\d+)')
_SSH_HOST_RE = re.compile(r'(?:([\w.-]+)@)?([\w.-]+)\s*$')

# NOTE: Only 'Running'/'Paused'/'Failed' are confirmed from the SDK source
# (jarvislabs.instances). Unrecognized statuses default to INIT (treated as
# still-transitioning) rather than silently dropped or marked UP, since that
# is always the safer of the two failure modes here.
_STATUS_MAP: Dict[str, Optional['status_lib.ClusterStatus']] = {
    'Running': status_lib.ClusterStatus.UP,
    'Paused': status_lib.ClusterStatus.STOPPED,
    'Failed': status_lib.ClusterStatus.INIT,
}

# JarvisLabs' real API spells this GPU with a hyphen; SkyPilot's catalog uses
# the no-hyphen form so it merges with GCP/RunPod's shared spelling in
# sky/catalog/__init__.py's get_common_gpus() instead of showing up as a
# separate "Other GPU" entry in `sky show-gpus`. This table is the single
# source of truth for the translation in both directions -- the catalog
# fetcher (sky/catalog/data_fetchers/fetch_jarvislabs.py) uses
# to_skypilot_gpu_type() when building catalog rows from the real API's
# response, and launch_instance() below uses to_jarvislabs_gpu_type() right
# before the actual SDK call, so the real API never sees the display
# spelling.
JARVISLABS_TO_SKYPILOT_GPU_TYPES: Dict[str, str] = {'RTX-PRO6000': 'RTXPRO6000'}
_SKYPILOT_TO_JARVISLABS_GPU_TYPES: Dict[str, str] = {
    v: k for k, v in JARVISLABS_TO_SKYPILOT_GPU_TYPES.items()
}


def to_skypilot_gpu_type(real_gpu_type: str) -> str:
    """Maps a real JarvisLabs API gpu_type to SkyPilot's catalog spelling.

    Unmapped names pass through unchanged.
    """
    return JARVISLABS_TO_SKYPILOT_GPU_TYPES.get(real_gpu_type, real_gpu_type)


def to_jarvislabs_gpu_type(display_gpu_type: str) -> str:
    """Maps a SkyPilot catalog gpu_type back to what the real API expects.

    Unmapped names pass through unchanged.
    """
    return _SKYPILOT_TO_JARVISLABS_GPU_TYPES.get(display_gpu_type,
                                                 display_gpu_type)


class JarvisLabsProvisionError(Exception):
    """Raised for JarvisLabs provisioning failures.

    NOTE: deliberately not named `JarvislabsError` -- that name is already
    the SDK's own exception base class. Catch `jarvislabs.exceptions.JarvislabsError` 
    to catch anything the SDK can raise.
    """


def get_credentials_path() -> str:
    """Returns the path to the JarvisLabs CLI config file.

    Delegates to the SDK's own resolution (platformdirs-based) rather than
    hardcoding a path, so this always matches wherever `jl setup` actually
    writes on the current platform.
    """
    return str(jarvislabs.config.config_path())


def resolve_api_key() -> Optional[str]:
    """Resolves the API key via the SDK's own precedence (env, then file)."""
    return jarvislabs.config.resolve_token()


def materialize_credentials_for_mount() -> Optional[str]:
    """Returns a local file path to mount to REMOTE_CREDENTIALS_PATH, or None.

    `resolve_api_key()` accepts a key from either the `JL_API_KEY` env var or
    the SDK's config file (env takes precedence). Only the file half of that
    is mountable to a remote node -- an env var set on this machine never
    reaches the remote skylet. That matters because AUTOSTOP/AUTODOWN run
    from the remote skylet itself (self-triggered on the idle timer, not
    invoked from this machine), so it needs its own copy of the credentials
    to call JarvisLabs' API.

    If a real local config file already exists, mount it directly. Otherwise,
    if a key resolves (necessarily from the env var, since the file doesn't
    exist), materialize a minimal config file with that key at a
    SkyPilot-managed staging path and mount that instead. Returns None if no
    key can be resolved at all (matches _check_compute_credentials already
    failing `sky check` in that case, so launch should never reach here).
    """
    real_path = os.path.expanduser(get_credentials_path())
    if os.path.exists(real_path):
        return real_path

    api_key = resolve_api_key()
    if api_key is None:
        return None

    staging_path = os.path.expanduser(_GENERATED_CREDENTIALS_PATH)
    os.makedirs(os.path.dirname(staging_path), exist_ok=True)
    # Minimal TOML basic-string escaping (backslash and double-quote) -- API
    # tokens aren't expected to contain these, but escape defensively rather
    # than assume.
    escaped_key = api_key.replace('\\', '\\\\').replace('"', '\\"')
    with open(staging_path, 'w', encoding='utf-8') as f:
        f.write(f'[auth]\ntoken = "{escaped_key}"\n')
    os.chmod(staging_path, 0o600)
    return staging_path


def get_client() -> 'Client':
    """Builds a JarvisLabs SDK client, raising a clear error if no key is
    configured."""
    try:
        return jarvislabs.Client()
    except jarvislabs.exceptions.AuthError as e:
        raise JarvisLabsProvisionError(
            f'JarvisLabs credentials not found. Set {ENV_API_KEY} or run '
            f'`jl setup` to write {get_credentials_path()}.') from e


def get_or_add_ssh_key(public_key: str) -> None:
    """Registers `public_key` with JarvisLabs if not already present.

    JarvisLabs SSH keys are account-wide (not passed per-instance at create
    time, unlike e.g. Mithril), so this only needs to run once via
    `sky.authentication.setup_jarvislabs_authentication`, not per launch.
    """
    client = get_client()
    key_parts = public_key.strip().split()
    key_content = ' '.join(
        key_parts[:2]) if len(key_parts) >= 2 else (public_key.strip())

    for key in client.ssh_keys.list():
        existing = key.ssh_key.strip()
        if key_content in existing or existing in key_content:
            logger.debug(f'Found matching SSH key: {key.key_id}')
            return

    user_name = common_utils.get_current_user_name()[:10]
    key_name = f'sky-{user_name}-{int(time.time())}'
    logger.debug(f'Uploading new SSH key {key_name!r} to JarvisLabs')
    client.ssh_keys.add(public_key.strip(), key_name)


def to_cluster_status(status: str) -> Optional['status_lib.ClusterStatus']:
    """Maps a JarvisLabs instance status string to ClusterStatus."""
    return _STATUS_MAP.get(status, status_lib.ClusterStatus.INIT)


def list_instances(name_prefix: Optional[str] = None) -> List['Instance']:
    """Lists instances, optionally filtered by a name prefix.

    JarvisLabs has no cloud-native tagging (the `Instance` model has no tags
    field), so cluster identification is done by name prefix, same as
    Mithril/Yotta.
    """
    client = get_client()
    try:
        instances = client.instances.list()
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(f'Failed to list instances: {e}') from e
    if name_prefix is None:
        return instances
    return [i for i in instances if (i.name or '').startswith(name_prefix)]


def get_instance(machine_id: int) -> 'Instance':
    client = get_client()
    try:
        return client.instances.get(machine_id)
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to get instance {machine_id}: {e}') from e


def launch_instance(*,
                    name: str,
                    gpu_type: str,
                    num_gpus: int,
                    storage_gb: int,
                    region: str,
                    http_ports: str = '') -> 'Instance':
    """Creates a VM-mode GPU instance and blocks until it's running.

    `region` is passed through as SkyPilot's own display code (e.g. 'IN2')
    -- the SDK's own region normalization handles converting that to the
    internal region id, so no region-mapping table is needed here.

    `gpu_type` is translated from SkyPilot's catalog display name to the
    real API's spelling via to_jarvislabs_gpu_type() (see
    JARVISLABS_TO_SKYPILOT_GPU_TYPES) -- a no-op for every GPU except
    RTX-PRO6000 today.
    """
    client = get_client()
    try:
        return client.instances.create(
            gpu_type=to_jarvislabs_gpu_type(gpu_type),
            num_gpus=num_gpus,
            template='vm',
            storage=storage_gb,
            name=name,
            region=region,
            http_ports=http_ports,
        )
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to launch instance {name!r}: {e}') from e


def launch_cpu_instance(*, name: str, vcpus: Optional[int],
                        ram_gb: Optional[int], storage_gb: int,
                        region: str) -> 'Instance':
    """Creates a VM-mode CPU-only instance and blocks until it's running.

    Goes through the SDK's dedicated `create_cpu_vm` (rather than
    `create(cpu=True, ...)`, which is just a thin validating wrapper around
    the same call) since there's no `gpu_type`/`num_gpus`/`template` to
    thread through for a CPU-only request.
    """
    client = get_client()
    try:
        return client.instances.create_cpu_vm(
            vcpus=vcpus,
            ram=ram_gb,
            storage=storage_gb,
            name=name,
            region=region,
        )
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to launch CPU instance {name!r}: {e}') from e


def pause_instance(machine_id: int) -> None:
    client = get_client()
    try:
        client.instances.pause(machine_id)
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to pause instance {machine_id}: {e}') from e


def resume_instance(machine_id: int) -> 'Instance':
    client = get_client()
    try:
        return client.instances.resume(machine_id)
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to resume instance {machine_id}: {e}') from e


def destroy_instance(machine_id: int) -> None:
    client = get_client()
    try:
        client.instances.destroy(machine_id)
    except jarvislabs.exceptions.JarvislabsError as e:
        raise JarvisLabsProvisionError(
            f'Failed to destroy instance {machine_id}: {e}') from e


def parse_ssh_command(
        ssh_command: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """Extracts (user, port) from an `Instance.ssh_command` string.
    """
    if not ssh_command:
        return None, None
    port_match = _SSH_PORT_RE.search(ssh_command)
    port = int(port_match.group(1)) if port_match else None
    host_match = _SSH_HOST_RE.search(ssh_command)
    user = host_match.group(1) if host_match else None
    return user, port
