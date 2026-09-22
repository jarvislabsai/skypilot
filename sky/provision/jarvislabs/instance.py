"""JarvisLabs instance provisioning."""

import typing
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision.jarvislabs import utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    from jarvislabs.models import Instance

PROVIDER_NAME = 'jarvislabs'

logger = sky_logging.init_logger(__name__)


def _filter_instances(cluster_name_on_cloud: str) -> List['Instance']:
    return utils.list_instances(name_prefix=cluster_name_on_cloud)


def run_instances(
    region: str,
    cluster_name: str,
    cluster_name_on_cloud: str,
    config: common.ProvisionConfig,
) -> common.ProvisionRecord:
    """Provisions the single instance for a JarvisLabs cluster.

    Multi-node is unreachable here: MULTI_NODE is already listed in
    JarvisLabs._CLOUD_UNSUPPORTED_FEATURES, so `config.count` is always 1.
    """
    del cluster_name  # unused
    logger.debug(f'run_instances: region={region}, '
                 f'cluster={cluster_name_on_cloud}, config={config}')
    assert config.count == 1, ('JarvisLabs only supports single-node clusters',
                               config.count)

    existing = _filter_instances(cluster_name_on_cloud)
    running = [i for i in existing if i.status == 'Running']
    if running:
        head_instance_id = str(running[0].machine_id)
        logger.debug(f'Reusing already-running instance {head_instance_id}')
        return common.ProvisionRecord(
            provider_name=PROVIDER_NAME,
            cluster_name=cluster_name_on_cloud,
            region=region,
            zone=None,
            head_instance_id=head_instance_id,
            resumed_instance_ids=[],
            created_instance_ids=[],
        )

    paused = [i for i in existing if i.status == 'Paused']
    if paused and config.resume_stopped_nodes:
        instance = utils.resume_instance(paused[0].machine_id)
        head_instance_id = str(instance.machine_id)
        logger.debug(f'Resumed paused instance {head_instance_id}')
        return common.ProvisionRecord(
            provider_name=PROVIDER_NAME,
            cluster_name=cluster_name_on_cloud,
            region=region,
            zone=None,
            head_instance_id=head_instance_id,
            resumed_instance_ids=[head_instance_id],
            created_instance_ids=[],
        )

    gpu_type = config.node_config.get('gpu_type')
    num_gpus = config.node_config.get('num_gpus')
    storage_gb = config.node_config.get('storage_gb', 100)
    http_ports = ','.join(
        str(p) for p in (config.ports_to_open_on_launch or []))

    if gpu_type and num_gpus:
        instance = utils.launch_instance(
            name=cluster_name_on_cloud,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            storage_gb=storage_gb,
            region=region,
            http_ports=http_ports,
        )
    else:
        # CPU-only instance type (gpu_type is None): goes through the SDK's
        # separate CPU-VM create path, which -- unlike the GPU path -- has
        # no `http_ports` parameter at all
        if http_ports:
            logger.warning(
                'Ports %s were requested but JarvisLabs CPU-only instances '
                'do not support opening ports; ignoring.', http_ports)
        instance = utils.launch_cpu_instance(
            name=cluster_name_on_cloud,
            vcpus=config.node_config.get('vcpus'),
            ram_gb=config.node_config.get('ram_gb'),
            storage_gb=storage_gb,
            region=region,
        )
    head_instance_id = str(instance.machine_id)
    logger.debug(f'Launched new instance {head_instance_id}')
    return common.ProvisionRecord(
        provider_name=PROVIDER_NAME,
        cluster_name=cluster_name_on_cloud,
        region=region,
        zone=None,
        head_instance_id=head_instance_id,
        resumed_instance_ids=[],
        created_instance_ids=[head_instance_id],
    )


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """No-op: `launch_instance`/`resume_instance` already block until ready
    (the SDK's own `create()`/`resume()` poll internally)."""
    del region, cluster_name_on_cloud, state  # unused


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    del provider_config, worker_only  # unused
    instances = _filter_instances(cluster_name_on_cloud)
    for instance in instances:
        utils.pause_instance(instance.machine_id)


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    del provider_config, worker_only  # unused
    instances = _filter_instances(cluster_name_on_cloud)
    for instance in instances:
        utils.destroy_instance(instance.machine_id)


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    del cluster_name, provider_config, retry_if_missing  # unused
    instances = _filter_instances(cluster_name_on_cloud)
    statuses: Dict[str, Tuple[Optional[status_lib.ClusterStatus],
                              Optional[str]]] = {}
    for instance in instances:
        cluster_status = utils.to_cluster_status(instance.status)
        if non_terminated_only and cluster_status is None:
            continue
        statuses[str(instance.machine_id)] = (cluster_status, None)
    return statuses


def get_cluster_info(
    region: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
) -> common.ClusterInfo:
    del region  # unused
    instances = _filter_instances(cluster_name_on_cloud)

    cluster_instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id: Optional[str] = None
    ssh_user: Optional[str] = None

    for instance in instances:
        if not instance.public_ip:
            logger.debug(
                f'Skipping instance {instance.machine_id} - no public IP yet')
            continue
        parsed_user, parsed_port = utils.parse_ssh_command(instance.ssh_command)
        if parsed_user is not None:
            ssh_user = parsed_user

        instance_id = str(instance.machine_id)
        cluster_instances[instance_id] = [
            common.InstanceInfo(
                instance_id=instance_id,
                internal_ip=instance.private_ip or instance.public_ip,
                external_ip=instance.public_ip,
                ssh_port=parsed_port or 22,
                tags={},
            )
        ]
        if head_instance_id is None:
            head_instance_id = instance_id

    return common.ClusterInfo(
        instances=cluster_instances,
        head_instance_id=head_instance_id,
        provider_name=PROVIDER_NAME,
        provider_config=provider_config,
        ssh_user=ssh_user,
    )


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: Optional[List[str]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """No-op: ports are only ever set at instance creation (`http_ports` on
    launch); nothing to clean up separately once the instance is destroyed by
    `terminate_instances`."""
    del cluster_name_on_cloud, ports, provider_config  # unused


def query_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    head_ip: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[common.Endpoint]]:
    """See sky/provision/__init__.py.

    NOTE: unverified against a real instance with http_ports set -- confirm
    whether `Instance.url`/`.endpoints` should be surfaced here as HTTPS
    endpoints instead of the plain socket passthrough used below.
    """
    del cluster_name_on_cloud, provider_config  # unused
    return common.query_ports_passthrough(ports, head_ip)
