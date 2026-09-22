"""JarvisLabs Cloud provider implementation for SkyPilot."""

import typing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sky import catalog
from sky import clouds
from sky import sky_logging
from sky.provision.jarvislabs import utils as jarvislabs_utils
from sky.utils import registry
from sky.utils import resources_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky.utils import volume as volume_lib

logger = sky_logging.init_logger(__name__)

_CLOUD = 'jarvislabs'

# Upper bound of 5000GB, based on the JarvisLabs website's create-instance
# form (the website also distinguishes GPU vs. CPU bounds; that distinction
# is ignored here).
#
# Lower bound of 20GB matches the real backend schema's own absolute floor
# (below this, every create request is rejected outright, GPU or CPU-only)
# -- NOT the practical 100GB floor that both GPU and CPU-only creates
# actually enforce beyond that (see _DEFAULT_STORAGE_GB below). Any request
# from 20GB up to just under 100GB is accepted here and then silently
# clamped up in make_deploy_resources_variables(), with a visible warning;
_MIN_STORAGE_GB = 20
_MAX_STORAGE_GB = 5000

# Named DEFAULT_ to match the JarvisLabs SDK's own DEFAULT_STORAGE_GB, but
# it's never actually used as a default here: `resources.disk_size` already
# falls back to SkyPilot's own generic DEFAULT_DISK_SIZE_GB (256, in
# sky/resources.py) before this cloud's code ever sees it. This constant's
# real job is the floor in the storage_gb clamp below
_DEFAULT_STORAGE_GB = 100

# JarvisLabs' CLI/SDK identify a machine by `gpu_type` + `num_gpus`
# (e.g. `jl create --gpu A100-80GB --num-gpus 2`), not a single opaque
# instance-type string. The catalog's InstanceType strings instead follow
# the cluster-naming convention (e.g. "1x_A100-80GB-16V-112G" for a GPU
# node, "CPU-2V-8G" for a CPU-only node: "<num_gpus>x_<gpu_name>-<vCPUs>V-
# <MemoryGiB>G"). That format is NOT safe to split on '-' to recover
# gpu_name, since GPU names can themselves contain hyphens (e.g.
# "A100-80GB"). make_deploy_resources_variables() below instead reads
# gpu_type/num_gpus from the catalog's structured AcceleratorName /
# AcceleratorCount columns (via get_accelerators_from_instance_type),
# which are unambiguous.


@registry.CLOUD_REGISTRY.register
class JarvisLabs(clouds.Cloud):
    """JarvisLabs GPU Cloud."""

    _REPR = 'JarvisLabs'
    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.MULTI_NODE:
            (f'Multi-node not yet exposed by {_REPR} on SkyPilot'),
        clouds.CloudImplementationFeatures.CLONE_DISK_FROM_CLUSTER:
            (f'Migrating disk is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.SPOT_INSTANCE:
            (f'Spot VM instances are not supported yet on {_REPR}'),
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            (f'Customizing disk tier is not supported on {_REPR}; only a '
             'single SSD disk type is offered.'),
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            (f'Custom network tier is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            (f'Mounting object stores is not supported on {_REPR}. To read '
             f'data from object stores on {_REPR}, use `mode: COPY` to '
             'copy the data to local disk.'),
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            (f'High availability controllers are not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            ('Customized multiple network interfaces are not supported on '
             f'{_REPR}.'),
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            (f'Local disk is not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.IMAGE_ID:
            (f'Custom image IDs are not supported on {_REPR}.'),
        clouds.CloudImplementationFeatures.AUTO_TERMINATE:
            (f'Auto-termination is not supported on {_REPR}.'),
    }

    # JarvisLabs instance names are capped at 40 characters (SDK/CLI `name`).
    _MAX_CLUSTER_NAME_LEN_LIMIT = 40

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    # Port rules are only known to be settable at instance-creation time
    # (`http_ports` on create()); no confirmed API exists yet to update
    # them on a running instance, even though the product supports it.
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.LAUNCH_ONLY

    @classmethod
    def _unsupported_features_for_resources(
        cls,
        resources: 'resources_lib.Resources',
        region: Optional[str] = None,
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del resources, region  # unused
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None,
    ) -> List[clouds.Region]:
        assert zone is None, 'JarvisLabs does not support zones.'
        del accelerators, zone  # unused
        regions = catalog.get_region_zones_for_instance_type(
            instance_type, use_spot, _CLOUD)
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls,
        instance_type: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        return catalog.get_vcpus_mem_from_instance_type(instance_type,
                                                        clouds=_CLOUD)

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> Iterator[None]:
        del num_nodes  # unused
        regions = cls.regions_with_offering(instance_type,
                                            accelerators,
                                            use_spot,
                                            region=region,
                                            zone=None)
        for r in regions:
            assert r.zones is None, r
            yield r.zones

    def instance_type_to_hourly_cost(self,
                                     instance_type: str,
                                     use_spot: bool,
                                     region: Optional[str] = None,
                                     zone: Optional[str] = None) -> float:
        return catalog.get_hourly_cost(instance_type,
                                       use_spot=use_spot,
                                       region=region,
                                       zone=zone,
                                       clouds=_CLOUD)

    def accelerators_to_hourly_cost(self,
                                    accelerators: Dict[str, int],
                                    use_spot: bool,
                                    region: Optional[str] = None,
                                    zone: Optional[str] = None) -> float:
        """Returns the hourly cost of the accelerators, in dollars/hour."""
        del accelerators, use_spot, region, zone  # unused
        return 0.0  # JarvisLabs includes accelerators in the hourly cost.

    def get_egress_cost(self, num_gigabytes: float) -> float:
        return 0.0

    def __repr__(self):
        return self._REPR

    def is_same_cloud(self, other: clouds.Cloud) -> bool:
        return isinstance(other, JarvisLabs)

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        """Returns the default instance type for JarvisLabs."""
        return catalog.get_default_instance_type(
            cpus=cpus,
            memory=memory,
            disk_tier=disk_tier,
            local_disk=local_disk,
            region=region,
            zone=zone,
            use_spot=use_spot,
            max_hourly_cost=max_hourly_cost,
            clouds=_CLOUD)

    @classmethod
    def get_accelerators_from_instance_type(
        cls,
        instance_type: str,
    ) -> Optional[Dict[str, Union[int, float]]]:
        return catalog.get_accelerators_from_instance_type(instance_type,
                                                           clouds=_CLOUD)

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: resources_utils.ClusterName,
        region: 'clouds.Region',
        zones: Optional[List['clouds.Zone']],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Any]:
        del dryrun, cluster_name  # unused
        assert zones is None, ('JarvisLabs does not support zones', zones)
        if num_nodes > 1:
            raise ValueError(
                'JarvisLabs currently only supports single-node clusters, '
                f'but {num_nodes} nodes were requested.')

        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)

        # `gpu_type`/`num_gpus` are what the JarvisLabs SDK's
        # `instances.create()` actually takes; `instance_type` below is only
        # SkyPilot's internal catalog key. See the module-level comment on
        # why these come from the catalog columns, not from parsing
        # `resources.instance_type` itself.
        gpu_type: Optional[str] = None
        num_gpus = 0
        if acc_dict is not None:
            assert len(acc_dict) == 1, (resources.instance_type, acc_dict)
            (gpu_type, raw_num_gpus), = acc_dict.items()
            num_gpus = int(raw_num_gpus)

        # Clamp up to the real API's storage floor (see the comment on
        # _DEFAULT_STORAGE_GB) -- this applies to *both* GPU and CPU-only
        # creates (confirmed live: a GPU create with storage=55 was rejected
        # server-side with "Disk size must be at least 100 GB for V2 VM
        # instances."), so it must run before the GPU/CPU-only branch below,
        # not just inside the CPU-only one. A caller-requested disk_size
        # below this (e.g. the managed-jobs controller's default 50GB) would
        # otherwise make every launch attempt fail. This silently increases
        # what the caller is billed for, so warn every time it actually
        # changes the requested value -- don't let the cost increase pass
        # without a visible trace, mirroring the ports-ignored warning in
        # sky/provision/jarvislabs/instance.py.
        requested_storage_gb = resources.disk_size or _DEFAULT_STORAGE_GB
        if requested_storage_gb < _DEFAULT_STORAGE_GB:
            logger.warning(f'JarvisLabs instances require at least '
                           f'{_DEFAULT_STORAGE_GB}GB storage; the requested '
                           f'{requested_storage_gb}GB will be increased to '
                           f'{_DEFAULT_STORAGE_GB}GB.')
        storage_gb = max(requested_storage_gb, _DEFAULT_STORAGE_GB)

        resources_vars: Dict[str, Any] = {
            'instance_type': resources.instance_type,
            'custom_resources': custom_resources,
            'region': region.name,
            'gpu_type': gpu_type,
            'num_gpus': num_gpus,
            'storage_gb': storage_gb,
        }

        if acc_dict is not None:
            # Confirmed necessary against a real GPU instance: JarvisLabs'
            # VM images do come with the NVIDIA Container Toolkit
            # pre-installed and pre-registered as a Docker runtime, but
            # `docker run` still needs `--gpus all` explicitly. Verified
            # both ways with `docker run --rm [--gpus all]
            # nvidia/cuda:... nvidia-smi`: with the flag it correctly
            # lists the GPU; without it, `nvidia-smi` isn't even present
            # in the container ("executable file not found in $PATH"),
            # since that's what the GPU-aware runtime injects.
            resources_vars['docker_run_options'] = ['--gpus all']
        else:
            # CPU-only instance type: the SDK's CPU-VM create path takes
            # vcpus/ram directly (there's no `instance_type` string on the
            # API side), so pull them from the same catalog row instead of
            # gpu_type/num_gpus.
            vcpus, mem_gib = self.get_vcpus_mem_from_instance_type(
                resources.instance_type)
            resources_vars['vcpus'] = int(vcpus) if vcpus is not None else None
            resources_vars['ram_gb'] = (int(mem_gib)
                                        if mem_gib is not None else None)

        return resources_vars

    def _get_feasible_launchable_resources(
        self, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        """Returns a list of feasible resources for the given resources."""
        if (resources.disk_size is not None and
                not _MIN_STORAGE_GB <= resources.disk_size <= _MAX_STORAGE_GB):
            logger.info(
                f'JarvisLabs disk size must be between {_MIN_STORAGE_GB}GB '
                f'and {_MAX_STORAGE_GB}GB. Input: {resources.disk_size}GB.')
            return resources_utils.FeasibleResources([], [], None)
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            resources = resources.copy(accelerators=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            resource_list = []
            for instance_type in instance_list:
                r = resources.copy(
                    cloud=JarvisLabs(),
                    instance_type=instance_type,
                    accelerators=None,
                    cpus=None,
                )
                resource_list.append(r)
            return resource_list

        # Currently, handle a filter on accelerators only.
        accelerators = resources.accelerators
        if accelerators is None:
            # Return a default instance type
            default_instance_type = JarvisLabs.get_default_instance_type(
                cpus=resources.cpus,
                memory=resources.memory,
                disk_tier=resources.disk_tier,
                local_disk=resources.local_disk,
                region=resources.region,
                zone=resources.zone,
                use_spot=resources.use_spot,
                max_hourly_cost=resources.max_hourly_cost)
            if default_instance_type is None:
                return resources_utils.FeasibleResources([], [], None)
            return resources_utils.FeasibleResources(
                _make([default_instance_type]), [], None)

        assert len(accelerators) == 1, resources
        acc, acc_count = list(accelerators.items())[0]
        (instance_list,
         fuzzy_candidate_list) = catalog.get_instance_type_for_accelerator(
             acc,
             acc_count,
             use_spot=resources.use_spot,
             cpus=resources.cpus,
             memory=resources.memory,
             local_disk=resources.local_disk,
             region=resources.region,
             zone=resources.zone,
             max_hourly_cost=resources.max_hourly_cost,
             clouds=_CLOUD)
        if instance_list is None:
            return resources_utils.FeasibleResources([], fuzzy_candidate_list,
                                                     None)
        return resources_utils.FeasibleResources(_make(instance_list),
                                                 fuzzy_candidate_list, None)

    @classmethod
    def _check_compute_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        """Checks if the user has access credentials to JarvisLabs."""
        try:
            api_key = jarvislabs_utils.resolve_api_key()
        except ImportError as e:
            # SDK not installed: surface the adaptor's own clean message
            # instead of letting the ImportError propagate as a traceback.
            return False, str(e)
        if api_key is None:
            msg = ('JarvisLabs credentials not found. Run the following '
                   'command of the JarvisLabs CLI:\n'
                   '      $ jl setup\n'
                   '    Alternatively, set a valid key in the terminal:\n'
                   f'      $ export {jarvislabs_utils.ENV_API_KEY}='
                   '<your-api-key>')
            return False, msg
        return True, None

    def get_credential_file_mounts(self) -> Dict[str, str]:
        local_path = jarvislabs_utils.materialize_credentials_for_mount()
        if local_path is None:
            return {}
        return {jarvislabs_utils.REMOTE_CREDENTIALS_PATH: local_path}

    @classmethod
    def get_user_identities(cls) -> Optional[List[List[str]]]:
        # NOTE: used for very advanced SkyPilot functionality
        # Can implement later if desired
        return None

    def instance_type_exists(self, instance_type: str) -> bool:
        return catalog.instance_type_exists(instance_type, _CLOUD)

    def validate_region_zone(self, region: Optional[str], zone: Optional[str]):
        return catalog.validate_region_zone(region, zone, clouds=_CLOUD)

    @classmethod
    def get_image_size(cls, image_id: str, region: Optional[str]) -> float:
        del image_id, region  # unused
        return 0.0
