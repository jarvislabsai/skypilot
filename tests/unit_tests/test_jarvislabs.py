"""Tests for JarvisLabs cloud provider."""
from unittest import mock

import pytest
import tomli

from sky import clouds
from sky.clouds import jarvislabs
from sky.provision.jarvislabs import utils as jarvislabs_utils


def test_jarvislabs_cloud_basic_attrs():
    """Tests basic identity/equality attributes on the JarvisLabs cloud
    class."""
    cloud = jarvislabs.JarvisLabs()
    assert cloud._REPR == 'JarvisLabs'
    assert repr(cloud) == 'JarvisLabs'
    assert cloud._MAX_CLUSTER_NAME_LEN_LIMIT == 40
    assert cloud.is_same_cloud(jarvislabs.JarvisLabs())
    assert not cloud.is_same_cloud(object())


def test_unsupported_features_have_reasons():
    """Every declared-unsupported feature must carry a real reason string.
    """
    cloud = jarvislabs.JarvisLabs()
    for feature in clouds.CloudImplementationFeatures:
        if feature in cloud._CLOUD_UNSUPPORTED_FEATURES:
            assert isinstance(cloud._CLOUD_UNSUPPORTED_FEATURES[feature], str)
            assert cloud._CLOUD_UNSUPPORTED_FEATURES[feature]


def test_validate_region_zone_disallows_zones():
    """Tests that specifying a zone raises, since JarvisLabs has no zone
    concept."""
    cloud = jarvislabs.JarvisLabs()
    with pytest.raises(ValueError, match='does not support zones'):
        cloud.validate_region_zone('IN2', 'zone-1')


def test_autostop_autodown_supported_but_auto_terminate_is_not():
    """Tests that AUTOSTOP/AUTODOWN stay supported (self-triggered by the
    remote skylet via a mounted credentials) while AUTO_TERMINATE stays
    unsupported -- included as a contrast so this can't trivially pass on
    an empty dict."""
    cloud = jarvislabs.JarvisLabs()
    assert (clouds.CloudImplementationFeatures.AUTOSTOP
            not in cloud._CLOUD_UNSUPPORTED_FEATURES)
    assert (clouds.CloudImplementationFeatures.AUTODOWN
            not in cloud._CLOUD_UNSUPPORTED_FEATURES)
    assert (clouds.CloudImplementationFeatures.AUTO_TERMINATE
            in cloud._CLOUD_UNSUPPORTED_FEATURES)


class TestCheckComputeCredentials:

    def test_missing(self, monkeypatch):
        """Tests that check_credentials() reports invalid, with a clear
        message, when no API key is resolved."""
        monkeypatch.setattr(jarvislabs_utils, 'resolve_api_key', lambda: None)
        valid, msg = jarvislabs.JarvisLabs.check_credentials(
            clouds.CloudCapability.COMPUTE)
        assert not valid
        assert 'JarvisLabs credentials not found' in msg

    def test_present(self, monkeypatch):
        """Tests that check_credentials() reports valid when an API key is
        resolved."""
        monkeypatch.setattr(jarvislabs_utils, 'resolve_api_key',
                            lambda: 'fake-key')
        valid, msg = jarvislabs.JarvisLabs.check_credentials(
            clouds.CloudCapability.COMPUTE)
        assert valid
        assert msg is None


class TestGetCredentialFileMounts:

    def test_file_exists(self, monkeypatch, tmp_path):
        """Tests that an existing credentials file is mounted to the remote
        path as-is."""
        cred_file = tmp_path / 'config.toml'
        cred_file.write_text('[auth]\ntoken = "x"\n')
        monkeypatch.setattr(jarvislabs_utils, 'get_credentials_path',
                            lambda: str(cred_file))
        cloud = jarvislabs.JarvisLabs()
        assert cloud.get_credential_file_mounts() == {
            jarvislabs_utils.REMOTE_CREDENTIALS_PATH: str(cred_file)
        }

    def test_no_file_no_env_returns_empty(self, monkeypatch, tmp_path):
        """Tests that no file mounts are returned when there's neither a
        credentials file nor an env var."""
        fake_path = tmp_path / 'nonexistent.toml'
        monkeypatch.setattr(jarvislabs_utils, 'get_credentials_path',
                            lambda: str(fake_path))
        monkeypatch.setattr(jarvislabs_utils, 'resolve_api_key', lambda: None)
        cloud = jarvislabs.JarvisLabs()
        assert not cloud.get_credential_file_mounts()

    def test_no_file_env_var_materializes_staging_file(self, monkeypatch,
                                                       tmp_path):
        """Tests that an API key from the env var gets materialized into a
        staging TOML file for mounting."""
        fake_path = tmp_path / 'nonexistent.toml'
        staging_path = tmp_path / 'generated' / 'config.toml'
        monkeypatch.setattr(jarvislabs_utils, 'get_credentials_path',
                            lambda: str(fake_path))
        monkeypatch.setattr(jarvislabs_utils, 'resolve_api_key',
                            lambda: 'sk-test-key')
        monkeypatch.setattr(jarvislabs_utils, '_GENERATED_CREDENTIALS_PATH',
                            str(staging_path))
        cloud = jarvislabs.JarvisLabs()
        mounts = cloud.get_credential_file_mounts()
        assert mounts == {
            jarvislabs_utils.REMOTE_CREDENTIALS_PATH: str(staging_path)
        }
        assert staging_path.read_text() == '[auth]\ntoken = "sk-test-key"\n'
        # 0o600: owner read/write only.
        assert (staging_path.stat().st_mode & 0o777) == 0o600

    def test_materialized_key_with_special_chars_round_trips(
            self, monkeypatch, tmp_path):
        """Tests that a value containing a backslash and a double-quote
        round-trips through a real TOML parser -- not just looks
        plausibly escaped. Hardens against malformed manual input (e.g.
        JL_API_KEY set by mistake to something else), not a real
        generated token: JarvisLabs tokens come from
        secrets.token_urlsafe(), whose alphabet structurally excludes
        both characters."""
        fake_path = tmp_path / 'nonexistent.toml'
        staging_path = tmp_path / 'generated' / 'config.toml'
        tricky_key = 'a\\b"c'
        monkeypatch.setattr(jarvislabs_utils, 'get_credentials_path',
                            lambda: str(fake_path))
        monkeypatch.setattr(jarvislabs_utils, 'resolve_api_key',
                            lambda: tricky_key)
        monkeypatch.setattr(jarvislabs_utils, '_GENERATED_CREDENTIALS_PATH',
                            str(staging_path))
        cloud = jarvislabs.JarvisLabs()
        cloud.get_credential_file_mounts()

        parsed = tomli.loads(staging_path.read_text())
        assert parsed['auth']['token'] == tricky_key


class TestMakeDeployResourcesVariables:
    """Covers the GPU vs. CPU-VM branch -- the one place this cloud class
    has genuinely custom logic beyond thin catalog delegation."""

    def _make_resources(self, instance_type, disk_size=None):
        resources = mock.MagicMock()
        resources.instance_type = instance_type
        resources.disk_size = disk_size
        resources.assert_launchable = mock.Mock(return_value=resources)
        return resources

    def test_gpu_instance_type(self, monkeypatch):
        """Tests that a GPU instance type sets gpu_type/num_gpus/
        docker_run_options and skips the CPU-only fields."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('1x_A100-80GB_16V_112G')
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: {'A100-80GB': 1})

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['gpu_type'] == 'A100-80GB'
        assert result['num_gpus'] == 1
        assert result['docker_run_options'] == ['--gpus all']
        assert 'vcpus' not in result
        assert 'ram_gb' not in result

    def test_cpu_instance_type(self, monkeypatch):
        """Tests that a CPU-only instance type sets vcpus/ram_gb and skips
        the GPU fields."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G')
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['gpu_type'] is None
        assert result['num_gpus'] == 0
        assert result['vcpus'] == 8
        assert result['ram_gb'] == 32
        assert 'docker_run_options' not in result

    def test_cpu_instance_type_default_storage(self, monkeypatch):
        """Tests the `disk_size or _DEFAULT_STORAGE_GB` fallback for a
        literal None input -- unreachable via a real Resources object in
        practice (its own disk_size property already defaults to 256),
        but still covered defensively."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G', disk_size=None)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['storage_gb'] == 100

    def test_cpu_instance_type_storage_below_floor_is_clamped(
            self, monkeypatch):
        """Tests that a CPU-only disk_size below the real 100GB floor gets
        silently clamped up rather than passed through and rejected."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G', disk_size=50)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['storage_gb'] == 100

    def test_cpu_instance_type_storage_below_floor_warns(
            self, monkeypatch, caplog):
        """Clamping the requested storage up silently increases what the
        caller is billed for, so it must be visibly logged -- not a silent
        substitution the caller would only discover on their invoice."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G', disk_size=55)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        with caplog.at_level('WARNING'):
            cloud.make_deploy_resources_variables(
                resources=resources,
                cluster_name=mock.MagicMock(),
                region=region,
                zones=None,
                num_nodes=1,
            )

        assert any('55' in msg and '100' in msg for msg in caplog.messages)

    def test_cpu_instance_type_storage_at_or_above_floor_does_not_warn(
            self, monkeypatch, caplog):
        """Tests that no warning is logged when the requested storage is
        already at or above the floor."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G', disk_size=100)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        with caplog.at_level('WARNING'):
            cloud.make_deploy_resources_variables(
                resources=resources,
                cluster_name=mock.MagicMock(),
                region=region,
                zones=None,
                num_nodes=1,
            )

        assert caplog.messages == []

    def test_cpu_instance_type_storage_above_floor_is_kept(self, monkeypatch):
        """A caller-requested size already above the floor must not be
        clamped down."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('CPU_8V_32G', disk_size=200)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: None)
        monkeypatch.setattr(jarvislabs.catalog,
                            'get_vcpus_mem_from_instance_type',
                            lambda instance_type, clouds: (8.0, 32.0))

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['storage_gb'] == 200

    def test_gpu_instance_type_storage_below_floor_is_clamped(
            self, monkeypatch):
        """Tests that the same 100GB floor is clamped for GPU instance
        types too, not just CPU-only ones."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('1x_A100-80GB_16V_112G', disk_size=55)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: {'A100-80GB': 1})

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['storage_gb'] == 100

    def test_gpu_instance_type_storage_above_floor_is_kept(self, monkeypatch):
        """Tests that a GPU disk_size already above the floor isn't clamped
        down."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('1x_A100-80GB_16V_112G', disk_size=200)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: {'A100-80GB': 1})

        result = cloud.make_deploy_resources_variables(
            resources=resources,
            cluster_name=mock.MagicMock(),
            region=region,
            zones=None,
            num_nodes=1,
        )

        assert result['storage_gb'] == 200

    def test_gpu_instance_type_storage_below_floor_warns(
            self, monkeypatch, caplog):
        """Tests that clamping a GPU instance's storage also logs the
        billing-impact warning."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources('1x_A100-80GB_16V_112G', disk_size=55)
        region = mock.MagicMock()
        region.name = 'IN2'

        monkeypatch.setattr(jarvislabs.catalog,
                            'get_accelerators_from_instance_type',
                            lambda instance_type, clouds: {'A100-80GB': 1})

        with caplog.at_level('WARNING'):
            cloud.make_deploy_resources_variables(
                resources=resources,
                cluster_name=mock.MagicMock(),
                region=region,
                zones=None,
                num_nodes=1,
            )

        assert any('55' in msg and '100' in msg for msg in caplog.messages)

    def test_multi_node_rejected(self):
        """Tests that requesting more than one node raises, since
        JarvisLabs is single-node only."""
        cloud = jarvislabs.JarvisLabs()
        with pytest.raises(ValueError, match='single-node'):
            cloud.make_deploy_resources_variables(
                resources=mock.MagicMock(),
                cluster_name=mock.MagicMock(),
                region=mock.MagicMock(),
                zones=None,
                num_nodes=2,
            )


class TestFeasibleResourcesDiskSizeBounds:
    """A disk_size outside [_MIN_STORAGE_GB, _MAX_STORAGE_GB] is rejected as
    infeasible up front, not silently clamped like the CPU-VM floor in
    make_deploy_resources_variables() -- truncating an out-of-range request
    would silently violate real capacity the caller asked for. Tests
    reference the module's constants directly, not hardcoded values, so
    they stay correct if the bounds are retuned."""

    def _make_resources(self, disk_size):
        resources = mock.MagicMock()
        resources.disk_size = disk_size
        return resources

    def test_disk_size_above_max_is_infeasible(self):
        """Tests that a disk_size above the max bound is rejected as
        infeasible."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources(jarvislabs._MAX_STORAGE_GB + 1)
        result = cloud._get_feasible_launchable_resources(resources)
        assert result.resources_list == []

    def test_disk_size_below_min_is_infeasible(self):
        """Tests that a disk_size below the min bound is rejected as
        infeasible."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources(jarvislabs._MIN_STORAGE_GB - 1)
        result = cloud._get_feasible_launchable_resources(resources)
        assert result.resources_list == []

    def test_disk_size_at_max_boundary_is_feasible(self):
        """Tests that a disk_size exactly at the max bound is accepted."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources(jarvislabs._MAX_STORAGE_GB)
        resources.instance_type = 'CPU_8V_32G'
        resources.is_launchable = mock.Mock(return_value=True)
        resources.copy = mock.Mock(return_value=resources)
        result = cloud._get_feasible_launchable_resources(resources)
        assert result.resources_list == [resources]

    def test_disk_size_at_min_boundary_is_feasible(self):
        """Tests that a disk_size exactly at the min bound is accepted."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources(jarvislabs._MIN_STORAGE_GB)
        resources.instance_type = 'CPU_8V_32G'
        resources.is_launchable = mock.Mock(return_value=True)
        resources.copy = mock.Mock(return_value=resources)
        result = cloud._get_feasible_launchable_resources(resources)
        assert result.resources_list == [resources]

    def test_disk_size_none_uses_default_and_is_feasible(self):
        """Tests that a literal disk_size=None isn't rejected by the bounds
        check -- defensive coverage only; see
        test_cpu_instance_type_default_storage for why a real Resources
        object never actually hits this."""
        cloud = jarvislabs.JarvisLabs()
        resources = self._make_resources(None)
        resources.instance_type = 'CPU_8V_32G'
        resources.is_launchable = mock.Mock(return_value=True)
        resources.copy = mock.Mock(return_value=resources)
        result = cloud._get_feasible_launchable_resources(resources)
        assert result.resources_list == [resources]
