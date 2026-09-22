"""Unit tests for sky.provision.jarvislabs (utils.py and instance.py)."""
from unittest import mock

from jarvislabs.exceptions import AuthError
import pytest

from sky.provision import common
from sky.provision.jarvislabs import instance as jarvislabs_instance
from sky.provision.jarvislabs import utils as jarvislabs_utils
from sky.utils import status_lib


def _make_instance(name, **attrs):
    """MagicMock standing in for a jarvislabs.models.Instance.
    `.name` must be set as a real string post-construction (not via the
    MagicMock(name=...) constructor kwarg, which sets the mock's own debug
    name instead) since _filter_instances() does a real
    `.name.startswith(prefix)` call on it.
    """
    instance = mock.MagicMock()
    instance.name = name
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


class TestToClusterStatus:
    """Unrecognized statuses must default to INIT (a deliberate safety
    choice per the module comment), never silently dropped or marked UP."""

    def test_running_maps_to_up(self):
        """Tests that a 'Running' API status maps to ClusterStatus.UP."""
        assert jarvislabs_utils.to_cluster_status(
            'Running') == status_lib.ClusterStatus.UP

    def test_paused_maps_to_stopped(self):
        """Tests that a 'Paused' API status maps to ClusterStatus.STOPPED."""
        assert jarvislabs_utils.to_cluster_status(
            'Paused') == status_lib.ClusterStatus.STOPPED

    def test_failed_maps_to_init(self):
        """Tests that a 'Failed' API status maps to ClusterStatus.INIT."""
        assert jarvislabs_utils.to_cluster_status(
            'Failed') == status_lib.ClusterStatus.INIT

    def test_unrecognized_status_defaults_to_init(self):
        """Tests that an unrecognized status string also maps to INIT,
        rather than raising or being mistaken for UP."""
        assert jarvislabs_utils.to_cluster_status(
            'SomeNewStatus') == status_lib.ClusterStatus.INIT


class TestGpuTypeDisplayNameMapping:
    """SkyPilot's catalog shows JarvisLabs' RTX-PRO6000 as RTXPRO6000 (no
    hyphen) so it merges with GCP/RunPod's shared spelling in
    sky/catalog/__init__.py's get_common_gpus() instead of landing in
    `sky show-gpus`'s separate "Other GPU" section. The real API only
    understands the hyphenated form."""

    def test_real_to_display(self):
        """Tests that the hyphenated real API name translates to SkyPilot's
        no-hyphen display name."""
        assert jarvislabs_utils.to_skypilot_gpu_type(
            'RTX-PRO6000') == 'RTXPRO6000'

    def test_display_to_real(self):
        """Tests that SkyPilot's display name translates back to the real
        API's hyphenated name."""
        assert jarvislabs_utils.to_jarvislabs_gpu_type(
            'RTXPRO6000') == 'RTX-PRO6000'

    def test_unmapped_gpu_type_passes_through_unchanged(self):
        """Tests that a GPU type with no mapping entry passes through
        unchanged in both directions."""
        assert jarvislabs_utils.to_skypilot_gpu_type('H100') == 'H100'
        assert jarvislabs_utils.to_jarvislabs_gpu_type('H100') == 'H100'

    def test_round_trip(self):
        """Tests that translating real -> display -> real recovers the
        original name."""
        real = 'RTX-PRO6000'
        assert jarvislabs_utils.to_jarvislabs_gpu_type(
            jarvislabs_utils.to_skypilot_gpu_type(real)) == real


class TestLaunchInstanceGpuTypeTranslation:
    """launch_instance() must translate the catalog's display name back to
    the real API's spelling right before the SDK call -- the display name
    only exists for `sky show-gpus` grouping; the real API never sees it."""

    def _mock_client(self):
        client = mock.MagicMock()
        client.instances.create.return_value = mock.MagicMock()
        return client

    def test_translates_known_alias(self, monkeypatch):
        """Tests that launch_instance() translates a known display alias
        back to the real spelling before calling the SDK."""
        client = self._mock_client()
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.launch_instance(name='c',
                                         gpu_type='RTXPRO6000',
                                         num_gpus=1,
                                         storage_gb=100,
                                         region='IN1')

        assert client.instances.create.call_args.kwargs['gpu_type'] == (
            'RTX-PRO6000')

    def test_passes_through_unmapped_gpu_type(self, monkeypatch):
        """Tests that launch_instance() passes an unmapped GPU type through
        unchanged."""
        client = self._mock_client()
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.launch_instance(name='c',
                                         gpu_type='H100',
                                         num_gpus=1,
                                         storage_gb=100,
                                         region='IN2')

        assert client.instances.create.call_args.kwargs['gpu_type'] == 'H100'


class TestRunInstancesRouting:
    """run_instances() must route to the GPU or CPU-VM SDK call based on
    whether node_config carries gpu_type/num_gpus."""

    def _make_config(self, node_config, ports=None):
        return common.ProvisionConfig(
            provider_config={},
            authentication_config={},
            docker_config={},
            node_config=node_config,
            count=1,
            tags={},
            resume_stopped_nodes=True,
            ports_to_open_on_launch=ports,
        )

    def _mock_client(self, machine_id):
        client = mock.MagicMock()
        client.instances.list.return_value = []
        instance = mock.MagicMock()
        instance.machine_id = machine_id
        client.instances.create.return_value = instance
        client.instances.create_cpu_vm.return_value = instance
        return client

    def test_gpu_config_calls_create(self, monkeypatch):
        """Tests that a GPU node_config routes to the SDK's GPU create()
        call, not create_cpu_vm()."""
        client = self._mock_client(machine_id=111)
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        record = jarvislabs_instance.run_instances(
            region='IN2',
            cluster_name='c',
            cluster_name_on_cloud='sky-c-abcd',
            config=self._make_config({
                'gpu_type': 'A100-80GB',
                'num_gpus': 1,
                'storage_gb': 100,
            }),
        )

        client.instances.create.assert_called_once()
        client.instances.create_cpu_vm.assert_not_called()
        assert client.instances.create.call_args.kwargs['gpu_type'] == (
            'A100-80GB')
        # instance_id is cluster_name_on_cloud, not the real machine_id --
        # see the module-level comment in instance.py for why.
        assert record.head_instance_id == 'sky-c-abcd'
        assert record.created_instance_ids == ['sky-c-abcd']

    def test_gpu_display_name_translated_end_to_end(self, monkeypatch):
        """node_config carries the catalog's display name (RTXPRO6000, no
        hyphen); the actual SDK call reaching the real API must carry the
        real spelling (RTX-PRO6000) -- exercises the full run_instances()
        -> launch_instance() path, not just launch_instance() in isolation.
        """
        client = self._mock_client(machine_id=555)
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_instance.run_instances(
            region='IN1',
            cluster_name='c',
            cluster_name_on_cloud='sky-c-rtx',
            config=self._make_config({
                'gpu_type': 'RTXPRO6000',
                'num_gpus': 1,
                'storage_gb': 100,
            }),
        )

        assert client.instances.create.call_args.kwargs['gpu_type'] == (
            'RTX-PRO6000')

    def test_cpu_only_config_calls_create_cpu_vm(self, monkeypatch):
        """Tests that a CPU-only node_config routes to create_cpu_vm() with
        the right vcpus/ram/storage."""
        client = self._mock_client(machine_id=222)
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        record = jarvislabs_instance.run_instances(
            region='IN1',
            cluster_name='c',
            cluster_name_on_cloud='sky-c-wxyz',
            config=self._make_config({
                'vcpus': 8,
                'ram_gb': 32,
                'storage_gb': 100,
            }),
        )

        client.instances.create_cpu_vm.assert_called_once_with(
            vcpus=8,
            ram=32,
            storage=100,
            name='sky-c-wxyz',
            region='IN1',
        )
        client.instances.create.assert_not_called()
        assert record.head_instance_id == 'sky-c-wxyz'

    def test_cpu_only_config_with_ports_drops_them_with_warning(
            self, monkeypatch, caplog):
        """Tests that requested ports are dropped, with a warning, for
        CPU-only creates, since create_cpu_vm() has no ports parameter."""
        client = self._mock_client(machine_id=333)
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        with caplog.at_level('WARNING'):
            jarvislabs_instance.run_instances(
                region='IN2',
                cluster_name='c',
                cluster_name_on_cloud='sky-c-ports',
                config=self._make_config(
                    {
                        'vcpus': 4,
                        'ram_gb': 16,
                        'storage_gb': 100,
                    },
                    ports=['8080']),
            )

        call_kwargs = client.instances.create_cpu_vm.call_args.kwargs
        assert 'http_ports' not in call_kwargs
        assert 'do not support opening ports' in caplog.text

    def test_reuses_running_instance_without_creating(self, monkeypatch):
        """Tests that an already-running instance matching the cluster name
        is reused instead of creating a new one."""
        client = mock.MagicMock()
        existing = _make_instance('sky-c-reuse-machine',
                                  status='Running',
                                  machine_id=444)
        client.instances.list.return_value = [existing]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        record = jarvislabs_instance.run_instances(
            region='IN2',
            cluster_name='c',
            cluster_name_on_cloud='sky-c-reuse',
            config=self._make_config({
                'gpu_type': 'A100-80GB',
                'num_gpus': 1,
                'storage_gb': 100,
            }),
        )

        client.instances.create.assert_not_called()
        client.instances.create_cpu_vm.assert_not_called()
        assert record.head_instance_id == 'sky-c-reuse'
        assert record.created_instance_ids == []


class TestStopAndTerminateInstances:
    """stop_instances()/terminate_instances() are what AUTOSTOP/AUTODOWN
    actually call on the remote skylet's self-triggered idle timer."""

    def test_stop_instances_pauses_each_matching_instance(self, monkeypatch):
        """Tests that stop_instances() pauses every instance whose name
        matches the cluster."""
        client = mock.MagicMock()
        inst1 = _make_instance('sky-c-abcd-head', machine_id=1)
        inst2 = _make_instance('sky-c-abcd-worker', machine_id=2)
        client.instances.list.return_value = [inst1, inst2]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_instance.stop_instances('sky-c-abcd')

        assert client.instances.pause.call_count == 2
        client.instances.pause.assert_any_call(1)
        client.instances.pause.assert_any_call(2)

    def test_stop_instances_ignores_non_matching_names(self, monkeypatch):
        """Tests that stop_instances() leaves instances belonging to other
        clusters untouched."""
        client = mock.MagicMock()
        other_cluster = _make_instance('sky-other-cluster', machine_id=9)
        client.instances.list.return_value = [other_cluster]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_instance.stop_instances('sky-c-abcd')

        client.instances.pause.assert_not_called()

    def test_terminate_instances_destroys_each_matching_instance(
            self, monkeypatch):
        """Tests that terminate_instances() destroys every instance whose
        name matches the cluster."""
        client = mock.MagicMock()
        inst = _make_instance('sky-c-abcd-head', machine_id=1)
        client.instances.list.return_value = [inst]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_instance.terminate_instances('sky-c-abcd')

        client.instances.destroy.assert_called_once_with(1)


class TestQueryInstances:

    def test_maps_status_keyed_by_cluster_name_on_cloud(self, monkeypatch):
        """Tests that query_instances() keys its result by
        cluster_name_on_cloud (not the real, resume-churning machine_id --
        see the module comment in instance.py) and maps the real status
        through to_cluster_status() correctly."""
        inst = _make_instance('sky-c-1-head', machine_id=1, status='Running')
        client = mock.MagicMock()
        client.instances.list.return_value = [inst]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        statuses = jarvislabs_instance.query_instances(
            cluster_name='c', cluster_name_on_cloud='sky-c-1')

        assert statuses == {'sky-c-1': (status_lib.ClusterStatus.UP, None)}


class TestGetClusterInfo:

    def test_skips_instance_without_public_ip(self, monkeypatch):
        """Tests that an instance with no public IP yet (still
        provisioning) is excluded from the cluster info instead of
        crashing."""
        no_ip = _make_instance('sky-c-1-head', public_ip=None)
        client = mock.MagicMock()
        client.instances.list.return_value = [no_ip]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        info = jarvislabs_instance.get_cluster_info(
            region='IN2', cluster_name_on_cloud='sky-c-1')

        assert info.instances == {}
        assert info.head_instance_id is None

    def test_extracts_ssh_user_and_builds_instance_info(self, monkeypatch):
        """Tests that ssh_user/ssh_port get parsed out of ssh_command and
        the rest of InstanceInfo is built correctly."""
        inst = _make_instance('sky-c-1-head',
                              machine_id=5,
                              public_ip='1.2.3.4',
                              private_ip='10.0.0.5',
                              ssh_command='ssh -p 2222 root@1.2.3.4')
        client = mock.MagicMock()
        client.instances.list.return_value = [inst]
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        info = jarvislabs_instance.get_cluster_info(
            region='IN2', cluster_name_on_cloud='sky-c-1')

        assert info.head_instance_id == 'sky-c-1'
        assert info.ssh_user == 'root'
        node_info = info.instances['sky-c-1'][0]
        assert node_info.internal_ip == '10.0.0.5'
        assert node_info.external_ip == '1.2.3.4'
        assert node_info.ssh_port == 2222


class TestGetClient:

    def test_auth_error_converted_to_provision_error(self, monkeypatch):
        """Tests that an SDK AuthError raised during client construction is
        converted to a JarvisLabsProvisionError with a clear message."""
        monkeypatch.setattr(jarvislabs_utils.jarvislabs, 'Client',
                            mock.Mock(side_effect=AuthError('invalid token')))

        with pytest.raises(jarvislabs_utils.JarvisLabsProvisionError,
                           match='credentials not found'):
            jarvislabs_utils.get_client()


class TestGetOrAddSshKey:

    def _client_with_keys(self, existing_keys):
        client = mock.MagicMock()
        client.ssh_keys.list.return_value = existing_keys
        return client

    def _existing_key(self, ssh_key, key_id='k1'):
        key = mock.MagicMock()
        key.ssh_key = ssh_key
        key.key_id = key_id
        return key

    def test_adds_key_when_none_exist(self, monkeypatch):
        """Tests that get_or_add_ssh_key() adds the key when the account
        has no SSH keys registered yet."""
        client = self._client_with_keys([])
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.get_or_add_ssh_key('ssh-ed25519 AAAAB3 user@host')

        client.ssh_keys.add.assert_called_once()
        added_key = client.ssh_keys.add.call_args.args[0]
        assert added_key == 'ssh-ed25519 AAAAB3 user@host'

    def test_skips_add_when_exact_match_present(self, monkeypatch):
        """Tests that get_or_add_ssh_key() is a no-op when an identical key
        is already registered on the account."""
        existing = self._existing_key('ssh-ed25519 AAAAB3 user@host')
        client = self._client_with_keys([existing])
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.get_or_add_ssh_key('ssh-ed25519 AAAAB3 user@host')

        client.ssh_keys.add.assert_not_called()

    def test_skips_add_when_only_trailing_comment_differs(self, monkeypatch):
        """Matching compares just the first two whitespace-separated fields
        (key type + base64 material) of the key being added against each
        existing key's full string, so a differing trailing comment (e.g.
        a different local username@hostname) must not cause a duplicate
        upload."""
        existing = self._existing_key('ssh-ed25519 AAAAB3 old-comment@old-host')
        client = self._client_with_keys([existing])
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.get_or_add_ssh_key(
            'ssh-ed25519 AAAAB3 new-comment@new-host')

        client.ssh_keys.add.assert_not_called()

    def test_adds_key_when_material_differs(self, monkeypatch):
        """Tests that get_or_add_ssh_key() adds the key when no existing
        key's material matches it."""
        existing = self._existing_key('ssh-ed25519 DIFFERENTMATERIAL user@host')
        client = self._client_with_keys([existing])
        monkeypatch.setattr(jarvislabs_utils, 'get_client', lambda: client)

        jarvislabs_utils.get_or_add_ssh_key('ssh-ed25519 AAAAB3 user@host')

        client.ssh_keys.add.assert_called_once()


class TestParseSshCommand:

    def test_extracts_user_and_port(self):
        """Tests that parse_ssh_command() extracts both the user and the
        port when both are present in the string."""
        user, port = jarvislabs_utils.parse_ssh_command(
            'ssh -p 2222 root@1.2.3.4')
        assert user == 'root'
        assert port == 2222

    def test_no_port_present(self):
        """Tests that parse_ssh_command() returns a None port when the
        string has no -p flag."""
        user, port = jarvislabs_utils.parse_ssh_command('ssh root@1.2.3.4')
        assert user == 'root'
        assert port is None

    def test_no_user_present(self):
        """Tests that parse_ssh_command() returns a None user when the
        string has no user@host form."""
        user, port = jarvislabs_utils.parse_ssh_command('ssh -p 22 1.2.3.4')
        assert user is None
        assert port == 22

    def test_empty_or_none_input_returns_none_none(self):
        """Tests that parse_ssh_command() returns (None, None) for both
        empty-string and None input."""
        assert jarvislabs_utils.parse_ssh_command('') == (None, None)
        assert jarvislabs_utils.parse_ssh_command(None) == (None, None)
