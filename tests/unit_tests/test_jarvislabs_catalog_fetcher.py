"""Tests for the JarvisLabs catalog fetcher (fetch_jarvislabs.py).

Exercises gpu_rows()/cpu_rows()/_headers()/create_catalog() directly against
plain dicts standing in for a `server_meta` API response -- no client or
network mocking needed, since the fetcher's own transformation logic is
pure-Python.
"""
import pytest

from sky.catalog.data_fetchers import fetch_jarvislabs


def _gpu_entry(gpu_type='A100-80GB',
               region='india-noida-01',
               workload_type='vm',
               price_per_hour=1.5,
               cpus_per_gpu=16,
               ram_per_gpu=112,
               vram=80,
               num_gpus=None):
    entry = {
        'gpu_type': gpu_type,
        'region': region,
        'workload_type': workload_type,
        'price_per_hour': price_per_hour,
        'cpus_per_gpu': cpus_per_gpu,
        'ram_per_gpu': ram_per_gpu,
        'vram': vram,
    }
    if num_gpus is not None:
        entry['num_gpus'] = num_gpus
    return entry


class TestHeaders:

    def test_sends_bearer_token_and_client_id(self):
        """Tests that the request headers carry the bearer token and
        X-Client-Id."""
        headers = fetch_jarvislabs._headers('my-api-key')
        assert headers['Authorization'] == 'Bearer my-api-key'
        # Case-insensitive: the backend lowercases this before comparing.
        assert headers['X-Client-Id'].lower() == 'skypilot'


class TestGpuRows:

    def test_single_gpu_pool_below_power_of_two_ceiling(self):
        """A pool whose ceiling isn't a power of two only emits rows up to
        the largest power of two <= ceiling (matches _GPU_COUNTS logic)."""
        rows = fetch_jarvislabs.gpu_rows([_gpu_entry(num_gpus=3)])
        counts = sorted(row[2] for row in rows)
        assert counts == [1, 2]  # not 3 -- 4 exceeds the ceiling

    def test_ceiling_absent_defaults_to_one(self):
        """num_gpus absent from the entry means a ceiling of 1 (per module
        docstring: "absent = 1")."""
        rows = fetch_jarvislabs.gpu_rows([_gpu_entry()])
        assert len(rows) == 1
        assert rows[0][2] == 1  # AcceleratorCount

    def test_price_scales_per_gpu_count(self):
        """price_per_hour is per-GPU; each row's Price must be
        price_per_hour * that row's GPU count."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(price_per_hour=0.44, num_gpus=8)])
        price_by_count = {row[2]: row[8] for row in rows}
        assert price_by_count[1] == 0.44
        assert price_by_count[8] == pytest.approx(0.44 * 8)

    def test_region_display_code_mapping(self):
        """Tests that internal region ids map to their SkyPilot display
        codes."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(region='europe-01', num_gpus=1)])
        assert rows[0][6] == 'EU1'  # Region column

    def test_container_workload_type_skipped(self):
        """Tests that container-workload entries are skipped (VM mode
        only)."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(workload_type='container')])
        assert rows == []

    def test_unrecognized_region_skipped(self):
        """Tests that an entry with an unmapped region id is skipped."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(region='some-unmapped-region')])
        assert rows == []

    def test_incomplete_entry_skipped(self):
        """Tests that an entry missing a required field is skipped."""
        entry = _gpu_entry()
        del entry['price_per_hour']
        assert fetch_jarvislabs.gpu_rows([entry]) == []

    def test_duplicate_pool_keeps_larger_ceiling(self):
        """Two pools for the same (gpu_type, region) keep whichever has the
        larger num_gpus ceiling."""
        rows = fetch_jarvislabs.gpu_rows([
            _gpu_entry(num_gpus=1),
            _gpu_entry(num_gpus=8),
        ])
        counts = sorted(row[2] for row in rows)
        assert counts == [1, 2, 4, 8]


class TestGpuRowsDisplayNameTranslation:
    """The catalog must show JarvisLabs' RTX-PRO6000 under the same
    no-hyphen spelling GCP/RunPod use (sky/catalog/__init__.py's
    get_common_gpus()), even though the real API reports it hyphenated --
    see sky.provision.jarvislabs.utils.JARVISLABS_TO_SKYPILOT_GPU_TYPES."""

    def test_hyphenated_gpu_type_translated_in_accelerator_name(self):
        """Tests that the hyphen is stripped in the AcceleratorName
        column."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(gpu_type='RTX-PRO6000', num_gpus=1)])
        assert len(rows) == 1
        assert rows[0][1] == 'RTXPRO6000'  # AcceleratorName

    def test_hyphenated_gpu_type_translated_in_instance_type(self):
        """Tests that the hyphen is stripped in the InstanceType
        string."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(gpu_type='RTX-PRO6000', num_gpus=1)])
        assert 'RTXPRO6000' in rows[0][0]  # InstanceType
        assert 'RTX-PRO6000' not in rows[0][0]

    def test_hyphenated_gpu_type_translated_in_gpu_info_json(self):
        """Tests that the hyphen is stripped in the embedded GpuInfo
        JSON."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(gpu_type='RTX-PRO6000', num_gpus=1)])
        assert "'Name': 'RTXPRO6000'" in rows[0][5]  # GpuInfo
        assert 'RTX-PRO6000' not in rows[0][5]

    def test_unmapped_gpu_type_left_unchanged(self):
        """Tests that a GPU type with no translation entry is left
        as-is."""
        rows = fetch_jarvislabs.gpu_rows(
            [_gpu_entry(gpu_type='H100', num_gpus=1)])
        assert rows[0][1] == 'H100'


class TestCpuRows:

    def _combo(self, vcpus=8, ram_gb=32, price_per_hour=0.1984, regions=None):
        return {
            'vcpus': vcpus,
            'ram_gb': ram_gb,
            'price_per_hour': price_per_hour,
            'regions': regions or {
                'india-noida-01': True
            },
        }

    def test_basic_expansion(self):
        """Tests that a single CPU combo expands into one correctly
        populated row."""
        rows = fetch_jarvislabs.cpu_rows({'combinations': [self._combo()]})
        assert len(rows) == 1
        row = rows[0]
        assert row[0] == 'CPU_8V_32G'  # InstanceType
        assert row[3] == 8  # vCPUs
        assert row[4] == 32  # MemoryGiB
        assert row[6] == 'IN2'  # Region
        assert row[8] == 0.1984  # Price

    def test_unavailable_region_skipped(self):
        """Tests that a region marked unavailable for a combo is
        skipped."""
        rows = fetch_jarvislabs.cpu_rows(
            {'combinations': [self._combo(regions={'india-noida-01': False})]})
        assert rows == []

    def test_multiple_regions_expand_to_multiple_rows(self):
        """Tests that a combo available in multiple regions produces one
        row per region."""
        rows = fetch_jarvislabs.cpu_rows({
            'combinations': [
                self._combo(regions={
                    'india-noida-01': True,
                    'india-chennai-01': True,
                })
            ]
        })
        regions = sorted(row[6] for row in rows)
        assert regions == ['IN1', 'IN2']

    def test_incomplete_combo_skipped(self):
        """Tests that a combo missing a required field is skipped."""
        combo = self._combo()
        del combo['vcpus']
        assert fetch_jarvislabs.cpu_rows({'combinations': [combo]}) == []

    def test_no_combinations_key_returns_empty(self):
        """Tests that a payload with no combinations key returns no
        rows."""
        assert fetch_jarvislabs.cpu_rows({}) == []


class TestVcpuColumnIsFloat:
    """The vCPUs column must be written as float (e.g. `28.0`, not `28`) so
    it round-trips through pandas as float64, not int64 -- avoids a
    numpy.int64/isinstance display bug in the CLI (see
    sky/clouds/jarvislabs.py). InstanceType names still keep the plain
    integer (`28V`, not `28.0V`)."""

    def test_gpu_rows_vcpus_is_float(self):
        """Tests that gpu_rows() writes the vCPUs value as a float."""
        rows = fetch_jarvislabs.gpu_rows([_gpu_entry(num_gpus=1)])
        assert len(rows) == 1
        vcpus = rows[0][3]
        assert isinstance(vcpus, float)
        assert not isinstance(vcpus, int)
        assert vcpus == 16  # cpus_per_gpu default * 1 gpu

    def test_gpu_rows_instance_type_name_keeps_plain_integer(self):
        """Tests that the InstanceType name still uses a plain integer,
        not the float representation."""
        rows = fetch_jarvislabs.gpu_rows([_gpu_entry(num_gpus=1)])
        assert '16V' in rows[0][0]
        assert '16.0V' not in rows[0][0]

    def test_cpu_rows_vcpus_is_float(self):
        """Tests that cpu_rows() writes the vCPUs value as a float."""
        rows = fetch_jarvislabs.cpu_rows({
            'combinations': [{
                'vcpus': 8,
                'ram_gb': 32,
                'price_per_hour': 0.1984,
                'regions': {
                    'india-noida-01': True
                },
            }]
        })
        assert len(rows) == 1
        vcpus = rows[0][3]
        assert isinstance(vcpus, float)
        assert not isinstance(vcpus, int)
        assert vcpus == 8

    def test_cpu_rows_instance_type_name_keeps_plain_integer(self):
        """Tests that the InstanceType name still uses a plain integer,
        not the float representation."""
        rows = fetch_jarvislabs.cpu_rows({
            'combinations': [{
                'vcpus': 8,
                'ram_gb': 32,
                'price_per_hour': 0.1984,
                'regions': {
                    'india-noida-01': True
                },
            }]
        })
        assert rows[0][0] == 'CPU_8V_32G'

    def test_written_csv_column_parses_as_float64_dtype(self, monkeypatch,
                                                        tmp_path):
        """End-to-end: a real create_catalog() run must produce a CSV whose
        vCPUs column pandas infers as float64, not int64."""
        pd = pytest.importorskip('pandas')
        monkeypatch.setattr(fetch_jarvislabs.jarvislabs_utils,
                            'resolve_api_key', lambda: 'fake-key')
        monkeypatch.setattr(
            fetch_jarvislabs, 'fetch_server_meta', lambda api_key: {
                'currency': 'USD',
                'server_meta': [_gpu_entry(num_gpus=1)],
                'cpu_meta': {}
            })

        output_path = tmp_path / 'vms.csv'
        fetch_jarvislabs.create_catalog(str(output_path))

        df = pd.read_csv(output_path)
        assert df['vCPUs'].dtype == 'float64'


class TestCreateCatalogCurrencyGuard:

    def test_non_usd_currency_raises(self, monkeypatch, tmp_path):
        """Tests that create_catalog() raises if the API response isn't
        priced in USD."""
        monkeypatch.setattr(fetch_jarvislabs.jarvislabs_utils,
                            'resolve_api_key', lambda: 'fake-key')
        monkeypatch.setattr(
            fetch_jarvislabs, 'fetch_server_meta', lambda api_key: {
                'currency': 'INR',
                'server_meta': [],
                'cpu_meta': {}
            })

        with pytest.raises(RuntimeError, match='INR'):
            fetch_jarvislabs.create_catalog(str(tmp_path / 'vms.csv'))

    def test_usd_currency_proceeds(self, monkeypatch, tmp_path):
        """Tests that create_catalog() writes a valid CSV when priced in
        USD."""
        monkeypatch.setattr(fetch_jarvislabs.jarvislabs_utils,
                            'resolve_api_key', lambda: 'fake-key')
        monkeypatch.setattr(
            fetch_jarvislabs, 'fetch_server_meta', lambda api_key: {
                'currency': 'USD',
                'server_meta': [_gpu_entry()],
                'cpu_meta': {}
            })

        output_path = tmp_path / 'vms.csv'
        fetch_jarvislabs.create_catalog(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert 'InstanceType' in content  # header row
        assert 'A100-80GB' in content
