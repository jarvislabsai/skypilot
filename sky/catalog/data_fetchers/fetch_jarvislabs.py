"""A script that generates the JarvisLabs catalog.

Usage:
    python fetch_jarvislabs.py

Requires JarvisLabs credentials resolved the same way as sky.clouds.jarvislabs
(JL_API_KEY environment variable, or the config file written by `jl setup`).
This script always requests USD pricing via the `X-Client-Id: skypilot`
header (see `_headers()`), regardless of the calling account's own billing
country -- see the currency note below.

NOTE: `GET /misc/server_meta` may change wihtout notice. if this script starts
failing, re-check that package's `server_meta.py`/`models.py`/`constants.py`
for the current shape.

Data model, confirmed against a real response (not assumed from docs):
  * The response is scoped to whatever the calling account/key can currently
    see -- it's a live routing/availability endpoint the SDK uses to pick a
    region at launch time, not a stable master price list. A quiet API key
    may see fewer GPU types than are generally offered; that's expected, and
    this script doesn't try to fill in gaps itself.
  * `currency` (top-level) is normally account-dependent: `Account.currency()`
    in the SDK returns 'INR' or 'USD' "based on user's payment location", and
    all prices in the response (GPU and CPU) are in that currency. This
    script instead sends `X-Client-Id: skypilot`. The
    `currency != 'USD'` check below is therefore a backstop against a
    backend regression, not something fixable by switching accounts -- a
    hardcoded/looked-up FX rate would drift and silently misprice the
    catalog, so this script refuses to guess instead of converting.
  * Each `server_meta` GPU entry describes one (gpu_type, region,
    workload_type) *pool*, and its `num_gpus` (absent = 1) is the pool's
    ceiling -- NOT a fixed bundle you must take as a whole. Any GPU count
    from 1 up to that ceiling is independently launchable against the same
    pool: For example, IN2 only ever lists an 8-GPU H200 pool in
    server_meta, yet `client.instances.create(gpu_type='H200', num_gpus=1,
    region='IN2')` succeeds directly against the real API. So this fetcher
    enumerates power-of-two counts up to each pool's ceiling rather than
    emitting a single row at the pool's exact listed size (matching the
    hand-curated catalog this replaced, which already did this).
  * `cpus_per_gpu`/`ram_per_gpu`/`price_per_hour` are all genuinely *per
    GPU* and need multiplying by the requested count -- confirmed against
    known USD prices (e.g. an 8-GPU L4 pool's `price_per_hour` equals the
    public single-L4 price, and `price_per_hour * 8` equals the previously
    hand-verified 8x price).
  * A GPU's real `gpu_type` string is translated to SkyPilot's catalog
    display name via `sky.provision.jarvislabs.utils.to_skypilot_gpu_type()`
    before being written into any row -- see that function's module for
    why (RTX-PRO6000 vs. RTXPRO6000).
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import requests

from sky.provision.jarvislabs import utils as jarvislabs_utils

logger = logging.getLogger(__name__)

# India-Noida is the SDK's own default (jarvislabs.constants.DEFAULT_REGION).
API_BASE_URL = 'https://backendn.jarvislabs.net'

# Internal region id -> SkyPilot-facing display code, mirroring
# jarvislabs.constants.REGION_DISPLAY_CODES in the `jarvislabs` package.
REGION_DISPLAY_CODES = {
    'india-chennai-01': 'IN1',
    'india-noida-01': 'IN2',
    'europe-01': 'EU1',
}

# We currently support vm-mode instances only. `workload_type` is `None` for
# some regions (e.g. europe-01) that have no dedicated 'vm' pool at all --
# confirmed live that these are not actually launchable, so `None` must not
# be treated as equivalent to 'vm'.
_VM_WORKLOAD_TYPES = ('vm',)


def _headers(api_key: str) -> Dict[str, str]:
    # X-Client-Id tells the backend to always return USD pricing here,
    # regardless of this account's own billing country -- see the currency
    # note in the module docstring.
    return {'Authorization': f'Bearer {api_key}', 'X-Client-Id': 'SkyPilot'}


def fetch_server_meta(api_key: str) -> Dict[str, Any]:
    """Fetches the raw `misc/server_meta` payload (GPU + CPU-VM catalog)."""
    endpoint = f'{API_BASE_URL}/misc/server_meta'
    try:
        response = requests.get(endpoint, headers=_headers(api_key), timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f'Failed to fetch server_meta: {e}') from e


def make_gpu_info_json(gpu_name: str, gpu_count: int,
                       gpu_memory_gb: float) -> str:
    """Create the GpuInfo JSON string, matching the other catalogs' format."""
    gpu_memory_mib = int(gpu_memory_gb * 1024)
    gpu_info = {
        'Gpus': [{
            'Name': gpu_name,
            'Manufacturer': 'Nvidia',
            'Count': gpu_count,
            'MemoryInfo': {
                'SizeInMiB': gpu_memory_mib
            },
        }],
        'TotalGpuMemoryInMiB': gpu_memory_mib * gpu_count,
    }
    return json.dumps(gpu_info).replace('"', '\'')


# GPU counts to materialize per pool, capped at that pool's own ceiling
# (`num_gpus` on the server_meta entry). Power-of-two rather than every
# integer 1..ceiling, matching the hand-curated catalog's own convention --
# not a hard API limit (unverified whether e.g. num_gpus=3 would also work).
_GPU_COUNTS = (1, 2, 4, 8)


def gpu_rows(server_meta: List[Dict[str, Any]]) -> List[List[Any]]:
    """Expands VM-workload `server_meta` GPU pools into per-count catalog
    rows. See the module docstring for why a pool's ceiling isn't a single
    fixed row.
    """
    # (gpu_type, display_region) -> the entry with the largest ceiling seen,
    # in case multiple pools are listed for the same (gpu_type, region).
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for gpu in server_meta:
        if gpu.get('workload_type') not in _VM_WORKLOAD_TYPES:
            continue
        region = gpu.get('region')
        display_region = REGION_DISPLAY_CODES.get(region) if region else None
        gpu_type = gpu.get('gpu_type')
        price_per_gpu = gpu.get('price_per_hour')
        cpus_per_gpu = gpu.get('cpus_per_gpu')
        ram_per_gpu = gpu.get('ram_per_gpu')
        vram_gb = gpu.get('vram')
        if (display_region is None or not gpu_type or price_per_gpu is None or
                cpus_per_gpu is None or ram_per_gpu is None or not vram_gb):
            logger.warning('Skipping incomplete/unrecognized GPU row: %s', gpu)
            continue
        key = (gpu_type, display_region)
        ceiling = int(gpu.get('num_gpus') or 1)
        existing = best.get(key)
        if existing is None or ceiling > int(existing.get('num_gpus') or 1):
            best[key] = gpu

    rows = []
    for (gpu_type, display_region), gpu in best.items():
        display_gpu_type = jarvislabs_utils.to_skypilot_gpu_type(gpu_type)
        ceiling = int(gpu.get('num_gpus') or 1)
        price_per_gpu = gpu['price_per_hour']
        cpus_per_gpu = int(gpu['cpus_per_gpu'])
        ram_per_gpu = int(gpu['ram_per_gpu'])
        vram_gb = float(gpu['vram'])

        for num_gpus in _GPU_COUNTS:
            if num_gpus > ceiling:
                break
            vcpus = num_gpus * cpus_per_gpu
            memory_gib = num_gpus * ram_per_gpu
            instance_type = (
                f'{num_gpus}x_{display_gpu_type}_{vcpus}V_{memory_gib}G')
            rows.append([
                instance_type,
                display_gpu_type,
                num_gpus,
                # Written as a float (not `vcpus` the int) so the CSV column
                # is "28.0"-style text: pandas then infers float64 dtype on
                # load even though this column is never actually missing
                # any values, matching every other cloud's own fetcher (see
                # e.g. fetch_yotta.py/fetch_runpod.py) and avoiding a display
                # bug in sky/client/cli/command.py, where numpy.int64 (what
                # an all-integer-looking column round-trips to) fails an
                # isinstance(cpu_count, (float, int)) check that
                # numpy.float64 passes.
                float(vcpus),
                memory_gib,
                make_gpu_info_json(display_gpu_type, num_gpus, vram_gb),
                display_region,
                # VM-mode instances don't support spot (spot is
                # container-only; see jarvislabs.cli.render's
                # display_spot_price in the SDK), so leave SpotPrice blank
                # even if the source row has a numeric spot_price.
                '',
                round(price_per_gpu * num_gpus, 4),
            ])
    return rows


def cpu_rows(cpu_meta: Dict[str, Any]) -> List[List[Any]]:
    """Expands `cpu_meta.combinations` into per-region CPU-only catalog
    rows. Unlike GPUs, these are fixed plans defined by the backend, not a
    range we choose."""
    rows = []
    for combo in cpu_meta.get('combinations', []):
        vcpus = combo.get('vcpus')
        ram_gb = combo.get('ram_gb')
        price = combo.get('price_per_hour', combo.get('price'))
        if vcpus is None or ram_gb is None or price is None:
            logger.warning('Skipping incomplete CPU VM plan: %s', combo)
            continue
        for region, available in (combo.get('regions') or {}).items():
            display_region = REGION_DISPLAY_CODES.get(region)
            if not available or display_region is None:
                continue
            instance_type = f'CPU_{int(vcpus)}V_{int(ram_gb)}G'
            rows.append([
                instance_type,
                '',
                '',
                # float, not int -- see the matching comment in gpu_rows().
                float(vcpus),
                int(ram_gb),
                '',
                display_region,
                '',
                price
            ])
    return rows


def create_catalog(output_path: str = 'jarvislabs/vms.csv') -> None:
    """Create JarvisLabs catalog CSV file."""
    api_key = jarvislabs_utils.resolve_api_key()
    assert api_key is not None, (
        f'JarvisLabs credentials not found. Set {jarvislabs_utils.ENV_API_KEY}'
        ' or run `jl setup` to write '
        f'{jarvislabs_utils.get_credentials_path()}.')

    logger.info('Fetching JarvisLabs server_meta...')
    meta = fetch_server_meta(api_key)

    currency = meta.get('currency')
    if currency != 'USD':
        raise RuntimeError(
            f'This JarvisLabs account returned {currency!r} prices. '
            'But the catalog requires USD prices. Contact JarvisLabs '
            'at support@jarvislabs.ai to resolve this.')

    rows = gpu_rows(meta.get('server_meta', [])) + cpu_rows(
        meta.get('cpu_meta', {}))
    logger.info('Generated %d catalog rows.', len(rows))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow([
            'InstanceType',
            'AcceleratorName',
            'AcceleratorCount',
            'vCPUs',
            'MemoryGiB',
            'GpuInfo',
            'Region',
            'SpotPrice',
            'Price',
        ])
        writer.writerows(rows)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    os.makedirs('jarvislabs', exist_ok=True)
    create_catalog('jarvislabs/vms.csv')
    logger.info('JarvisLabs catalog saved to jarvislabs/vms.csv')
