# JarvisLabs Feature Support

Status of `sky.clouds.CloudImplementationFeatures` and other cloud-level
capabilities for the JarvisLabs integration. "Verified live" means tested
against real JarvisLabs infra (not just inferred from source); "Confirmed by
source" means checked against the JarvisLabs SDK/backend source but not
independently re-run live.

Source of truth for the hard blocks: `_CLOUD_UNSUPPORTED_FEATURES` in
[`sky/clouds/jarvislabs.py`](../../clouds/jarvislabs.py).

Status legend: ✅ supported · 🟡 partially supported · ❌ not supported ·
⚪ dead/inert flag (declared but not actually enforced anywhere).

## Compute lifecycle

- **Launch / exec / status / down** — ✅ Supported. Verified live: GPU and
  CPU-only instances have been tested for all 3 regions.

- **Stop / start** (`STOP`) — ✅ Supported. Verified live. `STOPPED` maps to
  the real API's `Paused` state. Resume assigns a **new** instance ID on the
  backend even though it's the same logical cluster — harmless, since the
  provisioner discovers instances by name, not a cached ID.

- **Autostop / autodown** (`AUTOSTOP` / `AUTODOWN`) — ✅ Supported. Remote
  skylet self-triggers stop/down via a mounted credentials, since there's no
  cloud-native scheduler to do it externally.

- **Auto-terminate** (`AUTO_TERMINATE`) — ⚪ Dead flag, not applicable.
  Legacy: replaced over a year ago (commit `5540c9d`, 2025-04-11, "Split
  AUTO_TERMINATE to AUTO_DOWN and AUTO_STOP"). It has zero consumers
  anywhere in the codebase today -- the only references left at all are a
  few clouds' (JarvisLabs, Yotta, Hyperbolic) own unsupported-feature
  dicts, all added well after that split, likely copy-pasted from an
  outdated template. The real, actively-checked mechanism is
  `AUTOSTOP`/`AUTODOWN` (see above), which JarvisLabs does support.
  Declaring this unsupported is harmless but has no effect either way --
  there's nothing it currently blocks.

- **Spot instances** (`SPOT_INSTANCE`) — ❌ Not supported. Spot only exists for JarvisLabs'
  *container* workloads, which this integration doesn't use (VM mode
  only).

- **Docker image** (`DOCKER_IMAGE`) — ✅ Supported. Not blocked. GPU
  launches always pass `--gpus all` defensively for nvidia-container-runtime.

## Multi-node / clustering

- **Multi-node clusters** (`MULTI_NODE`) — ❌ Not supported. Enforced
  twice: the feature flag, and an explicit `ValueError` in
  `make_deploy_resources_variables()` if `num_nodes > 1`. **Caveat:**
  JarvisLabs has a real, documented InfiniBand fabric-clustering feature
  (currently H200-only) that isn't wired into this integration at this time. It will be taken up in future.

- **Clone disk from cluster** (`CLONE_DISK_FROM_CLUSTER`) — ❌ Not
  supported.

## Networking

- **Open ports** (`OPEN_PORTS`) — 🟡 Partially supported.
  `OPEN_PORTS_VERSION = LAUNCH_ONLY`: ports can only be set at
  instance-creation time (`http_ports` on `create()`); no confirmed API to
  update them on a running instance.

- **Open ports on CPU-only instances** — 🟡 Partially supported. The real
  API's CPU-VM create path (`create_cpu_vm()`) has no `http_ports`
  parameter at all, so requested ports are silently dropped with a warning
  log (`instance.py`). Verified live via `sky serve`: this turned out
  harmless in practice because JarvisLabs VMs have no firewall blocking
  arbitrary inbound ports, so the raw IP:port still worked -- but that's
  incidental to JarvisLabs' infra, not something SkyPilot's code
  guarantees.

- **Custom network tier** (`CUSTOM_NETWORK_TIER`) — ❌ Not supported.

- **Custom multi-network / multi-NIC** (`CUSTOM_MULTI_NETWORK`) — ❌ Not
  supported.

## Storage

- **Disk size** (`--disk-size`) — ✅ Supported. Bounded to 20GB–5000GB
  (`_MIN_STORAGE_GB`/`_MAX_STORAGE_GB` in `sky/clouds/jarvislabs.py`): the
  lower bound matches the real backend schema's own absolute floor, and
  the upper bound is a deliberately tighter cap, chosen to match the
  bounds shown on JarvisLabs' own website -- already generous enough for
  any realistic workload. Out-of-range requests are rejected up front as
  infeasible (no wasted API calls). Default is 256GB, from SkyPilot's own
  generic, cloud-agnostic `DEFAULT_DISK_SIZE_GB` -- not this cloud's own
  `_DEFAULT_STORAGE_GB` (100), which never actually acts as a default in
  practice (see below). **The real API also enforces a 100GB floor for
  *both* GPU and CPU-only creates** (confirmed live) -- an inherent trait
  of JarvisLabs' own storage system, not a gap in this integration.
  Requests below 100GB are silently clamped up rather than rejected
  outright, with a visible warning logged every time the clamp actually
  changes the value, since it silently increases what the caller is
  billed for.

- **Custom disk tier** (`CUSTOM_DISK_TIER`) — ❌ Not supported. Only a
  single SSD disk type is offered.

- **Local disk** (`LOCAL_DISK`) — ❌ Not supported.

- **Storage mounting** (`STORAGE_MOUNTING`, i.e. `mode: MOUNT`) — ❌ Not
  supported. Verified live: rejected with a clear message before any
  launch attempt. Workaround: use `mode: COPY` to copy object-store data
  to local disk instead of live-mounting it. Planned to be implemented in the future.

- **Volumes** (`sky volumes`) — ❌ Not supported. Not a
  `CloudImplementationFeatures` flag -- `sky volumes` uses a separate,
  hardcoded cloud allowlist (`VOLUME_TYPE_TO_CLOUD` in
  `sky/volumes/volume.py`) that only includes Kubernetes and RunPod.
  Verified live: JarvisLabs is correctly rejected (though via a raw Python
  traceback, not a clean error message -- confirmed to be a pre-existing,
  cloud-agnostic core bug, not specific to this integration).

## Images

- **Custom image ID** (`IMAGE_ID`) — ❌ Not supported. JarvisLabs' own API
  has no image-selection parameter at all; not actionable from SkyPilot's
  side.

## Controllers (managed jobs / SkyServe)

- **Hosting a jobs/serve controller** (`HOST_CONTROLLERS`) — ✅ Supported.
  Not blocked. Verified live: both the managed-jobs controller and the
  SkyServe controller launch and run successfully on JarvisLabs (forced
  via `--config jobs.controller.resources.cloud=jarvislabs` for jobs;
  serve's default controller resources happened to land on JarvisLabs
  too).

- **High-availability controllers** (`HIGH_AVAILABILITY_CONTROLLERS`) — ❌
  Not supported. Only the regular (non-HA) controller path works; a
  controller that needs to auto-restart itself isn't supported.

- **`sky jobs launch` end-to-end** — ✅ Supported. Verified live, after
  fixing two real bugs surfaced by this exact path: the CPU-VM 100GB
  storage floor (the controller's core-default `disk_size=50` was below
  it) and a catalog-propagation regression (the remote controller couldn't
  fetch the not-yet-upstream-published JarvisLabs catalog).

- **`sky serve up` end-to-end** — ✅ Supported. Verified live: controller
  launch, replica launch, readiness probing, load-balancer proxying
  (confirmed via real request routing, not just a 200 status), and all
  three `sky serve logs` targets (controller/load-balancer/replica) all
  work correctly.

## Regions / naming

- **Regions** — `IN1`, `IN2`, `EU1`. Each is handled internally
  by the JarvisLabs SDK's own region resolution -- SkyPilot just passes
  the display code through.

- **Zones** — Not supported. `validate_region_zone()` raises `ValueError`
  if a zone is given at all.

- **Max cluster name length** — 40 characters. Matches the real API's own
  instance-name limit.

- **Max GPUs per instance** — 8. Backend-enforced (`num_gpus <= 8`); the
  catalog fetcher enumerates power-of-two counts (1/2/4/8) up to each GPU
  pool's ceiling.

## Additional features to be implemented for integration (most are already supported by JarvisLabs)

- Containers
- Spots for Containers
- Multi-node clusters
- Storage mounting for Object Stores (upcoming on JarvisLabs)
- Security Groups post launch of Instance [not exposed in SDK yet]
- Security Groups at the time of launch of Instance (upcoming on JarvisLabs)


