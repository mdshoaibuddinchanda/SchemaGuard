# GPU Execution Policy

The GPU capacity workstream uses one foundation model per worker under the
`gpu-capacity.lock` file lock. Each launch receives a fresh CUDA free-memory
measurement, and the parent process continuously samples worker-tree RSS until
the worker exits or is terminated.

The selected scheduling strategy is `fresh_worker_per_inference`. This is an
execution-safety decision only; frozen model parameters and scientific settings
are unchanged. CUDA OOM, unavailable CUDA, timeout, telemetry failure, and soft
VRAM-limit breaches are explicit capability outcomes and cannot be reported as a
successful GPU profile.
