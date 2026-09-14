# Phase 02B GPU Execution Decision

The bounded GPU exploration passed for both frozen foundation models on the RTX 3050 Laptop GPU.

- Primary execution policy: one foundation model per subprocess under the GPU lock.
- CUDA profile envelope: peak reserved VRAM remained below the 3,600 MiB soft limit for all successful profiles.
- Cleanup evidence: isolated workers returned to near-zero allocated model memory after `empty_cache()` and process exit.
- Five-cycle medium-profile repeated inference passed with GPU maximum absolute difference `0.0` and probability simplex error within `1e-6`.
- The isolated-process candidate was not adopted as a scientific or model-configuration change. It remains the scheduling safety mechanism because it contains crashes and memory growth.
- CUDA OOM and missing-CUDA behavior are classified as capability outcomes and trigger the deterministic CPU fallback path.

No checkpoint, frozen constructor parameter, dataset, split, or scientific configuration was changed.
