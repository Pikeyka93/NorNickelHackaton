"""Print GPU visibility. Exits non-zero if CUDA isn't available (helps catch a
missing `--gpus all` at `docker run`)."""
import sys

try:
    import torch
except ImportError:
    print("[check_gpu] torch not installed — CPU-only mode.")
    sys.exit(0)

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    print(f"[check_gpu] OK: {n} GPU(s) visible: {names}")
    sys.exit(0)
else:
    print("[check_gpu] WARNING: CUDA not available. "
          "If in Docker, did you pass `--gpus all`? Falling back to CPU.")
    # Don't hard-fail: the app still runs on CPU (slower). Change to sys.exit(1)
    # if you want a hard guarantee of GPU.
    sys.exit(0)
