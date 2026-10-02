"""Print installed versions for one env: versions.py <env-label> <package>..."""

import sys
from importlib import metadata

label, packages = sys.argv[1], sys.argv[2:]
print(f"[{label}] python {sys.version.split()[0]}")
for name in packages:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    if name == "torch" and version != "not installed":
        import torch

        version = f"{torch.__version__} (CUDA {torch.version.cuda})"
    print(f"[{label}] {name:<14} {version}")
