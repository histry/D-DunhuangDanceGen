"""Resolve the registered successful stage; never pick outputs by quality or modification time."""
from pathlib import Path

def factorial_dir(root):
    root=Path(root)
    return root/("factorial_02" if (root/"recovery_inputs.json").is_file() else "factorial")
