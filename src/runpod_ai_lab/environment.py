import shutil
import subprocess

import torch

from .common import parser, save, setup


def main():
    p = parser("Inspect CUDA, driver and a real device operation")
    args = p.parse_args()
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=True)
    else:
        print("nvidia-smi not found (CPU-only hosts are supported)")
    print(f"CUDA build={torch.version.cuda}; available={torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}, VRAM={prop.total_memory / 2**30:.2f} GiB, capability={prop.major}.{prop.minor}")
    device = setup(args)
    a = torch.arange(6., device=device).reshape(2, 3)
    result = a @ a.T
    torch.testing.assert_close(result.cpu(), torch.tensor([[5., 14.], [14., 50.]]))
    print(result)
    save(args, device, "environment", {"matmul_check": "passed"})


if __name__ == "__main__":
    main()
