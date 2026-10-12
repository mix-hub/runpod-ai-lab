"""RunPod GPU 환경 확인: 사전 설치된 PyTorch만 사용한다."""
import shutil
import subprocess

import torch


def main():
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU를 사용할 수 없습니다. RunPod GPU 및 Jupyter 커널을 확인하세요.")

    for i in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {info.name}, VRAM {info.total_memory / 2**30:.2f} GiB")

    x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
    result = x @ x.T
    torch.testing.assert_close(result.cpu(), torch.tensor([[5., 14.], [14., 50.]]))
    print("GPU 행렬곱 결과:")
    print(result)


if __name__ == "__main__":
    main()
