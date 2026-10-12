import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def rate(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("runs"))
    return p


def setup(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check nvidia-smi and your PyTorch CUDA build")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"device={device}, torch={torch.__version__}")
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory(device):
    if device.type != "cuda":
        return {"allocated_mib": None, "reserved_mib": None, "peak_allocated_mib": None}
    return {"allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self.start


def save(args, device, experiment, metrics, state=None):
    folder = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + experiment)
    folder.mkdir(parents=True, exist_ok=False)
    report = {"experiment": experiment, "config": {k: str(v) if isinstance(v, Path) else v
               for k, v in vars(args).items()}, "environment": {
                   "python": platform.python_version(), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "memory": memory(device), "metrics": metrics}
    (folder / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if state is not None:
        torch.save(state, folder / "model.pt")
    print(f"results={folder}")
    print(json.dumps(report["memory"]))
    return folder
