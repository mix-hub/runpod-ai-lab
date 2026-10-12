import torch

from .common import Timer, parser, positive, save, setup


def main():
    p = parser("Tensor shapes, broadcasting, strides and synchronized matmul timing")
    p.add_argument("--size", type=positive, default=1024)
    p.add_argument("--repeats", type=positive, default=20)
    p.add_argument("--warmup", type=positive, default=3)
    p.add_argument("--dtype", choices=["float32", "float64", "float16"], default="float32")
    args = p.parse_args()
    device = setup(args)
    dtype = getattr(torch, args.dtype)
    x = torch.arange(6., device=device).reshape(2, 3)
    print(f"x={x}; broadcast x+[1,2,3]={x + torch.tensor([1,2,3], device=device)}")
    print(f"shape={x.shape}, stride={x.stride()}, transpose contiguous={x.T.is_contiguous()}")
    # Transfer is timed separately from the device-resident matrix multiplication.
    a_cpu = torch.randn(args.size, args.size, dtype=dtype)
    b_cpu = torch.randn_like(a_cpu)
    with Timer(device) as transfer:
        a, b = a_cpu.to(device), b_cpu.to(device)
    out = torch.empty_like(a)
    for _ in range(args.warmup):
        torch.mm(a, b, out=out)
    with Timer(device) as timer:
        for _ in range(args.repeats):
            torch.mm(a, b, out=out)
    # Small submatrix reference uses the full reduction dimension.
    reference = a_cpu[:4].double() @ b_cpu[:, :4].double()
    error = (out[:4, :4].cpu().double() - reference).abs().max().item()
    metrics = {"transfer_seconds": transfer.seconds, "matmul_seconds": timer.seconds / args.repeats,
               "gflops": 2 * args.size**3 * args.repeats / timer.seconds / 1e9,
               "reference_max_abs_error": error, "finite": bool(torch.isfinite(out).all().item()),
               "three_matrices_mib": 3 * a.numel() * a.element_size() / 2**20}
    print(metrics)
    save(args, device, "tensors", metrics)


if __name__ == "__main__":
    main()
