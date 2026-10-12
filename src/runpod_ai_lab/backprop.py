import torch

from .common import Timer, parser, positive, rate, save, setup


def forward_backward(x, y, params):
    """Two-layer tanh MLP; d(mean((prediction-y)^2))/d(parameters)."""
    w1, b1, w2, b2 = params
    h = torch.tanh(x @ w1 + b1)
    prediction = h @ w2 + b2
    loss = (prediction - y).square().mean()
    dp = 2 * (prediction - y) / y.numel()
    dw2 = h.T @ dp
    db2 = dp.sum(0)
    dz = (dp @ w2.T) * (1 - h.square())
    dw1 = x.T @ dz
    db1 = dz.sum(0)
    return loss, [dw1, db1, dw2, db2]


def main():
    p = parser("Manual chain rule and SGD, checked against autograd")
    p.add_argument("--steps", type=positive, default=500)
    p.add_argument("--hidden", type=positive, default=16)
    p.add_argument("--samples", type=positive, default=256)
    p.add_argument("--lr", type=rate, default=0.1)
    args = p.parse_args()
    device = setup(args)
    x = torch.randn(args.samples, 2, device=device, dtype=torch.float64)
    y = torch.sin(x[:, :1]) + 0.3 * x[:, 1:]
    params = [torch.randn(2, args.hidden, device=device, dtype=x.dtype) * 0.3,
              torch.zeros(args.hidden, device=device, dtype=x.dtype),
              torch.randn(args.hidden, 1, device=device, dtype=x.dtype) * 0.3,
              torch.zeros(1, device=device, dtype=x.dtype)]
    initial, manual = forward_backward(x, y, params)
    reference = [v.clone().requires_grad_() for v in params]
    w1, b1, w2, b2 = reference
    prediction = torch.tanh(x @ w1 + b1) @ w2 + b2
    (prediction - y).square().mean().backward()
    errors = []
    for actual, ref in zip(manual, reference):
        torch.testing.assert_close(actual, ref.grad, rtol=1e-7, atol=1e-9)
        errors.append((actual - ref.grad).abs().max().item())
    print(f"manual/autograd gradient max errors={errors}")
    history = []
    with Timer(device) as timer:
        for step in range(args.steps):
            loss, gradients = forward_backward(x, y, params)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; reduce --lr")
            # No backward(), nn.Module or optimizer in this training loop.
            for param, grad in zip(params, gradients):
                param.sub_(args.lr * grad)
            if step % max(1, args.steps // 10) == 0:
                history.append({"step": step, "loss": loss.item()})
                print(history[-1])
    final, _ = forward_backward(x, y, params)
    save(args, device, "backprop", {"initial_loss": initial.item(), "final_loss": final.item(),
         "gradient_max_errors": errors, "seconds": timer.seconds, "history": history},
         {k: v.cpu() for k, v in zip(["w1", "b1", "w2", "b2"], params)})


if __name__ == "__main__":
    main()
