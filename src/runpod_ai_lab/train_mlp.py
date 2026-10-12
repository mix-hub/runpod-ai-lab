import torch
from torch import nn

from .common import Timer, parser, positive, rate, save, setup


def make_data(count, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(count, 2, generator=g) * 2 - 1
    y = (x[:, 0] * x[:, 1] > 0).long()
    return x, y


def build_model(hidden, depth):
    layers = [nn.Linear(2, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    return nn.Sequential(*layers, nn.Linear(hidden, 2))


def main():
    p = parser("Train an MLP on synthetic XOR, evaluate a separate held-out dataset")
    p.add_argument("--epochs", type=positive, default=30)
    p.add_argument("--samples", type=positive, default=4096)
    p.add_argument("--batch-size", type=positive, default=256)
    p.add_argument("--hidden", type=positive, default=32)
    p.add_argument("--depth", type=positive, default=2)
    p.add_argument("--lr", type=rate, default=0.01)
    args = p.parse_args()
    device = setup(args)
    x, y = [v.to(device) for v in make_data(args.samples, args.seed)]
    vx, vy = [v.to(device) for v in make_data(1024, args.seed + 1)]
    model = build_model(args.hidden, args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    with Timer(device) as total:
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(x), device=device)
            loss_sum = torch.zeros((), device=device)
            with Timer(device) as training:
                for indices in order.split(args.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach() * len(indices)
            model.eval()
            with Timer(device) as inference:
                with torch.inference_mode():
                    logits = model(vx)
            with torch.inference_mode():
                val_loss = nn.functional.cross_entropy(logits, vy).item()
                accuracy = (logits.argmax(1) == vy).float().mean().item()
            row = {"epoch": epoch + 1, "train_loss": loss_sum.item() / len(x),
                   "val_loss": val_loss, "val_accuracy": accuracy,
                   "train_seconds": training.seconds, "samples_per_second": len(x) / training.seconds,
                   "inference_seconds": inference.seconds}
            history.append(row)
            print(row)
    save(args, device, "train_mlp", {"seconds": total.seconds, "history": history,
         "parameters": sum(p.numel() for p in model.parameters())},
         {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
          "hidden": args.hidden, "depth": args.depth})


if __name__ == "__main__":
    main()
