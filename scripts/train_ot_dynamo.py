import argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.control_sde import ControlNet, dsm_loss, control_cost, reverse_sample
from models.dynamo_field import DynamoAdapter, make_lambda_schedule
from models.ot_guidance import minibatch_ot_loss

# ---- Minimal score net stub; replace with your real one / import ----
class ScoreNet(nn.Module):
    def __init__(self, x_dim, cond_dim=0, hidden=1024):
        super().__init__()
        self.cond_dim = cond_dim
        self.fc_t = nn.Linear(128, 256)
        self.fc_c = nn.Linear(cond_dim, 256) if cond_dim>0 else None
        self.net = nn.Sequential(
            nn.Linear(x_dim + 256 + (256 if cond_dim>0 else 0), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim)
        )
    def time_embed(self, t, dim=128):
        device, half = t.device, dim//2
        freqs = torch.exp(torch.linspace(0, 6, half, device=device))
        ang = t[:, None] * freqs[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    def forward(self, x, t, c=None):
        te = self.fc_t(self.time_embed(t))
        if self.cond_dim>0 and c is not None:
            ce = self.fc_c(c)
            h = torch.cat([x, te, ce], dim=-1)
        else:
            h = torch.cat([x, te], dim=-1)
        return self.net(h)

def dummy_loader(x_dim=128, cond_dim=0, n=4096, batch=128):
    x0 = torch.randn(n, x_dim)
    x1 = x0 + 0.5 + 0.1*torch.randn_like(x0)  # pretend "drug" is shifted
    c  = torch.zeros(n, cond_dim) if cond_dim>0 else torch.zeros(n, 0)
    ds = TensorDataset(x0, c, x1)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-dim", type=int, default=128)
    ap.add_argument("--cond-dim", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ot-weight", type=float, default=0.1)
    ap.add_argument("--use-dynamo", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    loader = dummy_loader(args.x_dim, args.cond_dim, batch=args.batch_size)

    score = ScoreNet(args.x_dim, args.cond_dim).to(device)
    u_net = ControlNet(args.x_dim, args.cond_dim).to(device)
    opt = torch.optim.AdamW(list(score.parameters()) + list(u_net.parameters()), lr=args.lr)

    dyn = DynamoAdapter(callable_field=None, device=device)  # plug your real Dynamo call later
    lam_sched = make_lambda_schedule("cosine", 1.0, 0.1)

    for epoch in range(args.epochs):
        for x0, c, x1 in loader:
            x0, x1 = x0.to(device), x1.to(device)
            c = c.to(device) if c.numel() > 0 else None

            loss_score, aux = dsm_loss(score, x0, c)
            xt, t = aux["xt"], aux["t"]

            # Align u to Dynamo field (no-op if dyn is zero)
            lam_t = lam_sched(float(t.mean().item()))
            vf_loss = F.mse_loss(u_net(xt, c), lam_t * dyn.predict(xt))

            # Control cost (SB flavor)
            u_cost = control_cost(u_net, xt, c, w=1e-3)

            # OT endpoint matching via quick reverse samples
            with torch.no_grad():
                x_gen = reverse_sample(score, u_net, x_init=x0, c_drug=c, n_steps=64,
                                       lam_sched=lam_sched, f_dyn=dyn.predict if args.use_dynamo else None)
            ot_loss = minibatch_ot_loss(x_gen, x1, eps=0.05, p=2, iters=50)

            loss = loss_score + vf_loss + u_cost + args.ot_weight * ot_loss
            opt.zero_grad(); loss.backward(); opt.step()

        print(f"[epoch {epoch+1}] score={loss_score.item():.4f} vf={vf_loss.item():.4f} ot={ot_loss.item():.4f}")

if __name__ == "__main__":
    main()
PY

