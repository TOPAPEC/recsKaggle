import numpy as np, time, sys, pickle, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/workspace/recsKaggle/src")
from common import OUT, DATA, ndcg_at_k
import pandas as pd


def build_model(vocab, d, ml, blocks, heads, dropout):
    return SASRec(vocab, d, ml, blocks, heads, dropout)


class SASRec(nn.Module):
    def __init__(self, vocab, d, ml, blocks, heads, dropout):
        super().__init__()
        self.item = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(ml, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d),
                "attn": nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d),
                "ff1": nn.Linear(d, d * 4),
                "ff2": nn.Linear(d * 4, d),
            }))
        self.lnf = nn.LayerNorm(d)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, x):
        B, L = x.shape
        keep = (x != 0).unsqueeze(-1).float()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.item(x) + self.pos(pos))
        h = h * keep
        cmask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        for b in self.blocks:
            hn = b["ln1"](h)
            a, _ = b["attn"](hn, hn, hn, attn_mask=cmask, need_weights=False)
            h = h + a
            hn = b["ln2"](h)
            h = h + b["ff2"](F.relu(b["ff1"](hn)))
            h = h * keep
        return self.lnf(h)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--mode", default="val")
    P.add_argument("--maxlen", type=int, default=100)
    P.add_argument("--d", type=int, default=256)
    P.add_argument("--blocks", type=int, default=2)
    P.add_argument("--heads", type=int, default=2)
    P.add_argument("--dropout", type=float, default=0.3)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--wd", type=float, default=0.0)
    P.add_argument("--bs", type=int, default=512)
    P.add_argument("--epochs", type=int, default=40)
    P.add_argument("--patience", type=int, default=5)
    P.add_argument("--tag", default="sas")
    P.add_argument("--seed", type=int, default=0)
    args = P.parse_args()
    print(vars(args), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    t0 = time.time()

    src = "seqs.pkl" if args.mode == "val" else "seqs_full.pkl"
    with open(f"{OUT}/{src}", "rb") as f:
        D = pickle.load(f)
    seqs = D["seqs"]
    val_tgt = D.get("val_tgt", None)

    maxitem = 0
    for v in seqs.values():
        if len(v):
            maxitem = max(maxitem, int(v.max()))
    n_items = maxitem + 1
    VOCAB = n_items + 1
    ML = args.maxlen
    users = np.array(list(seqs.keys()), dtype=np.int64)
    N = len(users)
    print("n_items", n_items, "vocab", VOCAB, "users", N, round(time.time() - t0, 1), flush=True)

    INP = np.zeros((N, ML), dtype=np.int64)
    TGT = np.zeros((N, ML), dtype=np.int64)
    EVALPOS = np.zeros(N, dtype=np.int64)
    seen_lists = []
    for r, u in enumerate(users):
        s = seqs[u]
        if len(s) == 0:
            seen_lists.append(np.empty(0, np.int64))
            continue
        w = s[-(ML + 1):].astype(np.int64) + 1
        inp = w[:-1]
        tgt = w[1:]
        INP[r, :len(inp)] = inp
        TGT[r, :len(tgt)] = tgt
        full = s[-ML:].astype(np.int64) + 1
        EVALPOS[r] = len(full) - 1
        seen_lists.append(s.astype(np.int64) + 1)
    print("windows built", round(time.time() - t0, 1), flush=True)

    inp_all = torch.from_numpy(INP).to(dev)
    tgt_all = torch.from_numpy(TGT).to(dev)

    EVALINP = np.zeros((N, ML), dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u]
        if len(s) == 0:
            continue
        full = s[-ML:].astype(np.int64) + 1
        EVALINP[r, :len(full)] = full
    eval_inp = torch.from_numpy(EVALINP).to(dev)
    eval_pos = torch.from_numpy(EVALPOS).to(dev)

    flat = np.concatenate([seqs[u] for u in users if len(seqs[u])])
    pop_sorted = np.argsort(-np.bincount(flat, minlength=n_items))

    model = SASRec(VOCAB, args.d, ML, args.blocks, args.heads, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd)
    print("params", sum(p.numel() for p in model.parameters()), round(time.time() - t0, 1), flush=True)
    scaler = torch.amp.GradScaler("cuda")

    def train_epoch():
        model.train()
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        nb = 0
        for i in range(0, N, args.bs):
            idx = perm[i:i + args.bs]
            x = inp_all[idx]
            y = tgt_all[idx]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                h = model(x)
                mask = (y != 0)
                hv = h[mask]
                yv = y[mask]
                logits = hv @ model.item.weight.T
                logits[:, 0] = -1e4
                loss = F.cross_entropy(logits, yv)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        return tot / nb

    @torch.no_grad()
    def evaluate():
        model.eval()
        E = model.item.weight.detach()
        recs = {}
        B = 1024
        for i in range(0, N, B):
            idx = torch.arange(i, min(i + B, N), device=dev)
            x = eval_inp[idx]
            with torch.amp.autocast("cuda"):
                h = model(x)
                hl = h[torch.arange(x.shape[0], device=dev), eval_pos[idx]]
                scores = (hl @ E.T).float()
            scores[:, 0] = -1e9
            for bi, r in enumerate(idx.tolist()):
                sl = seen_lists[r]
                if len(sl):
                    scores[bi, torch.from_numpy(sl).to(dev)] = -1e9
            top = torch.topk(scores, 20, dim=1).indices.cpu().numpy() - 1
            for bi, r in enumerate(idx.tolist()):
                recs[int(users[r])] = top[bi].tolist()
        return recs

    best = -1
    bad = 0
    for ep in range(1, args.epochs + 1):
        tl = train_epoch()
        if args.mode == "val":
            recs = evaluate()
            nd = ndcg_at_k(recs, val_tgt, 20)
            print(f"ep{ep} loss {tl:.4f} ndcg@20 {nd:.5f} t {round(time.time() - t0, 1)}", flush=True)
            if nd > best:
                best = nd
                bad = 0
                torch.save(model.state_dict(), f"{OUT}/{args.tag}_best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop", flush=True)
                    break
        else:
            print(f"ep{ep} loss {tl:.4f} t {round(time.time() - t0, 1)}", flush=True)

    if args.mode == "val":
        print("BEST ndcg@20", round(best, 5), flush=True)
    else:
        torch.save(model.state_dict(), f"{OUT}/{args.tag}_full.pt")
        recs = evaluate()
        test_users = pd.read_csv(f"{DATA}/test_users.csv").user_id.tolist()
        pop_fill = [int(p) for p in pop_sorted[:200]]
        us, its = [], []
        miss = 0
        for u in test_users:
            r = recs.get(u)
            if r is None:
                r = pop_fill[:20]
                miss += 1
            us.extend([u] * 20)
            its.extend(r[:20])
        pd.DataFrame({"user_id": us, "item_id": its}).to_csv(f"{OUT}/sub_{args.tag}.csv", index=False)
        print(f"wrote sub_{args.tag}.csv miss {miss}", flush=True)
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
