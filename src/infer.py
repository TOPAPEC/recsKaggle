import numpy as np, time, sys, pickle, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/workspace/recsKaggle/src")
from common import OUT, DATA
import pandas as pd

P = argparse.ArgumentParser()
P.add_argument("--ckpt", required=True)
P.add_argument("--d", type=int, default=256)
P.add_argument("--blocks", type=int, default=2)
P.add_argument("--heads", type=int, default=2)
P.add_argument("--maxlen", type=int, default=100)
P.add_argument("--dropout", type=float, default=0.3)
P.add_argument("--device", default="cpu")
P.add_argument("--bs", type=int, default=512)
P.add_argument("--tag", default="infer")
args = P.parse_args()
print(vars(args), flush=True)
dev = args.device

t0 = time.time()
with open(f"{OUT}/seqs_full.pkl", "rb") as f:
    seqs = pickle.load(f)["seqs"]
maxitem = 0
for v in seqs.values():
    if len(v):
        maxitem = max(maxitem, int(v.max()))
n_items = maxitem + 1
VOCAB = n_items + 1
flat = np.concatenate([seqs[u] for u in seqs if len(seqs[u])])
pop_sorted = np.argsort(-np.bincount(flat, minlength=n_items))
print("vocab", VOCAB, round(time.time() - t0, 1), flush=True)

test_users = pd.read_csv(f"{DATA}/test_users.csv").user_id.tolist()
ML = args.maxlen
N = len(test_users)
W = np.zeros((N, ML), dtype=np.int64)
seen_lists = []
for r, u in enumerate(test_users):
    s = seqs.get(u)
    if s is None or len(s) == 0:
        seen_lists.append(np.empty(0, np.int64)); continue
    w = s[-ML:].astype(np.int64) + 1
    W[r, ML - len(w):] = w
    seen_lists.append(s.astype(np.int64) + 1)
inp = torch.from_numpy(W)
print("windows", round(time.time() - t0, 1), flush=True)


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

    def forward(self, x):
        B, L = x.shape
        pad = (x == 0)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.item(x) + self.pos(pos))
        cmask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        for b in self.blocks:
            hn = b["ln1"](h)
            a, _ = b["attn"](hn, hn, hn, attn_mask=cmask, key_padding_mask=pad, need_weights=False)
            h = h + a
            hn = b["ln2"](h)
            h = h + b["ff2"](F.relu(b["ff1"](hn)))
        return self.lnf(h)


model = SASRec(VOCAB, args.d, ML, args.blocks, args.heads, args.dropout).to(dev)
model.load_state_dict(torch.load(args.ckpt, map_location=dev))
model.eval()
E = model.item.weight.detach()
print("model loaded", round(time.time() - t0, 1), flush=True)

recs = {}
B = args.bs
with torch.no_grad():
    for i in range(0, N, B):
        x = inp[i:i + B].to(dev)
        h = model(x)[:, -1, :]
        sc = (h @ E.T).float()
        sc[:, 0] = -1e9
        for bi in range(x.shape[0]):
            s = seen_lists[i + bi]
            if len(s):
                sc[bi, torch.from_numpy(s).to(dev)] = -1e9
        top = torch.topk(sc, 20, dim=1).indices.cpu().numpy() - 1
        for bi in range(x.shape[0]):
            recs[test_users[i + bi]] = top[bi].tolist()
        if (i // B) % 50 == 0:
            print("batch", i, round(time.time() - t0, 1), flush=True)

pop_fill = [int(p) for p in pop_sorted[:100]]
us, its = [], []; miss = 0
for u in test_users:
    r = recs.get(u)
    if r is None:
        r = pop_fill[:20]; miss += 1
    us.extend([u] * 20); its.extend(r[:20])
pd.DataFrame({"user_id": us, "item_id": its}).to_csv(f"{OUT}/sub_{args.tag}.csv", index=False)
print(f"wrote sub_{args.tag}.csv miss {miss}", round(time.time() - t0, 1), flush=True)
