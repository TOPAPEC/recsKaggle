import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pickle


class SASRec(nn.Module):
    def __init__(s, vocab, d, ml, blocks, heads, dropout, fix):
        super().__init__()
        s.fix = fix
        s.item = nn.Embedding(vocab, d, padding_idx=0)
        s.pos = nn.Embedding(ml, d)
        s.drop = nn.Dropout(dropout)
        s.blocks = nn.ModuleList()
        for _ in range(blocks):
            s.blocks.append(nn.ModuleDict({'ln1': nn.LayerNorm(d), 'attn': nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True), 'ln2': nn.LayerNorm(d), 'ff1': nn.Linear(d, d * 4), 'ff2': nn.Linear(d * 4, d)}))
        s.lnf = nn.LayerNorm(d)

    def forward(s, x):
        B, L = x.shape
        pad = (x == 0)
        pos = torch.arange(L).unsqueeze(0).expand(B, L)
        h = s.drop(s.item(x) + s.pos(pos))
        cmask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for b in s.blocks:
            hn = b['ln1'](h)
            a, _ = b['attn'](hn, hn, hn, attn_mask=cmask, key_padding_mask=pad, need_weights=False)
            if s.fix:
                a = torch.nan_to_num(a)
            h = h + a
            hn = b['ln2'](h)
            h = h + b['ff2'](F.relu(b['ff1'](hn)))
        return s.lnf(h)


seqs = pickle.load(open('out/seqs.pkl', 'rb'))['seqs']
val_tgt = pickle.load(open('out/seqs.pkl', 'rb'))['val_tgt']
users = list(seqs.keys())
ML = 100
for fix in [False, True]:
    m = SASRec(34323, 256, ML, 2, 2, 0.3, fix)
    m.load_state_dict(torch.load('out/sas_d256_best.pt', map_location='cpu'))
    m.eval()

    def win(u):
        sq = seqs[u]
        w = sq[-ML:].astype(np.int64) + 1
        W = np.zeros(ML, np.int64)
        W[ML - len(w):] = w
        return torch.from_numpy(W).unsqueeze(0)
    print("=== fix =", fix, "===")
    for u in users[:3]:
        with torch.no_grad():
            h = m(win(u))[:, -1, :]
            sc = (h @ m.item.weight.T).float().squeeze(0)
            sc[0] = -1e9
            seen = (seqs[u] + 1)
            sc[seen] = -1e9
            top = (torch.topk(sc, 5).indices.numpy() - 1).tolist()
        print('u', u, 'top5', top, 'std', round(float(sc[1:][sc[1:] > -1e8].std()), 3))
