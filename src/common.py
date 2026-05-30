import pandas as pd, numpy as np

OUT = "/workspace/recsKaggle/out"
DATA = "/workspace/recsKaggle/data"


def load_inter():
    return pd.read_parquet(f"{OUT}/inter.pq")


def build_user_hist(df):
    trn = df[df.is_val == 0]
    g = trn.groupby("user_id")["item_id"].apply(lambda s: s.to_numpy())
    return g.to_dict()


def build_val_targets(df):
    val = df[df.is_val == 1]
    g = val.groupby("user_id")["item_id"].apply(lambda s: set(s.tolist()))
    return g.to_dict()


_LOG2 = None


def _disc(k):
    global _LOG2
    if _LOG2 is None or len(_LOG2) < k:
        _LOG2 = 1.0 / np.log2(np.arange(2, k + 2))
    return _LOG2[:k]


def ndcg_at_k(recs, targets, k=20):
    disc = _disc(k)
    tot = 0.0
    n = 0
    for u, tgt in targets.items():
        tset = set(tgt.tolist()) if isinstance(tgt, np.ndarray) else set(tgt)
        if len(tset) == 0:
            continue
        n += 1
        rec = recs.get(u)
        if rec is None or len(rec) == 0:
            continue
        rl = rec[:k]
        gains = np.array([1.0 if it in tset else 0.0 for it in rl])
        dcg = float((gains * disc[:len(gains)]).sum())
        ideal = min(len(tset), k)
        idcg = float(disc[:ideal].sum())
        tot += dcg / idcg if idcg > 0 else 0.0
    return tot / n if n else 0.0


def write_submission(recs, path, test_users):
    us = []
    its = []
    for u in test_users:
        r = recs[u][:20]
        us.extend([u] * len(r))
        its.extend(r)
    out = pd.DataFrame({"user_id": us, "item_id": its})
    out.to_csv(path, index=False)
    return out
