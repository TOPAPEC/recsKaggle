import numpy as np, sys, pickle, time
sys.path.insert(0, "/workspace/recsKaggle/src")
from common import OUT, DATA, ndcg_at_k
import pandas as pd

t0 = time.time()
df = pd.read_parquet(f"{OUT}/inter.pq")
print("loaded", df.shape, round(time.time() - t0, 1), flush=True)

trn = df[df.is_val == 0]
pop_trn = trn.item_id.value_counts().index.to_numpy()
pop_all = df.item_id.value_counts().index.to_numpy()

with open(f"{OUT}/seqs.pkl", "rb") as f:
    Dv = pickle.load(f)
seqs_tr = Dv["seqs"]; val_tgt = Dv["val_tgt"]

val_users = list(val_tgt.keys())
recs = {}
for u in val_users:
    seen = set(seqs_tr.get(u, np.empty(0)).tolist())
    rec = []
    for it in pop_trn:
        it = int(it)
        if it not in seen:
            rec.append(it)
        if len(rec) >= 20:
            break
    recs[u] = rec
print("POP val NDCG@20:", round(ndcg_at_k(recs, val_tgt, 20), 5), flush=True)

with open(f"{OUT}/seqs_full.pkl", "rb") as f:
    seqs_full = pickle.load(f)["seqs"]
test_users = pd.read_csv(f"{DATA}/test_users.csv").user_id.tolist()
pop_list = [int(x) for x in pop_all]
us, its = [], []
for u in test_users:
    seen = set(seqs_full.get(u, np.empty(0)).tolist())
    rec = []
    for it in pop_list:
        if it not in seen:
            rec.append(it)
        if len(rec) >= 20:
            break
    assert len(rec) == 20
    us.extend([u] * 20); its.extend(rec)
pd.DataFrame({"user_id": us, "item_id": its}).to_csv(f"{OUT}/sub_pop.csv", index=False)
print("wrote sub_pop.csv", round(time.time() - t0, 1), flush=True)
