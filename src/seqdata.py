import pandas as pd, numpy as np, time, sys, pickle
sys.path.insert(0, "/workspace/recsKaggle/src")
from common import OUT, DATA

t0 = time.time()
df = pd.read_parquet(f"{OUT}/inter.pq")
print("loaded", df.shape, round(time.time() - t0, 1), flush=True)

df = df.sort_values(["user_id", "ts"], kind="stable")
trn = df[df.is_val == 0]

g = trn.groupby("user_id")["item_id"].apply(list)
seqs = {int(u): np.asarray(v, dtype=np.int32) for u, v in g.items()}

val = df[df.is_val == 1]
vg = val.groupby("user_id")["item_id"].apply(lambda s: np.asarray(list(set(s)), dtype=np.int32))
val_tgt = {int(u): v for u, v in vg.items()}

with open(f"{OUT}/seqs.pkl", "wb") as f:
    pickle.dump({"seqs": seqs, "val_tgt": val_tgt}, f)

gf = df.groupby("user_id")["item_id"].apply(list)
seqs_full = {int(u): np.asarray(v, dtype=np.int32) for u, v in gf.items()}
with open(f"{OUT}/seqs_full.pkl", "wb") as f:
    pickle.dump({"seqs": seqs_full}, f)

print("n train seqs", len(seqs), "n val tgt", len(val_tgt), "n full seqs", len(seqs_full), flush=True)
print("done", round(time.time() - t0, 1), flush=True)
