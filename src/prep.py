import pandas as pd, numpy as np, time, os

DATA = "/workspace/recsKaggle/data"
OUT = "/workspace/recsKaggle/out"
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
tr = pd.read_parquet(f"{DATA}/train.pq", columns=["user_id", "item_id", "is_purchased", "rating", "timestamp"])
print("loaded", tr.shape, time.time() - t0)

tr["ts"] = tr["timestamp"].astype("int64") // 10**6  # us -> seconds
tr = tr.drop(columns=["timestamp"])
# stable sort by user then time
tr = tr.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)

# global time threshold for validation: last VAL_DAYS as holdout
ts_max = tr.ts.max()
VAL_DAYS = 14
cutoff = ts_max - VAL_DAYS * 86400
print("ts_max", ts_max, "cutoff", cutoff)
tr["is_val"] = (tr.ts > cutoff).astype(np.int8)
print("val interactions:", int(tr.is_val.sum()), "frac", float(tr.is_val.mean()))
print("val users (with >=1 val interaction):", tr.loc[tr.is_val == 1, "user_id"].nunique())

# downcast
tr["user_id"] = tr.user_id.astype(np.int32)
tr["item_id"] = tr.item_id.astype(np.int32)
tr["rating"] = tr.rating.astype(np.int8)
tr["is_purchased"] = tr.is_purchased.astype(np.int8)
tr["ts"] = tr.ts.astype(np.int64)

tr.to_parquet(f"{OUT}/inter.pq", index=False)
print("saved inter.pq", tr.shape, time.time() - t0)
print(tr.head())
print(tr.dtypes)
