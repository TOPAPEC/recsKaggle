import pandas as pd, numpy as np, json, sys

DATA = "/workspace/recsKaggle/data"
tr = pd.read_parquet(f"{DATA}/train.pq")
tu = pd.read_csv(f"{DATA}/test_users.csv")
it = pd.read_parquet(f"{DATA}/items.pq")

R = {}
R["train_rows"] = int(len(tr))
R["n_users_train"] = int(tr.user_id.nunique())
R["n_items_train"] = int(tr.item_id.nunique())
R["n_items_catalog"] = int(it.item_id.nunique())
R["n_test_users"] = int(tu.user_id.nunique())

tr["ts"] = pd.to_datetime(tr["timestamp"])
R["ts_min"] = str(tr.ts.min())
R["ts_max"] = str(tr.ts.max())
span = (tr.ts.max() - tr.ts.min())
R["span_days"] = span.days

test_set = set(tu.user_id.tolist())
train_users = set(tr.user_id.unique().tolist())
R["test_users_in_train"] = int(len(test_set & train_users))
R["test_users_cold(no train history)"] = int(len(test_set - train_users))

g = tr.groupby("user_id").size()
R["interactions_per_user"] = {k: float(v) for k, v in g.describe().items()}
gt = tr[tr.user_id.isin(test_set)].groupby("user_id").size()
R["interactions_per_TEST_user"] = {k: float(v) for k, v in gt.describe().items()}

R["is_purchased_mean"] = float(tr.is_purchased.mean())
R["rating_value_counts"] = {int(k): int(v) for k, v in tr.rating.value_counts().items()}
R["rating_mean_nonzero"] = float(tr.loc[tr.rating > 0, "rating"].mean())

ip = tr.item_id.value_counts()
R["item_pop_top10"] = {int(k): int(v) for k, v in ip.head(10).items()}
R["item_pop_describe"] = {k: float(v) for k, v in ip.describe().items()}
top20share = ip.head(20).sum() / len(tr)
R["top20_items_interaction_share"] = float(top20share)
R["top100_items_interaction_share"] = float(ip.head(100).sum() / len(tr))

dup = tr.groupby(["user_id", "item_id"]).size()
R["frac_user_item_pairs_repeated(>1)"] = float((dup > 1).mean())
R["max_repeats_same_pair"] = int(dup.max())

R["ts_monthly_counts"] = {str(k): int(v) for k, v in tr.ts.dt.to_period("M").value_counts().sort_index().items()}

print(json.dumps(R, indent=2, default=str))
