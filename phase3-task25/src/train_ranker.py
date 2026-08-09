"""
Stage B.2 - Build "A certification pack": train the ranking model itself.

Time-based split (day < 20 = train, day >= 20 = held-out) rather than a
random split, because a random split leaks future information into
training - the single most common way offline numbers lie about online
performance (Sec 5, "offline vs online metrics").
"""
import json, os, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import get_X, FEATURES

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = f"{ROOT}/data/logs.csv"
REG_DIR = f"{ROOT}/registry"
MODEL_DIR = f"{REG_DIR}/models"
EXP_LOG = f"{ROOT}/reports/experiment_log.csv"

def load():
    df = pd.read_csv(DATA).sort_values(["candidate_id", "job_id"])
    train = df[df.day < 20].copy()
    test = df[df.day >= 20].copy()
    return train, test

def to_group_sizes(df):
    return df.groupby("query_id", sort=False).size().values

def train_and_save():
    train, test = load()
    Xtr, ytr = get_X(train), train.relevance.values
    Xte, yte = get_X(test), test.relevance.values

    train_set = lgb.Dataset(Xtr, label=ytr, group=to_group_sizes(train))
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
                  learning_rate=0.08, num_leaves=31, min_data_in_leaf=20,
                  verbose=-1, seed=25)
    t0 = time.time()
    model = lgb.train(params, train_set, num_boost_round=150)
    train_seconds = time.time() - t0

    os.makedirs(MODEL_DIR, exist_ok=True)
    version = "v2.0"
    model_path = f"{MODEL_DIR}/ranker_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    registry_entry = {
        "version": version,
        "trained_at_unix": int(time.time()),
        "train_rows": len(train), "test_rows": len(test),
        "features": FEATURES,
        "train_seconds": round(train_seconds, 2),
        "params": params,
        "data_split": "time-based day<20 train / day>=20 held-out",
        "model_path": model_path,
    }
    reg_file = f"{REG_DIR}/model_registry.json"
    registry = json.load(open(reg_file)) if os.path.exists(reg_file) else []
    registry.append(registry_entry)
    json.dump(registry, open(reg_file, "w"), indent=2)

    os.makedirs(os.path.dirname(EXP_LOG), exist_ok=True)
    header = not os.path.exists(EXP_LOG)
    with open(EXP_LOG, "a") as f:
        if header:
            f.write("stage,step,metric,value,unix_time\n")
        f.write(f"train,train_ranker,train_seconds,{train_seconds:.2f},{int(time.time())}\n")
        f.write(f"train,train_ranker,train_rows,{len(train)},{int(time.time())}\n")
        f.write(f"train,train_ranker,test_rows,{len(test)},{int(time.time())}\n")

    print(f"Trained {version} on {len(train)} rows, held out {len(test)} rows, "
          f"in {train_seconds:.2f}s -> {model_path}")
    return model, test, registry_entry

if __name__ == "__main__":
    train_and_save()
