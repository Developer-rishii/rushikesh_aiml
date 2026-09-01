#!/usr/bin/env python3
"""
fresh_env_check.py — Step 5: "test loading in a fresh environment and
predicting." Run as a SEPARATE PROCESS via subprocess.run() from
run_serialize.py — this script has no in-memory access to the `pipeline`
object created during training. It only ever touches the saved files on
disk, exactly like a real deployment would, which is what makes this a
genuine fresh-environment test rather than reusing the trained object
under a different function name.

Usage: python fresh_env_check.py <store_dir> <artifact_filename>
       <metadata_filename> <raw_data_path> <locked_features_path> <target_col>
Prints one JSON line to stdout with the result. Exits non-zero on any failure.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.serialize.store import load_artifact, predict, ArtifactLoadError


def main():
    if len(sys.argv) != 7:
        print(json.dumps({"error": "wrong number of arguments"}))
        sys.exit(2)
    store_dir, artifact_filename, metadata_filename, raw_data_path, locked_features_path, target_col = sys.argv[1:]

    try:
        pipeline, metadata, version_mismatches = load_artifact(store_dir, artifact_filename, metadata_filename)
    except ArtifactLoadError as e:
        print(json.dumps({"error": f"load failed: {e}"}))
        sys.exit(1)

    try:
        df = pd.read_csv(raw_data_path)
        features = json.loads(Path(locked_features_path).read_text())["final_feature_set"]
        for feat in features:
            if feat not in df.columns and feat.startswith("coeff_variation_"):
                m = feat.replace("coeff_variation_", "").replace("_", " ")
                mean_c, err_c = f"mean {m}", f"{m} error"
                if mean_c in df.columns and err_c in df.columns:
                    df[feat] = df[err_c] / (df[mean_c].abs() + 1e-6)
        sample = df[features].sample(n=5, random_state=123)
    except Exception as e:
        print(json.dumps({"error": f"could not build sample input: {e}"}))
        sys.exit(1)

    try:
        prediction_result = predict(pipeline, metadata, sample)
    except ArtifactLoadError as e:
        print(json.dumps({"error": f"predict failed: {e}"}))
        sys.exit(1)

    output = {
        "loaded_ok": True,
        "artifact_version": metadata["artifact_version"],
        "n_features_expected": metadata["n_features"],
        "library_version_mismatches": version_mismatches,
        "sample_prediction": prediction_result,
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
