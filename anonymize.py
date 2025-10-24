import argparse, yaml
from typing import List
import pandas as pd

from rules import build_rules
from io_adapters import load_frame, write_frame


def apply_pipeline(df: pd.DataFrame, text_columns: List[str], rules) -> pd.DataFrame:
    out = df.copy()
    for col in text_columns:
        if col not in out.columns:
            continue
        out[col] = out[col].astype(str)
        for rule in rules:
            out[col] = out[col].apply(rule.apply)
    return out


def main():
    ap = argparse.ArgumentParser(description="Config-driven casenote anonymizer (no-LM).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run = cfg["run"]
    options = cfg.get("options", {})
    resources = cfg.get("resources", {})
    rules_cfg = cfg.get("rules", [])

    df = load_frame(run["input"])
    text_cols = run["input"]["text_columns"]

    rules = build_rules(rules_cfg, options, resources)
    out_df = apply_pipeline(df, text_cols, rules)

    out_path = write_frame(out_df, run["input"], run["output"])
    print(f"Anonymized file written to: {out_path}")


if __name__ == "__main__":
    main()
