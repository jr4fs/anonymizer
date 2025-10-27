# import argparse, yaml
# from typing import List
# import pandas as pd

# from rules import build_rules
# from io_adapters import load_frame, write_frame


# def apply_pipeline(df: pd.DataFrame, text_columns: List[str], rules) -> pd.DataFrame:
#     out = df.copy()
#     for col in text_columns:
#         if col not in out.columns:
#             continue
#         out[col] = out[col].astype(str)
#         for rule in rules:
#             out[col] = out[col].apply(rule.apply)
#     return out


# def main():
#     ap = argparse.ArgumentParser(description="Config-driven casenote anonymizer (no-LM).")
#     ap.add_argument("--config", required=True, help="Path to YAML config.")
#     args = ap.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     run = cfg["run"]
#     options = cfg.get("options", {})
#     resources = cfg.get("resources", {})
#     rules_cfg = cfg.get("rules", [])

#     df = load_frame(run["input"])
#     text_cols = run["input"]["text_columns"]

#     rules = build_rules(rules_cfg, options, resources)
#     out_df = apply_pipeline(df, text_cols, rules)

#     out_path = write_frame(out_df, run["input"], run["output"])
#     print(f"Anonymized file written to: {out_path}")


# if __name__ == "__main__":
#     main()


import argparse
import yaml
import pandas as pd
from typing import List

from rules import build_rules
from io_adapters import load_frame, write_frame

# try tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    class tqdm:  # fallback no-op
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable or []
        def __iter__(self):
            for x in self.iterable:
                yield x
        def update(self, *a, **k): pass
        def close(self): pass
        @staticmethod
        def write(msg): print(msg)


# def apply_pipeline(df: pd.DataFrame, text_columns: List[str], rules) -> pd.DataFrame:
#     """
#     For each text column:
#       - cast to str
#       - apply each rule in order
#     Shows a tqdm progress bar per column if tqdm is available.
#     """
#     out = df.copy()

#     for col in text_columns:
#         if col not in out.columns:
#             tqdm.write(f"[WARN] Column '{col}' not found in dataframe, skipping.")
#             continue

#         tqdm.write(f"[INFO] Anonymizing column '{col}' ...")

#         # ensure string dtype
#         out[col] = out[col].astype(str)

#         # We'll build the full transform row-by-row so we can show a row progress bar.
#         # For large frames this is slower than vectorized .apply(rule.apply) per rule,
#         # but it's transparent and easier to debug. If performance becomes a problem,
#         # we can flip back to the vectorized approach.
#         series = out[col]

#         # row-level loop with progress bar
#         it = series.items()
#         if TQDM_AVAILABLE:
#             it = tqdm(it, total=len(series), desc=f"  {col}", unit="row")

#         new_values = []
#         for idx, cell in it:
#             text_val = cell
#             for rule in rules:
#                 text_val = rule.apply(text_val)
#             new_values.append(text_val)

#         out[col] = new_values

#     return out

def apply_pipeline(df: pd.DataFrame, text_columns: List[str], rules) -> pd.DataFrame:
    """
    For each text column:
      - create a new column <col>_anonymized
      - leave the original column untouched
    Shows tqdm progress per column.
    """
    out = df.copy()

    for col in text_columns:
        if col not in out.columns:
            tqdm.write(f"[WARN] Column '{col}' not found in dataframe, skipping.")
            continue

        tqdm.write(f"[INFO] Anonymizing column '{col}' ...")

        # ensure string dtype
        source_series = out[col].astype(str)

        # row-level loop w/ tqdm
        it = source_series.items()
        if TQDM_AVAILABLE:
            it = tqdm(it, total=len(source_series), desc=f"  {col}", unit="row")

        new_values = []
        for idx, cell in it:
            text_val = cell
            for rule in rules:
                text_val = rule.apply(text_val)
            new_values.append(text_val)

        # write anonymized text to a *new* column
        out[f"{col}_anonymized"] = new_values

    return out

def main():
    ap = argparse.ArgumentParser(description="Config-driven casenote anonymizer (no-LM).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    # load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run = cfg["run"]
    options = cfg.get("options", {})
    resources = cfg.get("resources", {})
    rules_cfg = cfg.get("rules", [])

    # load data
    df = load_frame(run["input"])
    text_cols = run["input"]["text_columns"]

    # build rules (note: rules are applied in the order they appear in config.yaml)
    rules = build_rules(rules_cfg, options, resources)

    # apply anonymization pipeline with progress bars
    out_df = apply_pipeline(df, text_cols, rules)

    # write output
    out_path = write_frame(out_df, run["input"], run["output"])
    print(f"\n[INFO] Anonymized file written to: {out_path}")


if __name__ == "__main__":
    main()
