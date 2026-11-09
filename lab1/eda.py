"""Reproducible AG News EDA (variant 39, text modality P3)."""
import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lab1.config import Lab1Config, read_config
from lab1.hashes import sha256, verify_manifest

CLASSES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
TOKEN_RE = re.compile(r"\b[a-z][a-z'-]*\b", re.I)
STOPWORDS = set("""the a an and or of to in on at by from with as for is are was were be been
that this it its their his her they he she i we you not but have has had will would can
could should do does did about after before over under into out up down than more less
new says said say news reuters ap afp us u s uk one two three year years day days week
weeks month months today yesterday thursday friday monday saturday sunday wednesday
monday tuesday first last next now which who what when where how all so if also only
other some many much most those these per off through during while amid since such
between each may might much across around get got like company companies report reports
""".split())


def tokenize(text: str, min_length: int) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(text) if len(term) >= min_length]


def load_split(path: Path, expected: int, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    if list(frame.columns) != ["text", "label"]:
        raise ValueError(f"{split}: require fields exactly text,label; got {list(frame.columns)}")
    if len(frame) != expected:
        raise ValueError(f"{split}: expected {expected} rows; got {len(frame)}")
    if frame["label"].isna().any() or not set(frame["label"].unique()).issubset(CLASSES):
        raise ValueError(f"{split}: invalid class label; expected 0,1,2,3")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = frame["label"].astype(int)
    return frame


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "not_initialized"


def push(rows: list[dict], group: str, metric: str, value: float | int,
         unit: str = "count", split: str = "train", run_id: str = "") -> None:
    rows.append({"split": split, "group": group, "metric": metric,
                 "value": round(float(value), 6), "unit": unit, "run_id": run_id})


def bar_chart(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> None:
    train_counts = train["label"].value_counts().reindex(CLASSES.keys(), fill_value=0)
    test_counts = test["label"].value_counts().reindex(CLASSES.keys(), fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for ax, counts, title in zip(axes, [train_counts, test_counts], ["TRAIN", "TEST"]):
        sns.barplot(x=list(CLASSES.values()), y=counts.to_numpy(), ax=ax,
                    color="#3b82a5", errorbar=None)
        for i, count in enumerate(counts):
            ax.text(i, count, f"{count:,}\n({count / counts.sum() * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} · n={counts.sum():,}")
        ax.set_ylabel("Number of documents")
        ax.set_xlabel("Topic")
        ax.set_ylim(0, max(counts) * 1.18)
    fig.suptitle("AG News — class balance across official splits", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=175, bbox_inches="tight")
    plt.close(fig)


def text_lengths(train: pd.DataFrame, out: Path, max_tokens: int) -> None:
    data = train.loc[:, ["label", "word_count"]].copy()
    data["label"] = data["label"].map(CLASSES)
    capped = data["word_count"].clip(upper=max_tokens)
    data["word_count_capped"] = capped
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=data, x="label", y="word_count_capped", ax=ax,
                color="#7c9cbd", showfliers=False)
    medians = train.groupby("label")["word_count"].median()
    for label, median in medians.items():
        ax.text(label, min(median + 8, max_tokens - 3), f"median {median:.0f}",
                ha="center", fontsize=9)
    ax.set_title("Text length by topic (word counts)")
    ax.set_xlabel("Topic")
    ax.set_ylabel(f"Tokens (values clipped at {max_tokens} for plotting)")
    fig.tight_layout()
    fig.savefig(out, dpi=175)
    plt.close(fig)


def topic_terms_plot(top_terms: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for label, ax in zip(CLASSES, axes.flat):
        relevant = [v for v in top_terms if v["class_id"] == label][:10]
        if relevant:
            sns.barplot(x=[v["frequency"] for v in relevant],
                        y=[v["token"] for v in relevant],
                        ax=ax, color="#478fa7", errorbar=None)
            for i, val in enumerate(relevant):
                ax.text(val["frequency"], i, f" {val['frequency']:,}", va="center", fontsize=8)
        if relevant:
            ax.set_xlim(0, max(v["frequency"] for v in relevant) * 1.35)
        ax.set_title(CLASSES[label])
        ax.set_xlabel("Occurrences in training documents")
        ax.set_ylabel("")
    fig.suptitle("Most frequent non-stopword terms by topic", fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.12, right=0.98, hspace=0.47, wspace=0.38)
    fig.savefig(out, dpi=165)
    plt.close(fig)


def frequency_buckets_plot(buckets: dict, out: Path) -> None:
    labels = list(buckets)
    values = list(buckets.values())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=labels, y=values, ax=ax, color="#468c92", errorbar=None)
    for i, n in enumerate(values):
        ax.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Vocabulary: term-frequency buckets (TRAIN only)")
    ax.set_xlabel("Total occurrences per token")
    ax.set_ylabel("Distinct token count")
    ax.set_ylim(0, max(values) * 1.14 if any(values) else 1)
    fig.tight_layout()
    fig.savefig(out, dpi=175)
    plt.close(fig)


def overlap_plot(class_counters: dict, out: Path) -> dict:
    # Lexical overlap is a proxy for ambiguity, not true model confusion.
    vocab = {label: {term for term, count in counts.most_common(100)}
             for label, counts in class_counters.items()}
    matrix = np.zeros((4, 4))
    for i in CLASSES:
        for j in CLASSES:
            united = vocab[i] | vocab[j]
            matrix[i, j] = len(vocab[i] & vocab[j]) / len(united) if united else 0
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1,
                xticklabels=list(CLASSES.values()), yticklabels=list(CLASSES.values()),
                cmap="Blues", ax=ax, cbar_kws={"label": "Jaccard similarity"})
    ax.set_title("Topic overlap: 100 most frequent content words")
    fig.tight_layout()
    fig.savefig(out, dpi=175)
    plt.close(fig)
    pair = max(combinations(CLASSES, 2), key=lambda ab: matrix[ab[0], ab[1]])
    return {"pair": f"{CLASSES[pair[0]]} / {CLASSES[pair[1]]}",
            "jaccard": float(matrix[pair[0], pair[1]])}


def write_csv(path: Path, rows: list[dict], names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def run_eda(config: Lab1Config, *, validate_manifest: bool = True,
            mlflow_enabled: bool | None = None) -> dict:
    out = config.report_dir
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "hash_manifest.json"
    if validate_manifest:
        verified, details = verify_manifest(manifest_path)
        if not verified:
            raise ValueError("Raw files have failed integrity check:\n" + "\n".join(details))

    train = load_split(config.input_dir / "train.csv", config.expected_train, "train")
    test = load_split(config.input_dir / "test.csv", config.expected_test, "test")
    train["word_count"] = train["text"].apply(lambda s: len(TOKEN_RE.findall(s)))
    train["char_count"] = train["text"].str.len()
    test["word_count"] = test["text"].apply(lambda s: len(TOKEN_RE.findall(s)))
    test["char_count"] = test["text"].str.len()
    train_dup = int(train.duplicated(subset=["text"]).sum())
    test_dup = int(test.duplicated(subset=["text"]).sum())
    overlap = int(test["text"].isin(set(train["text"])).sum())
    empty_train = int(train["text"].str.strip().eq("").sum())
    empty_test = int(test["text"].str.strip().eq("").sum())

    all_tokens = Counter()
    class_counters = {k: Counter() for k in CLASSES}
    for label, text in train[["label", "text"]].itertuples(index=False, name=None):
        tokens = tokenize(text, config.min_token_length)
        all_tokens.update(tokens)
        class_counters[label].update(t for t in tokens if t not in STOPWORDS)
    filtered = Counter({word: count for word, count in all_tokens.items()
                        if word not in STOPWORDS})
    top_terms = []
    for label, counter in class_counters.items():
        for position, (term, count) in enumerate(counter.most_common(config.max_top_terms), 1):
            top_terms.append({"class_id": label, "topic": CLASSES[label],
                              "rank": position, "token": term, "frequency": count})
    bins = {
        "1": sum(1 for c in all_tokens.values() if c == 1),
        "2": sum(1 for c in all_tokens.values() if c == 2),
        "3–5": sum(1 for c in all_tokens.values() if 3 <= c <= 5),
        "6–10": sum(1 for c in all_tokens.values() if 6 <= c <= 10),
        "11–50": sum(1 for c in all_tokens.values() if 11 <= c <= 50),
        "51–100": sum(1 for c in all_tokens.values() if 51 <= c <= 100),
        ">100": sum(1 for c in all_tokens.values() if c > 100),
    }
    metrics = []
    for split, data, duplicates, empties in [("train", train, train_dup, empty_train),
                                             ("test", test, test_dup, empty_test)]:
        push(metrics, "all", "records", len(data), split=split)
        push(metrics, "all", "null_texts", empties, split=split)
        push(metrics, "all", "duplicate_texts", duplicates, split=split)
        for statistic in ["mean", "median", "max"]:
            push(metrics, "all", "word_count_" + statistic,
                 getattr(data["word_count"], statistic)(), "tokens", split)
            push(metrics, "all", "char_count_" + statistic,
                 getattr(data["char_count"], statistic)(), "characters", split)
        push(metrics, "all", "word_count_p90", data["word_count"].quantile(.90), "tokens", split)
        for cls, name in CLASSES.items():
            part = data[data["label"] == cls]
            push(metrics, name, "records", len(part), split=split)
            push(metrics, name, "class_share_pct", len(part) / len(data) * 100,
                 "percent", split)
            push(metrics, name, "word_count_median", part["word_count"].median(),
                 "tokens", split)
    push(metrics, "all", "train_test_overlapping_texts", overlap, split="both")
    push(metrics, "all", "vocabulary_size", len(all_tokens), split="train")
    push(metrics, "all", "content_vocabulary_size", len(filtered), split="train")
    push(metrics, "all", "total_tokens", sum(all_tokens.values()), split="train")
    push(metrics, "all", "hapax_legomena", bins["1"], split="train")
    push(metrics, "all", "hapax_share_pct", bins["1"] / max(len(all_tokens), 1) * 100,
         "percent", "train")
    for name, count in bins.items():
        push(metrics, "frequency_bucket", name, count)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    bar_chart(train, test, fig_dir / "class_balance.png")
    text_lengths(train, fig_dir / "length_by_topic.png", config.plot_max_tokens)
    topic_terms_plot(top_terms, fig_dir / "top_words_by_topic.png")
    frequency_buckets_plot(bins, fig_dir / "vocabulary_frequency_buckets.png")
    high_overlap = overlap_plot(class_counters, fig_dir / "topic_word_overlap.png")
    push(metrics, high_overlap["pair"], "max_topic_jaccard", high_overlap["jaccard"],
         "ratio")

    write_csv(out / "eda_stats.csv", metrics,
              ["split", "group", "metric", "value", "unit", "run_id"])
    write_csv(out / "eda_terms.csv", top_terms,
              ["class_id", "topic", "rank", "token", "frequency"])
    write_csv(out / "length_buckets.csv", [
        {"class_id": i, "topic": CLASSES[i], "0-20": int(((train.label == i) & (train.word_count <= 20)).sum()),
         "21-40": int(((train.label == i) & train.word_count.between(21, 40)).sum()),
         "41-80": int(((train.label == i) & train.word_count.between(41, 80)).sum()),
         "81-160": int(((train.label == i) & train.word_count.between(81, 160)).sum()),
         ">160": int(((train.label == i) & (train.word_count > 160)).sum())}
        for i in CLASSES], ["class_id", "topic", "0-20", "21-40", "41-80", "81-160", ">160"])

    run_id = "not_logged"
    if mlflow_enabled is None:
        mlflow_enabled = config.use_mlflow
    if mlflow_enabled:
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError("Install mlflow or set use_mlflow: false") from error
        mlflow.set_tracking_uri("file:" + str(Path("mlruns").resolve()))
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name="eda_ag_news_v39") as current:
            run_id = current.info.run_id
            mlflow.log_params({"variant": config.variant, "seed": config.seed,
                               "source": config.dataset_id,
                               "min_token_length": config.min_token_length})
            mlflow.log_metrics({"n_train": len(train), "n_test": len(test),
                                "vocab_size": len(all_tokens),
                                "empty_train": empty_train,
                                "duplicate_train": train_dup,
                                "train_test_text_overlap": overlap})
            mlflow.log_artifacts(str(out / "figures"), artifact_path="figures")
            mlflow.log_artifact(str(out / "eda_stats.csv"))
            mlflow.log_artifact(str(manifest_path))
    for row in metrics:
        row["run_id"] = run_id
    write_csv(out / "eda_stats.csv", metrics,
              ["split", "group", "metric", "value", "unit", "run_id"])
    stats_hash = sha256(out / "eda_stats.csv")
    def metric(split: str, group: str, name: str) -> float:
        return next(r["value"] for r in metrics if r["split"] == split
                    and r["group"] == group and r["metric"] == name)
    return {
        "n_train": len(train), "n_test": len(test),
        "train_count_per_class": {name: int((train.label == label).sum())
                                  for label, name in CLASSES.items()},
        "test_count_per_class": {name: int((test.label == label).sum())
                                 for label, name in CLASSES.items()},
        "empty_train": empty_train, "empty_test": empty_test,
        "train_duplicates": train_dup, "test_duplicates": test_dup,
        "train_test_overlap": overlap, "vocab_size": len(all_tokens),
        "hapax_share_pct": metric("train", "all", "hapax_share_pct"),
        "median_words": metric("train", "all", "word_count_median"),
        "p90_words": metric("train", "all", "word_count_p90"),
        "max_overlap_topics": high_overlap,
        "eda_csv_sha256": stats_hash,
        "git_commit": git_revision(), "run_id": run_id,
    }


def write_report(config: Lab1Config, result: dict) -> None:
    from lab1.report import write_report as generate_report
    generate_report(config, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lab1.yaml")
    parser.add_argument("--skip-hash-verification", action="store_true",
                        help="ONLY for local automated tests; NEVER for final EDA")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="ONLY for offline smoke checks")
    args = parser.parse_args()
    config = read_config(args.config)
    result = run_eda(config, validate_manifest=not args.skip_hash_verification,
                     mlflow_enabled=False if args.no_mlflow else None)
    write_report(config, result)
    output = config.report_dir / "eda_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
