"""
Evaluate the agent's fraud-flagging accuracy against a real, labeled dataset
(Kaggle's Credit Card Fraud Detection dataset).

This is separate from the main app pipeline - it doesn't touch Plaid or your
SQLite database. It's purely a test: feed the agent transactions with known
right answers (fraud or not), and measure how well its "flag" decisions
line up with reality.

Usage:
    1. Download creditcard.csv from Kaggle's "Credit Card Fraud Detection"
       dataset and put it in this same folder.
    2. Run: python3 eval_fraud_detection.py
"""
import random
import time
from pathlib import Path

import pandas as pd

import ai_agent

CSV_PATH = Path(__file__).resolve().parent / "creditcard.csv"
BATCH_SIZE = 20          # transactions per API call
NUM_FRAUD_SAMPLES = 50   # how many known-fraud rows to test
NUM_LEGIT_SAMPLES = 150  # how many known-legit rows to test
RANDOM_SEED = 42


def load_sample() -> pd.DataFrame:
    """
    Build a balanced-ish test sample. The full dataset is extremely
    imbalanced (fraud is ~0.17% of rows), so testing on a random sample of
    the whole thing would barely include any fraud cases at all. Instead we
    deliberately sample a mix so precision/recall are actually measurable.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {CSV_PATH}. Download creditcard.csv from Kaggle's "
            "'Credit Card Fraud Detection' dataset and place it in this folder."
        )

    df = pd.read_csv(CSV_PATH)

    fraud = df[df["Class"] == 1].sample(
        n=min(NUM_FRAUD_SAMPLES, (df["Class"] == 1).sum()), random_state=RANDOM_SEED
    )
    legit = df[df["Class"] == 0].sample(n=NUM_LEGIT_SAMPLES, random_state=RANDOM_SEED)

    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_SEED)  # shuffle
    return sample.reset_index(drop=True)


def to_agent_format(sample: pd.DataFrame) -> list[dict]:
    """
    Convert dataset rows into the same shape the agent already expects.
    Note: this dataset has no real merchant names or categories (the
    features are anonymized), so the agent is only working from the
    transaction amount here - that's a real limitation of this dataset,
    not of the agent itself.
    """
    transactions = []
    for i, row in sample.iterrows():
        transactions.append({
            "id": int(i),
            "date": "unknown",
            "merchant_raw": "Unknown Merchant",
            "amount": float(row["Amount"]),
            "plaid_category": None,
        })
    return transactions


def run_eval(sample: pd.DataFrame) -> pd.DataFrame:
    """Send the sample through the agent in batches and collect predictions."""
    transactions = to_agent_format(sample)
    all_results = []

    for start in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[start:start + BATCH_SIZE]
        print(f"Sending batch {start // BATCH_SIZE + 1} "
              f"({len(batch)} transactions)...")
        results = ai_agent.categorize_transactions(batch)
        all_results.extend(results)
        time.sleep(2)  # be polite to the free-tier rate limit

    predictions = {r["id"]: bool(r.get("flag")) for r in all_results}

    sample = sample.copy()
    sample["predicted_flag"] = [predictions.get(i, False) for i in sample.index]
    sample["actual_fraud"] = sample["Class"] == 1
    return sample


def compute_metrics(results: pd.DataFrame) -> dict:
    """Precision, recall, F1, and accuracy from a confusion matrix."""
    tp = int(((results["predicted_flag"]) & (results["actual_fraud"])).sum())
    fp = int(((results["predicted_flag"]) & (~results["actual_fraud"])).sum())
    fn = int(((~results["predicted_flag"]) & (results["actual_fraud"])).sum())
    tn = int(((~results["predicted_flag"]) & (~results["actual_fraud"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def main() -> None:
    random.seed(RANDOM_SEED)

    print("Loading test sample...")
    sample = load_sample()
    print(f"Testing on {len(sample)} transactions "
          f"({(sample['Class'] == 1).sum()} fraud, {(sample['Class'] == 0).sum()} legit)\n")

    results = run_eval(sample)
    metrics = compute_metrics(results)

    print("\n--- Results ---")
    print(f"True Positives (fraud correctly flagged):     {metrics['true_positives']}")
    print(f"False Positives (legit incorrectly flagged):  {metrics['false_positives']}")
    print(f"False Negatives (fraud missed):                {metrics['false_negatives']}")
    print(f"True Negatives (legit correctly not flagged):  {metrics['true_negatives']}")
    print()
    print(f"Precision: {metrics['precision']:.2%}  (of everything flagged, how much was real fraud)")
    print(f"Recall:    {metrics['recall']:.2%}  (of all real fraud, how much did the agent catch)")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")

    results.to_csv(Path(__file__).resolve().parent / "eval_results.csv", index=False)
    print("\nFull results saved to eval_results.csv")


if __name__ == "__main__":
    main()
