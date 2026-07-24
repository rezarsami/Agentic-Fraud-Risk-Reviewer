"""
Evaluate the agent's fraud-flagging accuracy against a labeled synthetic
dataset (Kaggle's "Synthetic Financial Fraud Dataset") - a genuinely
different schema than the bank_transactions_data.csv project, so this is a
separate, standalone eval rather than reusing ai_agent.assess_fraud_risk(),
which is built around the bank dataset's behavioral fields (device/location
changes, account balance, login attempts) that don't exist here.

This dataset instead gives: amount, transaction_type, merchant_category,
country, hour, and two pre-computed risk scores (device_risk_score,
ip_risk_score). We reuse ai_agent's low-level model-calling machinery
(_call_model, batching, rate-limit retries) for consistency, but write a
schema-appropriate prompt and validation here.

IMPORTANT HONESTY NOTE (found by actually inspecting the data before
building this): device_risk_score and ip_risk_score each correlate with
is_fraud at ~0.87, and amount correlates at ~0.64 - these are extremely
strong, almost deterministic signals for synthetic data. Expect very high
precision/recall here. That reflects this dataset being easy, not proof the
agent would perform this well on messier, real-world signals.

Usage:
    python3 eval_synthetic_fraud.py
"""
import time
from pathlib import Path

import pandas as pd

import ai_agent

CSV_PATH = Path(__file__).resolve().parent / "synthetic_fraud_dataset.csv"
BATCH_SIZE = 10
NUM_FRAUD_SAMPLES = 200   # out of 500 total fraud rows available
NUM_LEGIT_SAMPLES = 400   # out of 9,500 total legit rows available
RANDOM_SEED = 42


def load_sample() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {CSV_PATH}. Place synthetic_fraud_dataset.csv in this folder."
        )

    df = pd.read_csv(CSV_PATH)

    fraud = df[df["is_fraud"] == 1].sample(
        n=min(NUM_FRAUD_SAMPLES, (df["is_fraud"] == 1).sum()), random_state=RANDOM_SEED
    )
    legit = df[df["is_fraud"] == 0].sample(n=NUM_LEGIT_SAMPLES, random_state=RANDOM_SEED)

    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_SEED)  # shuffle
    return sample.reset_index(drop=True)


def engineer_features(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Add a per-user amount z-score, same shrinkage-estimation idea used in
    the main project (blend each user's own mean/std with the population's,
    weighted by how many transactions that user has), since this dataset
    averages 10 transactions/user - enough for it to be meaningful.
    """
    sample = sample.copy()
    pop_mean = sample["amount"].mean()
    pop_std = sample["amount"].std()

    user_stats = sample.groupby("user_id")["amount"].agg(["mean", "std", "count"])
    user_stats["std"] = user_stats["std"].fillna(pop_std)

    SHRINKAGE_STRENGTH = 5

    def compute_zscore(row):
        stats = user_stats.loc[row["user_id"]]
        weight = stats["count"] / (stats["count"] + SHRINKAGE_STRENGTH)
        blended_mean = weight * stats["mean"] + (1 - weight) * pop_mean
        blended_std = weight * stats["std"] + (1 - weight) * pop_std
        if blended_std == 0:
            return 0.0
        return (row["amount"] - blended_mean) / blended_std

    sample["amount_zscore"] = sample.apply(compute_zscore, axis=1)
    return sample


def to_agent_format(sample: pd.DataFrame) -> list[dict]:
    transactions = []
    for _, row in sample.iterrows():
        transactions.append({
            "id": str(row["transaction_id"]),
            "amount": float(row["amount"]),
            "amount_zscore": round(float(row["amount_zscore"]), 2),
            "transaction_type": row["transaction_type"],
            "merchant_category": row["merchant_category"],
            "country": row["country"],
            "hour": int(row["hour"]),
            "device_risk_score": round(float(row["device_risk_score"]), 3),
            "ip_risk_score": round(float(row["ip_risk_score"]), 3),
        })
    return transactions


def build_prompt(batch: list[dict]) -> str:
    import json
    payload = json.dumps(batch, indent=2)
    return f"""
You are a fraud risk analyst reviewing financial transactions. Each
transaction includes:

- amount: the transaction amount
- amount_zscore: how unusual this amount is compared to this specific
  user's own typical transaction size (0 = typical, further from 0 = more
  unusual)
- transaction_type: ATM, Online, POS, or QR
- merchant_category: what kind of merchant this was
- country: where the transaction occurred
- hour: hour of day (0-23) the transaction occurred
- device_risk_score: a pre-computed reputation score for the device used
  (0 = low risk, 1 = high risk)
- ip_risk_score: a pre-computed reputation score for the IP address used
  (0 = low risk, 1 = high risk)

For each transaction, decide whether it looks risky enough to flag for
human review, weighing all signals together - high device_risk_score AND
high ip_risk_score together is a much stronger signal than either alone,
and an unusual amount_zscore combined with high risk scores is stronger
still. Unusual hours (very late night/early morning) can also be worth
factoring in.

CRITICAL - stay grounded in the actual numbers: only cite a signal as a
reason if its value actually supports that claim. Don't call a device or
IP "risky" unless its score is actually above roughly 0.5, and don't call
an amount "unusually large" unless amount_zscore is meaningfully far from 0
(roughly beyond +/-1.5). Never state a reason that isn't backed by the
specific values you were given for that transaction.

Return a JSON array with one object per transaction:
- id: the transaction id from the input
- flag: true or false
- flag_reason: a short plain-English reason if flag is true, otherwise ""

Return a JSON array only - no extra commentary, no markdown fences.

Transactions:
{payload}
"""


def _validate_result_item(item: dict, valid_ids: set) -> bool:
    """
    Same validation logic as ai_agent._validate_result_item - check the
    right fields are present, correctly typed, and match a real
    transaction we sent, so we never trust malformed or hallucinated data.
    """
    if not isinstance(item, dict):
        return False
    if not {"id", "flag", "flag_reason"}.issubset(item.keys()):
        return False
    if item.get("id") not in valid_ids:
        return False
    if not isinstance(item.get("flag"), bool):
        return False
    if not isinstance(item.get("flag_reason"), str):
        return False
    if item["flag"] and not item["flag_reason"].strip():
        return False
    return True


def _check_grounding(item: dict, tx: dict) -> None:
    """
    Same grounding-check idea as ai_agent._check_grounding, adapted to this
    dataset's fields: if the model claims high device/IP risk as the reason
    but the actual risk scores don't support that, print a warning.
    """
    if not item.get("flag"):
        return
    reason = (item.get("flag_reason") or "").lower()

    claims_device_risk = any(w in reason for w in ["device risk", "risky device", "suspicious device"])
    if claims_device_risk and tx.get("device_risk_score", 0) < 0.5:
        print(f"GROUNDING WARNING: transaction {item.get('id')} cites device risk, but "
              f"device_risk_score is only {tx.get('device_risk_score'):.3f} (not actually high). "
              f"Reason given: '{item.get('flag_reason')}'")

    claims_ip_risk = any(w in reason for w in ["ip risk", "risky ip", "suspicious ip"])
    if claims_ip_risk and tx.get("ip_risk_score", 0) < 0.5:
        print(f"GROUNDING WARNING: transaction {item.get('id')} cites IP risk, but "
              f"ip_risk_score is only {tx.get('ip_risk_score'):.3f} (not actually high). "
              f"Reason given: '{item.get('flag_reason')}'")

    claims_unusual_amount = any(w in reason for w in ["large amount", "unusual amount", "unusually large"])
    if claims_unusual_amount and abs(tx.get("amount_zscore", 0)) < 1.0:
        print(f"GROUNDING WARNING: transaction {item.get('id')} claims an unusual amount, but "
              f"amount_zscore is only {tx.get('amount_zscore'):.2f} (not actually unusual). "
              f"Reason given: '{item.get('flag_reason')}'")


def _validate_and_clean(results: list[dict], batch: list[dict]) -> list[dict]:
    """Same validate-then-fallback pattern as ai_agent._validate_and_clean."""
    valid_ids = {t["id"] for t in batch}
    by_id = {t["id"]: t for t in batch}

    seen_ids = set()
    cleaned = []
    rejected_count = 0

    for item in results:
        if _validate_result_item(item, valid_ids):
            cleaned.append(item)
            seen_ids.add(item["id"])
            _check_grounding(item, by_id[item["id"]])
        else:
            rejected_count += 1

    missing_ids = valid_ids - seen_ids
    for missing_id in missing_ids:
        cleaned.append({"id": missing_id, "flag": False, "flag_reason": ""})

    if rejected_count or missing_ids:
        print(f"Validation: rejected {rejected_count} malformed result(s), "
              f"filled in {len(missing_ids)} missing transaction(s) with fallback.")

    return cleaned


def run_eval(sample: pd.DataFrame) -> pd.DataFrame:
    import json
    transactions = to_agent_format(sample)
    all_results = []

    total_batches = (len(transactions) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Assessing batch {batch_num}/{total_batches} ({len(batch)} transactions)...")

        prompt = build_prompt(batch)
        try:
            raw = ai_agent._call_model(prompt, max_tokens=1000).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            results = _validate_and_clean(parsed, batch)
        except Exception as exc:
            print(f"Batch failed ({exc}); treating this batch as unflagged.")
            results = [{"id": t["id"], "flag": False, "flag_reason": ""} for t in batch]

        all_results.extend(results)
        if batch_num < total_batches:
            time.sleep(15)

    predictions = {r["id"]: bool(r.get("flag")) for r in all_results}
    reasons = {r["id"]: r.get("flag_reason", "") for r in all_results}

    sample = sample.copy()
    sample["transaction_id"] = sample["transaction_id"].astype(str)
    sample["predicted_flag"] = sample["transaction_id"].map(predictions).fillna(False)
    sample["predicted_reason"] = sample["transaction_id"].map(reasons).fillna("")
    sample["actual_fraud"] = sample["is_fraud"] == 1
    return sample


def compute_metrics(results: pd.DataFrame) -> dict:
    tp = int(((results["predicted_flag"]) & (results["actual_fraud"])).sum())
    fp = int(((results["predicted_flag"]) & (~results["actual_fraud"])).sum())
    fn = int(((~results["predicted_flag"]) & (results["actual_fraud"])).sum())
    tn = int(((~results["predicted_flag"]) & (~results["actual_fraud"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0

    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": precision, "recall": recall,
        "f1_score": f1, "accuracy": accuracy,
    }


def main() -> None:
    print("Loading test sample...")
    sample = load_sample()
    print(f"Testing on {len(sample)} transactions "
          f"({(sample['is_fraud'] == 1).sum()} fraud, {(sample['is_fraud'] == 0).sum()} legit)\n")

    print("Engineering per-user amount z-scores...")
    sample = engineer_features(sample)

    results = run_eval(sample)
    metrics = compute_metrics(results)

    print("\n--- Results ---")
    print(f"True Positives (fraud correctly flagged):     {metrics['true_positives']}")
    print(f"False Positives (legit incorrectly flagged):  {metrics['false_positives']}")
    print(f"False Negatives (fraud missed):                {metrics['false_negatives']}")
    print(f"True Negatives (legit correctly not flagged):  {metrics['true_negatives']}")
    print()
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 Score:  {metrics['f1_score']:.2%}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print()
    print("NOTE: device_risk_score and ip_risk_score correlate with is_fraud at ~0.87 in this")
    print("dataset - an unusually strong, near-deterministic signal for synthetic data. High")
    print("scores here reflect this dataset being easy, not proof of real-world performance.")

    results.to_csv(Path(__file__).resolve().parent / "synthetic_eval_results.csv", index=False)
    print("\nFull results saved to synthetic_eval_results.csv")


if __name__ == "__main__":
    main()
