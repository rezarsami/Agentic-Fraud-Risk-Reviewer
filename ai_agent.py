"""
This is the ONLY place in the project that talks to an AI model, and the
ONLY place that decides whether a transaction is flagged as risky.

data_loader.py computes behavioral evidence (z-scores, device/location/IP
changes, etc.) but makes no flag/no-flag decision itself - it hands that
evidence to assess_fraud_risk() here, which is where the actual judgment
call happens, with a plain-English explanation attached.

Deliberately excluded from the risk-scoring prompt: CustomerAge and
CustomerOccupation. Using demographic traits to help decide who looks
"suspicious" is a well-documented source of bias in real fraud systems -
this project only reasons over behavioral signals (amount patterns, device/
location/IP changes, login attempts, timing), not who the customer is.

Uses Groq's free-tier API (console.groq.com).
"""
import json
import os
import time
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.1-8b-instant has a MUCH higher free-tier daily limit (14,400
# requests/day) than llama-3.3-70b-versatile (only 1,000/day) - important
# for a project like this that can send 100+ requests per full run. Switch
# back to the 70b model only for a final, limited demo run if you want the
# stronger reasoning quality, and mind the 1,000/day cap if you do.
MODEL = "llama-3.1-8b-instant"

# How many transactions to send per API call. Sending everything in one call
# (e.g. all 2,512 rows at once) produces a response too long to fit in a
# reasonable token budget, which gets cut off mid-JSON and fails to parse -
# silently falling back to "no flags" for the ENTIRE batch. Small batches
# avoid this AND keep each call's token usage well under the free tier's
# tokens-per-minute limit (llama-3.1-8b-instant: 6,000 TPM) - a batch of 25
# with a large max_tokens budget can use enough tokens that even 2 calls in
# the same minute exceed that cap, regardless of the (much higher) daily
# request limit.
BATCH_SIZE = 10

# The agent now outputs a continuous risk_score (0-100) instead of a raw
# true/false flag. This threshold is what turns that score into a flag for
# display/storage purposes - and unlike a model-decided boolean, it can be
# adjusted (e.g. in the dashboard) without re-running the agent, since the
# underlying score is preserved. 60 is a reasonable starting point; a lower
# threshold catches more real fraud at the cost of more false alarms
# (higher recall, lower precision), and vice versa for a higher threshold.
RISK_FLAG_THRESHOLD = 60


def _has_api_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _call_model(prompt: str, max_tokens: int = 1000, max_retries: int = 4, model: str = None) -> str:
    """
    Calls the model, automatically retrying with backoff if we hit a rate
    limit (429). Without this, any feature - flagging, summary, explain,
    chat - can fail right after a burst of calls (like the ~101 batch calls
    during the initial load) even though the request itself was fine; the
    free tier just needs a moment to free up capacity.

    Pass `model` to override the module-level default MODEL for this call -
    useful for a one-off task (like an eval script) that benefits from a
    different model's rate-limit profile (e.g. higher tokens-per-minute)
    without changing what the main app uses.
    """
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model or MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    delay = 3
    for attempt in range(max_retries):
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=120)
        if response.status_code == 429 and attempt < max_retries - 1:
            print(f"Rate limited (429) - waiting {delay}s before retry {attempt + 2}/{max_retries}...")
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    # Should not be reached, but keeps type-checkers happy.
    response.raise_for_status()


def _fallback_assess(transactions: List[dict]) -> List[dict]:
    """
    Used ONLY when GROQ_API_KEY is missing, so the project is still runnable
    without live credentials. A simple, explicit stand-in - never a
    competing implementation, and never used when the real agent works.
    """
    results = []
    for tx in transactions:
        results.append({
            "id": tx.get("id"),
            "risk_score": 0,
            "flag": False,
            "flag_reason": "",
        })
    return results


def _validate_result_item(item: dict, valid_ids: set) -> bool:
    """
    Check one result against the schema we need. Returns False if anything
    is missing, the wrong type, or doesn't match a real transaction we sent -
    so we never silently trust malformed or hallucinated data.
    """
    if not isinstance(item, dict):
        return False
    if not {"id", "risk_score", "flag_reason"}.issubset(item.keys()):
        return False
    if item.get("id") not in valid_ids:
        return False
    risk_score = item.get("risk_score")
    if not isinstance(risk_score, (int, float)) or isinstance(risk_score, bool):
        return False
    if not (0 <= risk_score <= 100):
        return False
    if not isinstance(item.get("flag_reason"), str):
        return False
    # A meaningfully risky score should come with an explanation - an
    # unexplained high score is a sign the model didn't follow the format.
    if risk_score >= 40 and not item["flag_reason"].strip():
        return False
    return True


def _check_grounding(item: dict, tx: dict) -> None:
    """
    Lightweight sanity check: if the model's flag_reason claims the amount
    was unusual but the actual amount_zscore doesn't support that, print a
    warning so this kind of mismatch is visible rather than silent. This
    doesn't change the result - it's a visibility check, since editing the
    model's stated reasoning ourselves would misrepresent what it actually
    said.
    """
    if item.get("risk_score", 0) < RISK_FLAG_THRESHOLD:
        return
    reason = (item.get("flag_reason") or "").lower()
    amount_zscore = tx.get("amount_zscore")
    claims_unusual_amount = any(word in reason for word in ["large amount", "unusual amount", "unusually large"])
    if claims_unusual_amount and amount_zscore is not None and abs(amount_zscore) < 1.0:
        print(f"GROUNDING WARNING: transaction {item.get('id')} flag_reason claims an unusual "
              f"amount, but amount_zscore is only {amount_zscore:.2f} (not actually unusual). "
              f"Reason given: '{item.get('flag_reason')}'")


def _validate_and_clean(results: List[dict], transactions: List[dict]) -> List[dict]:
    valid_ids = {tx.get("id") for tx in transactions}
    by_id = {tx.get("id"): tx for tx in transactions}

    seen_ids = set()
    cleaned = []
    rejected_count = 0

    for item in results:
        if _validate_result_item(item, valid_ids):
            item["flag"] = item["risk_score"] >= RISK_FLAG_THRESHOLD
            cleaned.append(item)
            seen_ids.add(item["id"])
            _check_grounding(item, by_id[item["id"]])
        else:
            rejected_count += 1

    missing_ids = valid_ids - seen_ids
    for missing_id in missing_ids:
        cleaned.extend(_fallback_assess([by_id[missing_id]]))

    if rejected_count or missing_ids:
        print(f"Validation: rejected {rejected_count} malformed result(s), "
              f"filled in {len(missing_ids)} missing transaction(s) with fallback.")

    return cleaned


def _assess_batch(transactions: List[dict]) -> List[dict]:
    """Send ONE batch (already small enough to fit a response) to the model."""
    payload = json.dumps(transactions, indent=2, default=str)
    prompt = f"""
You are a fraud risk analyst reviewing bank transactions. Each transaction
includes pre-computed behavioral signals (not raw demographic data):

- amount_zscore: how unusual this amount is compared to this account's own
  typical transaction size (0 = normal, further from 0 = more unusual)
- duration_zscore: how unusual this transaction's duration was compared to
  this account's own typical duration
- pct_of_balance: what fraction of the account's balance this transaction
  represents (close to 1.0 means it nearly emptied the account)
- hours_since_previous: hours since this account's last transaction (very
  small values mean rapid, back-to-back activity)
- device_changed / location_changed / ip_changed / channel_changed: whether
  this transaction came from a different device, location, IP, or channel
  than the account's previous transaction
- login_attempts: number of login attempts before this transaction (high
  values can indicate credential stuffing or brute-force attempts)
- account_transaction_count: how many transactions we have on record for
  this account in total. IMPORTANT: treat device/location/IP/channel
  "changed" signals as much weaker evidence when this count is low (e.g. 2-3
  transactions) - a change is unremarkable for an account we've barely seen
  before, and only becomes meaningful once there's an established pattern
  to deviate from. Weight these signals more heavily for accounts with a
  higher transaction count.

For each transaction, assign a risk_score from 0 to 100 reflecting how
risky it looks, weighing these signals holistically - no single signal
alone should automatically mean "very risky" or "not risky." A device
change alone is common (new phone) and should only push the score up
slightly; a device change AND a location change AND an unusually large
amount together should push it up much more. Use the full range
thoughtfully rather than clustering everything near 0 or 100 - a score
should reflect genuine relative risk, e.g. roughly:
- 0-20: nothing notable
- 20-40: minor, common patterns worth noting but not concerning
- 40-70: some real signals worth a human glancing at
- 70-100: multiple strong signals together, worth prioritizing for review

CRITICAL - stay grounded in the actual numbers: only describe a signal as
notable if its value actually supports that claim. For example, don't call
an amount "unusually large" unless amount_zscore is meaningfully far from 0
(roughly beyond +/-1.5) - if amount_zscore is close to 0 (e.g. between -0.5
and 0.5), the amount is typical for this account and should NOT be cited as
a reason, even if other signals (like a device or location change) are
present and worth flagging on their own. Never state a reason that isn't
actually backed by the specific values you were given for that transaction.

Return a JSON array with one object per transaction:
- id: the transaction id from the input
- risk_score: an integer from 0 to 100
- flag_reason: a short plain-English reason for the score if risk_score is
  40 or above, otherwise ""

Return a JSON array only - no extra commentary, no markdown fences.

Transactions:
{payload}
"""
    try:
        raw = _call_model(prompt, max_tokens=1000).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return _validate_and_clean(parsed, transactions)
    except Exception as exc:
        print(f"Model risk assessment call failed for a batch ({exc}); "
              f"using local fallback for these {len(transactions)} transaction(s).")
        return _fallback_assess(transactions)


def assess_fraud_risk(transactions: List[dict]) -> List[dict]:
    """
    The core agentic step: send transactions, each with pre-computed
    behavioral evidence, to the model in small batches and get back a flag
    decision + plain-English reasoning for each one. This is the ONLY place
    flag decisions get made in the whole project.

    Processed in batches of BATCH_SIZE rather than all at once - a single
    call covering thousands of transactions produces a response too long to
    fit in a reasonable token budget, which silently falls back to "no
    flags" for everything once parsing fails.
    """
    if not transactions:
        return []

    if not _has_api_key():
        print("GROQ_API_KEY not set - using a local fallback so the app still runs. "
              "Set the key in .env to see real agent behavior.")
        return _fallback_assess(transactions)

    all_results = []
    total_batches = (len(transactions) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(transactions), BATCH_SIZE):
        batch = transactions[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Assessing batch {batch_num}/{total_batches} ({len(batch)} transactions)...")
        all_results.extend(_assess_batch(batch))
        if batch_num < total_batches:
            time.sleep(15)  # be polite to the free-tier tokens-per-minute limit

    return all_results


def generate_summary(summary_context: dict) -> str:
    """
    Ask the model for a short natural-language summary of the reviewed
    batch. Takes a COMPACT summary dict (counts, totals, and the flagged
    transactions with their reasons) rather than the full raw transaction
    list - sending thousands of raw records in one prompt is too large for
    a single request and causes a 400 error.
    """
    total = summary_context.get("total_transactions", 0)
    if not total:
        return "No transactions were available to review."

    if not _has_api_key():
        flagged = summary_context.get("flagged_transactions", [])
        if not flagged:
            return "No transactions were flagged. (Local fallback summary - no API key set.)"
        reasons = "; ".join(
            f"{t.get('id')} was flagged: {t.get('flag_reason')}" for t in flagged[:3]
        )
        return f"Summary (local fallback, no API key set): {reasons}"

    payload = json.dumps(summary_context, indent=2, default=str)
    prompt = f"""
Write a short (2-3 sentence) plain-English summary of this batch of
already-reviewed bank transactions for a fraud analyst. Call out anything
flagged and why, and note the overall pattern.

Speak naturally, like a colleague giving a verbal update - never mention
field/variable names like "flagged_count" or "total_transactions"
directly. If nothing was flagged, just say something like "nothing here
looked suspicious" rather than restating the raw numbers.

Summary data:
{payload}
"""
    try:
        return _call_model(prompt, max_tokens=300).strip()
    except Exception as exc:
        return f"Summary unavailable right now ({exc})."


def explain_flag(transaction: dict) -> str:
    """Ask the model to explain, in more depth, why a specific transaction was flagged."""
    if not _has_api_key():
        reason = transaction.get("flag_reason") or "No strong concern was detected for this transaction."
        return f"(Local fallback - no API key set)\n{reason}"

    prompt = f"""
Briefly explain why this transaction might be worth a second look, in
plain English, for a fraud analyst reviewing this account's activity.

Transaction details:
{json.dumps(transaction, indent=2, default=str)}
"""
    try:
        return _call_model(prompt, max_tokens=300).strip()
    except Exception as exc:
        return f"Explanation unavailable right now ({exc})."


def chat_reply(user_message: str, context: dict) -> str:
    """Answer a free-form question about the reviewed batch, grounded in a compact summary."""
    if not _has_api_key():
        return ("(Local fallback - no API key set) I can help summarize flagged transactions "
                "once GROQ_API_KEY is set.")

    prompt = f"""
You are a helpful fraud-review assistant talking to a human analyst. Answer
the user's question using only the summary of transaction data below, but
speak naturally and conversationally, like a knowledgeable colleague - not
like you're reading the data back to them.

Never mention field/variable names like "flagged_count" or
"flagged_transactions" - translate them into plain English instead. For
example, if nothing was flagged, just say something like "none of these
looked suspicious" or "nothing stood out as risky here," not "the
flagged_count is 0."

Be concise (2-4 sentences).

Data summary:
{json.dumps(context, indent=2, default=str)}

User question: {user_message}
"""
    try:
        return _call_model(prompt, max_tokens=300).strip()
    except Exception as exc:
        return f"I couldn't reach the assistant right now ({exc})."
