# Fraud Risk Reviewer

An agentic fraud-triage tool that engineers behavioral risk signals from
real transaction data, hands them to an LLM agent to make the actual risk
judgment with plain-English reasoning, and surfaces everything in an
interactive Streamlit dashboard with a chat assistant for follow-up
questions.

Built and iterated end-to-end: real dataset investigation, feature
engineering, LLM-based reasoning, response validation, an independent
accuracy evaluation against labeled ground truth, and multiple rounds of
measured improvement.

## Highlights

- **Continuous 0-100 risk scoring**, not a binary yes/no - lets a human
  reviewer set their own risk tolerance via an interactive threshold
  slider, without re-running the agent.
- **Shrinkage estimation** to handle accounts with very little transaction
  history (a real "cold start" problem in this dataset), blending each
  account's own behavior with the population baseline.
- **Deliberate exclusion of demographic data** (age, occupation) from the
  risk-scoring logic, to avoid a well-documented bias pattern in real
  fraud systems.
- **A grounding check** that catches the model citing a reason its own
  data doesn't actually support (e.g., calling an amount "unusually large"
  when its z-score says otherwise) - and measurably improved accuracy when
  added, not just explanation quality.
- **An independent, labeled accuracy evaluation** against a second dataset
  with a different schema - built as a genuinely separate evaluation
  rather than force-fitting mismatched fields, which would have produced
  an invalid comparison.
- **Real engineering around free-tier API limits** - batching, exponential
  backoff on rate limits, and a model choice driven by actual quota math,
  not guesswork.

## Architecture

```
bank_transactions_data.csv
        |
        v
data_loader.py   -- loads the CSV, engineers risk features, stores in SQLite
        |            (makes NO risk-scoring decisions itself)
        v
ai_agent.py      -- the ONLY place that calls the AI model and the ONLY
        |            place a risk score gets assigned
        v
dashboard.py     -- Streamlit UI: pure display + an interactive threshold
                     slider + calls ai_agent.py directly for "explain this
                     flag" and the chat assistant
```

`run_agent.py` is a command-line entry point that runs the same pipeline
without the UI, useful for a quick sanity check.

## Files

- **`data_loader.py`** - loads the CSV, computes every behavioral feature
  described below, and stores raw + engineered data in SQLite. Contains no
  AI calls and no risk-scoring logic - purely feature computation and
  storage.
- **`ai_agent.py`** - the single agent module. Every AI call in the
  project (risk scoring, summaries, flag explanations, chat) goes through
  this file and nowhere else. Also handles batching, rate-limit retries,
  response validation, and the grounding sanity check.
- **`run_agent.py`** - CLI runner: load → engineer features → run the
  agent → print results. No UI, useful for a first test.
- **`dashboard.py`** - the Streamlit app. Purely a display/interaction
  layer; it never recomputes a risk score itself. Includes the
  interactive flagging-threshold slider.
- **`eval_synthetic_fraud.py`** - evaluates risk-scoring accuracy against
  a labeled synthetic fraud dataset with a different schema (see
  "Accuracy evaluation" below).
- **`eval_fraud_detection.py`** - an earlier, unused eval script written
  for a different, PCA-anonymized dataset. Superseded by
  `eval_synthetic_fraud.py`; kept for reference.

## Setup

1. Put `bank_transactions_data.csv` in this same folder (Kaggle's "Bank
   Transaction Dataset for Fraud Detection").
2. Create a `.env` file: `GROQ_API_KEY=your_key_here` (free key from
   console.groq.com, no credit card required).
3. `pip install pandas streamlit plotly requests python-dotenv`
4. `python3 run_agent.py` to test the pipeline from the terminal first.
5. `streamlit run dashboard.py` for the full UI.

**Optional - dev mode:** set `DEV_ROW_LIMIT` to process only a sample
while iterating (e.g. `export DEV_ROW_LIMIT=100`), instead of the full
~2,512 transactions, to avoid burning through API quota while debugging.
Unset it (`unset DEV_ROW_LIMIT`) for a real, full run.

## The risk-scoring logic

Every transaction is scored using **behavioral signals compared against
that specific account's own history** - not flat, one-size-fits-all
thresholds. The reasoning behind each signal:

- **`amount_zscore`** - how unusual this transaction's amount is compared
  to what's normal *for this account*. A $500 charge is unremarkable for
  one account and highly unusual for another; comparing against the
  account's own baseline is far more meaningful than a fixed dollar
  threshold.
- **`duration_zscore`** - same idea, applied to `TransactionDuration`.
- **`pct_of_balance`** - what fraction of the account's balance this one
  transaction represents. A transaction draining 90%+ of an account's
  balance is a classic account-takeover pattern.
- **`hours_since_previous`** - hours since this account's actual previous
  transaction. **Computed from the account's own sorted transaction
  history, not from the CSV's `PreviousTransactionDate` column** - that
  column's values all cluster within a few minutes on a single date
  regardless of each transaction's real date, meaning it's a
  data-generation artifact, not genuine transaction sequencing. Found by
  checking the actual date ranges in the file, not assumed from
  documentation.
- **`device_changed` / `location_changed` / `ip_changed` / `channel_changed`**
  - whether this transaction came from a different device, location, IP,
  or channel than this account's immediately preceding transaction. New
  device/location/IP together is a much stronger signal than any one
  alone.
- **`login_attempts`** - high values before a successful transaction can
  indicate credential stuffing or brute-force attempts.
- **`account_transaction_count`** - how many transactions we have on
  record for this account. This exists to solve a real "cold start"
  problem: this dataset averages only ~5 transactions per account, so a
  "device changed" flag is nearly meaningless for an account we've only
  seen twice, and becomes meaningful only once there's an established
  pattern to deviate from. The agent is explicitly instructed to weight
  change-signals more heavily as this count goes up.

### Shrinkage estimation (handling accounts with thin history)

With only ~5 transactions per account on average, computing `amount_zscore`
and `duration_zscore` from an account's own data alone is statistically
noisy - one ordinary purchase can look like an extreme outlier purely by
chance. The fix (a standard technique called **shrinkage estimation**):
each account's own mean/standard deviation is *blended* with the
population-wide mean/standard deviation, weighted by how many transactions
that account actually has:

```
weight = account_transaction_count / (account_transaction_count + SHRINKAGE_STRENGTH)
blended_mean = weight * account_mean + (1 - weight) * population_mean
blended_std  = weight * account_std  + (1 - weight) * population_std
```

`SHRINKAGE_STRENGTH` (currently 5, in `data_loader.py`) controls how many
of an account's own transactions it takes before its own baseline starts
to dominate the population baseline. An account with 1-2 transactions
leans almost entirely on the population baseline; an account with 20+
leans almost entirely on its own. The same technique is reused in
`eval_synthetic_fraud.py` for that dataset's per-user amount z-score.

### Deliberately excluded: demographic data

`CustomerAge` and `CustomerOccupation` are **never** fed into the
risk-scoring prompt. Using demographic traits to help decide who looks
"suspicious" is a well-documented source of bias in real fraud systems.
Age/occupation are still shown in the dashboard as context for a human
reviewer, but never influence the agent's risk score.

## Continuous risk scoring, not a binary flag

Rather than asking the model for a simple true/false, `ai_agent.py`'s
`assess_fraud_risk()` asks for a **risk_score from 0 to 100** per
transaction. A `RISK_FLAG_THRESHOLD` constant (default 60) then derives
the actual flag *in code*, not trusted from the model - which means:

- **The threshold can change without re-running the agent.** The
  dashboard exposes this as a live slider: moving it instantly changes
  which transactions count as "flagged," the flagged count, and the
  channel-breakdown chart, all recomputed from the same stored scores.
- **The tradeoff between catching more fraud and raising more false
  alarms becomes visible and adjustable**, instead of being locked into
  whatever a single model-decided boolean happened to output.
- **The evaluation script can report a genuine precision/recall curve**
  (see below) instead of one fixed number - a materially more informative
  result for understanding how the system actually behaves.

A transaction only gets a detailed `flag_reason` recorded once its score
reaches 40 - low scores don't need an explanation, which also keeps token
usage down across a large batch.

## How the agent weighs everything together

The prompt sent to the model explicitly instructs it to:

1. **Weigh signals holistically, not with rigid rules.** A device change
   alone is common (new phone) and should only nudge the score up
   slightly; a device change *and* a location change *and* an unusually
   large amount together should push it up much more.
2. **Discount change-signals for thin-history accounts.** Using
   `account_transaction_count` as described above.
3. **Stay grounded in the actual numbers.** The prompt explicitly forbids
   calling an amount "unusually large" unless `amount_zscore` is actually
   far from 0 (roughly beyond +/-1.5) - added after catching the model
   stating "unusually large amount" as a reason on a transaction whose
   z-score was only -0.17 (i.e., statistically typical).

Every result is schema-validated (`_validate_result_item`) before being
trusted - checking the right fields are present, correctly typed, and
that any meaningfully risky score comes with a non-empty reason - with
malformed results falling back safely rather than corrupting the
database. A separate `_check_grounding()` function automatically prints a
warning to the console any time the model's stated reason doesn't
actually match its own data, so this kind of mismatch is visible rather
than silent.

## Handling Groq's free-tier limits

Real issues came up running this against ~2,512 real transactions, and
all are handled in `ai_agent.py`:

- **Batching:** transactions are sent to the model in small batches
  (`BATCH_SIZE`), not all at once. Sending everything in a single request
  produces a response too long to fit in a reasonable token budget, which
  gets cut off mid-JSON and fails to parse - silently falling back to "no
  risk" for the *entire* dataset if unbatched.
- **Rate-limit retries:** `_call_model()` automatically retries with
  exponential backoff (3s, 6s, 12s, 24s) on a 429, since a burst of batch
  calls can temporarily exceed the free tier's per-minute token limit even
  when the account has daily quota remaining.
- **Model choice:** the project uses `llama-3.1-8b-instant` rather than
  `llama-3.3-70b-versatile`, specifically because of daily request quotas
  - the 70B model is capped at only 1,000 requests/day on Groq's free
  tier, while the 8B model allows 14,400/day, which comfortably supports a
  full run plus repeated iteration while developing.

## Tone

The summary, flag-explanation, and chat features are explicitly prompted
to speak naturally ("nothing here looked suspicious") rather than reciting
raw field names back at the user ("flagged_count is 0") - an early version
did the latter, and the prompts were tightened to fix it.

## Known data quirks (found by testing against the real file, not assumed from documentation)

- The IP column is named `"IP Address"` (with a space) in the raw CSV, not
  `"IPAddress"`.
- `PreviousTransactionDate` is not reliable for computing time between
  transactions (see `hours_since_previous` above) - it's computed
  independently instead.

## Accuracy evaluation

`bank_transactions_data.csv` has no fraud label column, so there's no way
to measure precision/recall against it directly - its risk scores can
only be spot-checked for reasonableness, not scored against ground truth.

To get a real, defensible accuracy number, `eval_synthetic_fraud.py`
evaluates the same reasoning approach against a separate, labeled dataset
(Kaggle's "Synthetic Financial Fraud Dataset" - 10,000 transactions, 5%
fraud rate, 1,000 users averaging 10 transactions each).

### Why this is a separate script, not a reuse of `assess_fraud_risk()`

This dataset has a genuinely different feature set - no account balance,
no login attempts, no device/location "changed" history - but it does
include two pre-computed reputation scores (`device_risk_score`,
`ip_risk_score`) the bank dataset doesn't have. Forcing this data through
the bank-specific prompt would mean fabricating fields it doesn't have
(`device_changed=False`, `login_attempts=0`, etc. for every row), which
would test the agent against made-up data rather than real signals - an
invalid comparison, not a shortcut.

Instead, `eval_synthetic_fraud.py` is a separate script with its own
schema-appropriate prompt, but deliberately **shares the same engineering
principles** as `ai_agent.py`: the same shrinkage-estimation technique,
the same schema validation, the same grounding-check pattern (adapted to
check `device_risk_score`/`ip_risk_score` claims instead of amount
claims), and the same underlying model-calling infrastructure (batching,
rate-limit retries).

### Final results

Tested on a balanced sample of 200 fraud + 400 legitimate transactions
(600 total), using the continuous risk-score approach at the default
flagging threshold of 60:

| Metric | Score |
|---|---|
| Precision | 83.2% |
| Recall | 91.5% |
| F1 Score | 87.1% |
| Accuracy | 91.0% |

Confusion matrix: 183 true positives, 37 false positives, 17 false
negatives, 363 true negatives.

**Reading these honestly:** when the agent flags something, it's right
about 5 out of 6 times (83.2% precision) - and it now catches over 9 out
of 10 real fraud cases in the sample (91.5% recall). The test sample was
1/3 fraud, so a model that never flags anything would already score
~66.7% accuracy - 91.0% is a substantial improvement over that floor.

### The improvement story, measured at each step

This result isn't the first thing that came out - it's the third
iteration, and each step's improvement was measured against the same
labeled data, not just assumed:

| Version | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Initial (binary flag, no grounding check) | 74.4% | 46.5% | 57.2% | 76.8% |
| + schema validation & grounding check | 76.6% | 54.0% | 63.3% | 79.2% |
| + continuous risk scoring (0-100) | 83.2% | 91.5% | 87.1% | 91.0% |

Two genuinely different engineering changes each produced a measurable
improvement: forcing the model to only cite reasons its own data actually
supports improved judgment quality, not just explanation honesty; and
replacing a binary decision with a continuous score - even before
considering the threshold-tuning it enables - gave the model more room to
express graded confidence instead of being forced into a premature
yes/no, which recall benefited from substantially (+37.5 points overall
across both changes).

### Precision/recall tradeoff by threshold

Because the agent now outputs a continuous score, `eval_synthetic_fraud.py`
reports metrics across a sweep of thresholds (30 through 80) rather than
one fixed cutoff - showing exactly how catching more fraud trades off
against more false alarms, so a threshold can be chosen based on the
actual cost of a missed fraud vs. an unnecessary review, rather than
guessed at.

### Also worth knowing

`device_risk_score` and `ip_risk_score` each correlate with the fraud
label at ~0.87 in this dataset - an unusually strong, near-deterministic
signal for synthetic data. That means these results likely overstate how
well this approach would generalize to messier, real-world fraud signals,
where no single feature is anywhere near that predictive. This is stated
directly in the eval script's own output, not just here.