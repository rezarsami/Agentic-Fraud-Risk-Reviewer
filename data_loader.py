"""
Loads the bank transaction CSV, computes behavioral risk-signal features per
account, and stores everything in SQLite.

This module does NOT decide whether anything is fraudulent - it only
computes evidence (z-scores, device/location/IP changes, etc.) that gets
handed to ai_agent.py, which is the only place an actual flag/no-flag
decision gets made. Keeping "compute evidence" and "make the judgment call"
in separate places is intentional - see CHANGES.md for why.

Expected CSV columns (from the Kaggle "Bank Transaction Dataset for Fraud
Detection" dataset):
TransactionID, AccountID, TransactionAmount, TransactionDate,
PreviousTransactionDate, TransactionType, Location, DeviceID, IPAddress,
MerchantID, Channel, CustomerAge, CustomerOccupation, TransactionDuration,
LoginAttempts, AccountBalance

Note: exact column names/date formats may need small tweaks once you load
the real file - Kaggle dataset pages don't always expose a machine-readable
schema, so this is built from public documentation of the dataset rather
than the file itself.
"""
import sqlite3
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "bank_transactions_data.csv"
DB_PATH = Path(__file__).resolve().parent / "transactions.db"


def load_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {CSV_PATH}. Download the CSV from Kaggle's "
            "'Bank Transaction Dataset for Fraud Detection' and save it as "
            "bank_transactions_data.csv in this folder."
        )
    df = pd.read_csv(CSV_PATH)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral risk signals per transaction, blending each account's
    own history with the overall population's behavior.

    Why blend instead of using each account's own stats alone: this dataset
    only has ~5 transactions per account on average. A z-score computed from
    4-5 data points is noisy - one ordinary purchase can look like an
    extreme outlier purely by chance, and "device changed since last time"
    is nearly meaningless for an account we've only seen twice. This is the
    classic "cold start" problem in fraud detection.

    The fix (a standard technique called shrinkage estimation): for
    accounts with little history, lean more on the population-wide
    baseline; for accounts with lots of history, lean more on their own
    baseline. An account's own signal is only trusted in proportion to how
    much history actually backs it up.
    """
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    grouped = df.groupby("AccountID")

    # How much history do we actually have for each account? This drives how
    # much weight its own baseline gets vs. the population baseline.
    account_counts = grouped.size().rename("account_transaction_count")
    df = df.join(account_counts, on="AccountID")

    # SHRINKAGE_STRENGTH: roughly "how many of the account's own transactions
    # would it take to trust its own baseline as much as the population's."
    # Higher = lean on the population baseline longer before trusting an
    # account's own thin history. 5 is a reasonable starting point given
    # this dataset's ~5 transactions/account average - tune if needed.
    SHRINKAGE_STRENGTH = 5

    def blended_stats(column: str, mean_col: str, std_col: str) -> pd.DataFrame:
        pop_mean = df[column].mean()
        pop_std = df[column].std()

        acct_stats = grouped[column].agg(["mean", "std"]).rename(
            columns={"mean": "acct_mean", "std": "acct_std"}
        )
        acct_stats["acct_std"] = acct_stats["acct_std"].fillna(pop_std)

        result = df[["AccountID", "account_transaction_count"]].join(acct_stats, on="AccountID")
        weight = result["account_transaction_count"] / (result["account_transaction_count"] + SHRINKAGE_STRENGTH)

        blended = pd.DataFrame(index=df.index)
        blended[mean_col] = weight * result["acct_mean"] + (1 - weight) * pop_mean
        blended[std_col] = weight * result["acct_std"] + (1 - weight) * pop_std
        return blended

    df = df.join(blended_stats("TransactionAmount", "blend_amount_mean", "blend_amount_std"))
    df = df.join(blended_stats("TransactionDuration", "blend_duration_mean", "blend_duration_std"))

    def safe_zscore(value, mean, std):
        if pd.isna(std) or std == 0:
            return 0.0
        return (value - mean) / std

    df["amount_zscore"] = df.apply(
        lambda r: safe_zscore(r["TransactionAmount"], r["blend_amount_mean"], r["blend_amount_std"]), axis=1
    )
    df["duration_zscore"] = df.apply(
        lambda r: safe_zscore(r["TransactionDuration"], r["blend_duration_mean"], r["blend_duration_std"]), axis=1
    )

    # Percentage of the account's balance this single transaction represents.
    df["pct_of_balance"] = df.apply(
        lambda r: (r["TransactionAmount"] / r["AccountBalance"]) if r["AccountBalance"] not in (0, None) else 0.0,
        axis=1,
    )

    # Hours since this account's own previous transaction. NOTE: the CSV's
    # "PreviousTransactionDate" column does not actually track this - every
    # value in it clusters within a few minutes on a single date regardless
    # of the transaction's real date, which means it's some kind of
    # record-sync timestamp rather than genuine transaction history. We
    # compute the real gap ourselves from each account's actual sorted
    # transaction dates instead of trusting that column.
    df["hours_since_previous"] = (
        grouped["TransactionDate"].diff().dt.total_seconds() / 3600
    )

    # Device / location / IP / channel changes vs. this account's own last
    # transaction. These booleans are still just facts ("did it change") -
    # account_transaction_count (passed to the agent) is what signals how
    # much weight a "changed" flag deserves. An account we've seen twice
    # should have its "device changed" flag treated as much weaker evidence
    # than an account with 20 consistent prior transactions.
    # Note: the real CSV names this column "IP Address" (with a space), not "IPAddress".
    df["device_changed"] = grouped["DeviceID"].shift(1).ne(df["DeviceID"]) & grouped["DeviceID"].shift(1).notna()
    df["location_changed"] = grouped["Location"].shift(1).ne(df["Location"]) & grouped["Location"].shift(1).notna()
    df["ip_changed"] = grouped["IP Address"].shift(1).ne(df["IP Address"]) & grouped["IP Address"].shift(1).notna()
    df["channel_changed"] = grouped["Channel"].shift(1).ne(df["Channel"]) & grouped["Channel"].shift(1).notna()

    return df




def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        account_id TEXT,
        amount REAL,
        transaction_date TEXT,
        transaction_type TEXT,
        location TEXT,
        device_id TEXT,
        ip_address TEXT,
        merchant_id TEXT,
        channel TEXT,
        customer_age INTEGER,
        customer_occupation TEXT,
        transaction_duration REAL,
        login_attempts INTEGER,
        account_balance REAL,
        amount_zscore REAL,
        duration_zscore REAL,
        account_transaction_count INTEGER,
        pct_of_balance REAL,
        hours_since_previous REAL,
        risk_score REAL DEFAULT 0,
        device_changed INTEGER,
        location_changed INTEGER,
        ip_changed INTEGER,
        channel_changed INTEGER,
        flag INTEGER DEFAULT 0,
        flag_reason TEXT DEFAULT ''
    )
    """)
    conn.commit()
    return conn


def clear_transactions(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM transactions")
    conn.commit()


def save_transactions(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    clear_transactions(conn)
    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT OR REPLACE INTO transactions (
                id, account_id, amount, transaction_date, transaction_type, location,
                device_id, ip_address, merchant_id, channel, customer_age,
                customer_occupation, transaction_duration, login_attempts, account_balance,
                amount_zscore, duration_zscore, account_transaction_count, pct_of_balance,
                hours_since_previous, device_changed, location_changed, ip_changed,
                channel_changed, flag, flag_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '')
            """,
            (
                str(row["TransactionID"]), str(row["AccountID"]), float(row["TransactionAmount"]),
                str(row["TransactionDate"]), str(row.get("TransactionType", "")), str(row.get("Location", "")),
                str(row.get("DeviceID", "")), str(row.get("IP Address", "")), str(row.get("MerchantID", "")),
                str(row.get("Channel", "")), int(row["CustomerAge"]) if pd.notna(row.get("CustomerAge")) else None,
                str(row.get("CustomerOccupation", "")), float(row.get("TransactionDuration", 0) or 0),
                int(row.get("LoginAttempts", 0) or 0), float(row.get("AccountBalance", 0) or 0),
                float(row["amount_zscore"]), float(row["duration_zscore"]), int(row["account_transaction_count"]),
                float(row["pct_of_balance"]),
                float(row["hours_since_previous"]) if pd.notna(row["hours_since_previous"]) else None,
                int(bool(row["device_changed"])), int(bool(row["location_changed"])),
                int(bool(row["ip_changed"])), int(bool(row["channel_changed"])),
            ),
        )
    conn.commit()


def fetch_for_review(conn: sqlite3.Connection) -> list[dict]:
    """Get transactions in the shape the agent needs to assess risk."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, account_id, amount, transaction_type, channel, location,
               amount_zscore, duration_zscore, account_transaction_count, pct_of_balance,
               hours_since_previous, device_changed, location_changed, ip_changed,
               channel_changed, login_attempts
        FROM transactions
        ORDER BY transaction_date
    """).fetchall()
    return [dict(row) for row in rows]


def apply_agent_results(conn: sqlite3.Connection, results: list[dict]) -> None:
    for item in results:
        conn.execute(
            "UPDATE transactions SET risk_score = ?, flag = ?, flag_reason = ? WHERE id = ?",
            (
                item.get("risk_score", 0),
                1 if item.get("flag") else 0,
                item.get("flag_reason") or "",
                item.get("id"),
            ),
        )
    conn.commit()


def main() -> None:
    print("Loading CSV and engineering risk features...")
    df = load_csv()
    df = engineer_features(df)
    print(f"Loaded {len(df)} transactions across {df['AccountID'].nunique()} accounts")

    conn = init_db()
    save_transactions(conn, df)
    print(f"Saved to {DB_PATH}")
    print("Run run_agent.py next to have the agent assess fraud risk.")


if __name__ == "__main__":
    main()
