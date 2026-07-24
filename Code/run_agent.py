"""
Loads the CSV, engineers behavioral features, and sends transactions through
the agent for a fraud-risk assessment - the same single-agent-module
pattern as before, just pointed at this new dataset instead of Plaid.
"""
import ai_agent
from data_loader import apply_agent_results, engineer_features, fetch_for_review, init_db, load_csv, save_transactions


def main() -> None:
    print("Loading and preparing data...")
    df = load_csv()
    df = engineer_features(df)

    conn = init_db()
    save_transactions(conn, df)
    print(f"Saved {len(df)} transactions to the database")

    transactions = fetch_for_review(conn)
    print(f"Sending {len(transactions)} transactions to the agent for risk assessment...")

    results = ai_agent.assess_fraud_risk(transactions)
    apply_agent_results(conn, results)

    flagged = [r for r in results if r.get("flag")]
    print(f"\nAgent flagged {len(flagged)} of {len(results)} transactions:")
    for item in flagged:
        print(f"  {item['id']}: {item['flag_reason']}")

    summary = ai_agent.generate_summary(results)
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
