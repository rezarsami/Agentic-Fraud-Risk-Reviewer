"""
Dashboard for the bank transaction fraud-risk reviewer.

Display-only, same as before: every AI decision comes from ai_agent.py,
never recomputed independently here.
"""
import sqlite3
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_agent
from data_loader import DB_PATH, apply_agent_results, engineer_features, fetch_for_review, init_db, load_csv, save_transactions

st.set_page_config(page_title="Fraud Risk Reviewer", layout="wide")

# Dev-mode row limit: while iterating/debugging, set this env var to process
# only a small sample instead of all 2,512 transactions - this avoids
# burning through the free-tier daily request quota on every restart. Set
# DEV_ROW_LIMIT=0 (or unset it) to process everything for a real run.
DEV_ROW_LIMIT = int(os.getenv("DEV_ROW_LIMIT", "0"))


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_transactions() -> pd.DataFrame:
    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT id, account_id, amount, transaction_date, transaction_type, location,
                   device_id, channel, customer_age, customer_occupation,
                   amount_zscore, duration_zscore, account_transaction_count, pct_of_balance,
                   hours_since_previous, device_changed, location_changed, ip_changed, channel_changed,
                   login_attempts, risk_score, flag, flag_reason
            FROM transactions
            ORDER BY transaction_date DESC
        """).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return pd.DataFrame([dict(row) for row in rows])


def load_and_review() -> str:
    """Load the CSV, engineer features, store them, and run the agent."""
    try:
        df = load_csv()
        df = engineer_features(df)

        if DEV_ROW_LIMIT:
            df = df.head(DEV_ROW_LIMIT)
            print(f"DEV_ROW_LIMIT set - processing only {len(df)} of the full dataset.")

        conn = init_db()
        save_transactions(conn, df)

        transactions = fetch_for_review(conn)
        results = ai_agent.assess_fraud_risk(transactions)
        apply_agent_results(conn, results)
        conn.close()

        return f"Loaded {len(df)} transactions and ran them through the agent."
    except Exception as exc:
        return f"Load failed: {exc}"


def get_selected_row_indices(selection_state) -> list[int]:
    if selection_state is None:
        return []
    selection = selection_state.get("selection", {}) if hasattr(selection_state, "get") else {}
    rows = selection.get("rows", []) if isinstance(selection, dict) else []
    if not isinstance(rows, list):
        return []
    return [int(row) for row in rows if isinstance(row, (int, float))]


def build_chat_context(transactions: pd.DataFrame) -> dict:
    flagged = transactions[transactions["flag"] == 1]
    # Include id + amount + account + the actual reason for each flagged
    # transaction (capped at 30 to keep the prompt small) - without this,
    # the assistant can only see IDs and counts, and can't answer "why was
    # X flagged" questions with any real detail.
    flagged_details = [
        {
            "id": str(row["id"]),
            "account_id": str(row["account_id"]),
            "amount": float(row["amount"]),
            "risk_score": float(row.get("risk_score", 0)),
            "flag_reason": row.get("flag_reason") or "",
        }
        for _, row in flagged.head(30).iterrows()
    ]
    return {
        "total_transactions": len(transactions),
        "flagged_count": int(len(flagged)),
        "flagged_transactions": flagged_details,
        "total_volume": float(transactions["amount"].sum()),
        "channel_breakdown": transactions["channel"].value_counts().to_dict(),
    }


st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    .hero-card {
        background: linear-gradient(120deg, #0f172a 0%, #b91c1c 45%, #f97316 100%);
        padding: 1.25rem 1.35rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(185, 28, 28, 0.18);
    }
    .hero-card h1 { margin: 0; font-size: 2rem; }
    .hero-card p { margin: 0.35rem 0 0; font-size: 1rem; opacity: 0.95; }
    </style>
    <div class="hero-card">
        <h1>Fraud Risk Reviewer</h1>
        <p>Behavioral fraud-risk triage powered by a single AI agent that reasons over
        account-relative signals - not raw demographics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "load_done" not in st.session_state:
    st.session_state.load_done = False

if not st.session_state.load_done:
    with st.spinner("Loading data and running the agent..."):
        message = load_and_review()
    st.session_state.load_done = True
    st.session_state.load_message = message

st.caption(st.session_state.get("load_message", ""))

transactions = load_transactions()

if transactions.empty:
    st.info("No transactions available yet. Make sure bank_transactions_data.csv is in this folder.")
    st.stop()

transactions["risk_score"] = transactions["risk_score"].fillna(0)

st.subheader("Flagging threshold")
st.caption(
    "The agent assigns every transaction a 0-100 risk score. Moving this slider changes "
    "which transactions count as \"flagged\" below, without re-running the agent - a lower "
    "threshold catches more real fraud at the cost of more false alarms, and vice versa."
)
risk_threshold = st.slider("Flag transactions scoring at or above:", 0, 100, ai_agent.RISK_FLAG_THRESHOLD)
transactions["flag"] = (transactions["risk_score"] >= risk_threshold).astype(int)
transactions = transactions.sort_values("risk_score", ascending=False).reset_index(drop=True)

summary = ai_agent.generate_summary(build_chat_context(transactions))
st.subheader("Summary")
st.success(summary)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", len(transactions))
c2.metric("Flagged", int((transactions["flag"] == 1).sum()))
c3.metric("Total Volume", f"${transactions['amount'].sum():,.2f}")

st.subheader("Flagged transactions by channel")
flagged_df = transactions[transactions["flag"] == 1]
if not flagged_df.empty:
    channel_counts = flagged_df["channel"].value_counts()
    fig = go.Figure(go.Bar(x=channel_counts.index, y=channel_counts.values, marker_color="#b91c1c"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Flagged count",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No flagged transactions to chart yet.")

st.subheader("Transactions")
display_df = transactions.copy()
display_df["Flagged"] = display_df["flag"].apply(lambda v: "Yes" if v == 1 else "No")
display_df["Amount"] = display_df["amount"].apply(lambda v: f"${v:,.2f}")
display_cols = display_df[["transaction_date", "account_id", "Amount", "channel", "location", "risk_score", "Flagged"]]
display_cols = display_cols.rename(columns={
    "transaction_date": "Date", "account_id": "Account", "channel": "Channel", "location": "Location",
    "risk_score": "Risk Score",
})

selection_state = st.dataframe(
    display_cols, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
)
selected_indices = get_selected_row_indices(selection_state)
selected_index = selected_indices[0] if selected_indices else None

st.subheader("Flag details")
if selected_index is not None:
    tx = transactions.iloc[selected_index].to_dict()
    st.info(f"Account: {tx.get('account_id')}  |  Amount: ${tx.get('amount'):,.2f}  |  Channel: {tx.get('channel')}")
    st.metric("Risk score", f"{tx.get('risk_score', 0):.0f} / 100")
    reason = tx.get("flag_reason") or ""
    if reason.strip():
        st.write(f"**Reason:** {reason}")
    else:
        st.write("**Reason:** No detailed reason recorded - this score was low enough "
                 "that the agent wasn't asked to explain it in depth.")
    st.markdown(f"""
**Signals:**
- Amount z-score: `{tx.get('amount_zscore'):.2f}` (0 = typical for this account, further from 0 = more unusual)
- Account history: `{tx.get('account_transaction_count')}` transactions on record
- Percent of balance: `{tx.get('pct_of_balance', 0) * 100:.0f}%`
- Device changed: `{bool(tx.get('device_changed'))}`
- Location changed: `{bool(tx.get('location_changed'))}`
- Login attempts: `{tx.get('login_attempts')}`
""")
    if st.button("Explain this flag"):
        explanation = ai_agent.explain_flag(tx)
        st.text_area("Agent explanation", explanation, height=150)
else:
    st.info("Click a transaction row to see its risk signals.")

st.subheader("Ask the assistant")
chat_prompt = st.text_input("Ask about the flagged transactions", placeholder="Example: which accounts had the most flags?")
if st.button("Ask"):
    if chat_prompt.strip():
        with st.spinner("Thinking..."):
            context = build_chat_context(transactions)
            reply = ai_agent.chat_reply(chat_prompt, context)
        st.markdown(f"**You:** {chat_prompt}")
        st.markdown(f"**Assistant:** {reply}")
    else:
        st.warning("Type a question first.")
