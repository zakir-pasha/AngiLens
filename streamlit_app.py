import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import csv
import uuid
from pathlib import Path
from datetime import datetime
from openai import OpenAI

st.set_page_config(
    page_title="AngiLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #1a1a2e; }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .main .block-container { padding-top: 1rem; max-width: 860px; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CHAT_LOG_PATH = "chat_log.csv"

@st.cache_resource(show_spinner="Loading AngiLens...")
def load_data():
    df = pd.read_csv("df_4_13.csv")
    df["START_TIME"] = pd.to_datetime(df["START_TIME"], format="mixed", utc=True, errors="coerce")
    df.columns = df.columns.str.upper()

    def extract_tables(query_text):
        if pd.isna(query_text):
            return []
        pattern = r'\b([a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-Z_]+)\b'
        matches = re.findall(pattern, str(query_text).lower())
        return list(set([m for m in matches if m.count('.') == 2]))

    df["tables"] = df["QUERY_TEXT"].apply(extract_tables)
    df_tables = df.explode("tables").dropna(subset=["tables"])

    user_tables = (
        df_tables.groupby(["USER_NAME", "tables"]).size()
        .reset_index(name="query_count")
        .sort_values(["USER_NAME", "query_count"], ascending=[True, False])
    )

    with open("embeddings_cache.pkl", "rb") as f:
        df_emb = pickle.load(f)
    matrix = np.vstack(df_emb["embedding"].values)

    col_df = pd.read_csv("column_metadata.csv")
    col_df.columns = col_df.columns.str.lower()

    return df, df_tables, user_tables, df_emb, matrix, col_df


def log_message(session_id, role, content):
    file_exists = Path(CHAT_LOG_PATH).exists()
    with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "session_id", "role", "content"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            session_id,
            role,
            content.replace("\n", " ")
        ])


SYSTEM_PROMPT = """
You are AngiLens, an internal Analytics Knowledge Assistant for Angi's data and analytics team.
Your job is to help analysts find institutional knowledge buried in Snowflake query history.

Before answering, identify which category best matches the question and follow only those rules.

1. SUBJECT EXPERT QUESTION
   → Identify the person who appears most frequently for that topic.
   → Lead with their name. Mention 1-2 tables showing their expertise.
   → If no longer at Angi, say so and suggest the next best person.
   → 3-4 sentences max.

2. USER ONBOARDING
   → Summarize their top focus areas based on most-used tables.
   → Group by domain. Mention their manager. Flag if terminated.
   → 5-6 sentences.

3. TEAM ONBOARDING
   → Summarize collective focus and individual standouts.
   → Do NOT list tables. 3-5 sentences.

4. SQL CHECK
   → Identify issues: missing filters, bad joins, wrong aggregation.
   → Suggest a fix.

5. GENERAL / SEMANTIC
   → Answer conversationally using context provided.
   → Use chat history to understand follow-up questions.

Always be concise. Never hallucinate table or column names.
"""


def cosine_similarity(vec, matrix):
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ vec


def classify_intent(question, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Classify into one of:
expert_finder, user_onboarding, team_onboarding, table_schema,
column_lookup, column_usage, table_queries, sql_check, general
Reply with ONLY the label."""},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


def extract_table_from_question(question):
    pattern = r'\b([a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-Z_]+)\b'
    matches = re.findall(pattern, question.lower())
    return matches[0] if matches else None


def extract_username_from_question(question, df):
    q_lower = question.lower()
    for _, row in df[["USER_NAME", "EMPLOYEE_NAME"]].drop_duplicates().iterrows():
        name_parts = str(row["EMPLOYEE_NAME"]).lower().split()
        if any(part in q_lower for part in name_parts if len(part) > 3):
            return row["USER_NAME"]
    return None


def get_user_top_tables(user_name, user_tables, top_k=5):
    result = user_tables[user_tables["USER_NAME"] == user_name].head(top_k)
    return result if not result.empty else None


def get_recent_queries_for_table(table_name, df, df_tables, top_k=5):
    matches = df_tables[df_tables["tables"] == table_name.lower()]
    if matches.empty:
        return None
    recent_ids = matches.sort_values("START_TIME", ascending=False)["QUERY_ID"].unique()[:top_k]
    return df[df["QUERY_ID"].isin(recent_ids)]


def search_similar_queries(question, df_emb, matrix, client, top_k=5):
    response = client.embeddings.create(model="text-embedding-3-small", input=[question])
    q_vec = np.array(response.data[0].embedding)
    scores = cosine_similarity(q_vec, matrix)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return df_emb.iloc[top_idx]


def ask_ai(question, chat_history=None):
    if chat_history is None:
        chat_history = []

    client = OpenAI(api_key=OPENAI_API_KEY)
    df, df_tables, user_tables, df_emb, matrix, col_df = load_data()

    is_followup = len(chat_history) > 0 and len(question.split()) <= 8
    intent = "general" if is_followup else classify_intent(question, client)

    if intent == "expert_finder":
        table_name = extract_table_from_question(question)
        if table_name:
            top_users = (
                df_tables[df_tables["tables"] == table_name]
                .groupby("USER_NAME").size()
                .reset_index(name="query_count")
                .sort_values("query_count", ascending=False)
                .head(5)
            )
            if top_users.empty:
                return f"No query history found for `{table_name}`."
            output = f"\n### Who uses `{table_name}` the most?\n"
            for _, row in top_users.iterrows():
                match = df[df["USER_NAME"] == row["USER_NAME"]].iloc[0]
                status = " ⚠️ (no longer at Angi)" if match["TERMINATED"] == "Y" else ""
                output += f"- **{match['EMPLOYEE_NAME']}**{status} — {row['query_count']} queries\n"
            return output
        intent = "general"

    if intent == "table_queries":
        table_name = extract_table_from_question(question)
        if not table_name:
            return "I couldn't detect a table name. Use the full format: `db.schema.table`"
        recent = get_recent_queries_for_table(table_name, df, df_tables)
        if recent is None or recent.empty:
            return f"No queries found for `{table_name}`."
        output = f"\n### Recent Queries Using `{table_name}`\n"
        for i, (_, row) in enumerate(recent.iterrows(), 1):
            match = df[df["USER_NAME"] == row["USER_NAME"]].iloc[0]
            status = " ⚠️ (no longer at Angi)" if match["TERMINATED"] == "Y" else ""
            output += f"\n{i}. **{match['EMPLOYEE_NAME']}**{status}\n```sql\n{row['QUERY_TEXT']}\n```\n"
        return output

    if intent == "table_schema":
        table_name = extract_table_from_question(question)
        if not table_name:
            return "I couldn't detect a table name. Use the full format: `db.schema.table`"
        matches = col_df[col_df["full_table_name"].str.lower() == table_name.lower()]
        if matches.empty:
            return f"`{table_name}` doesn't exist in the AngiLens dataset."
        output = f"\n### Columns in `{table_name}`\n{len(matches)} columns:\n\n"
        for _, row in matches.iterrows():
            output += f"- `{row['column_name']}` ({row['data_type']})\n"
        return output

    if intent == "column_lookup":
        extraction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the column name. Reply with just the column name, lowercase, nothing else."},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        column_name = extraction.choices[0].message.content.strip().lower()
        matches = col_df[col_df["column_name"].str.lower() == column_name]
        if matches.empty:
            return f"No tables found containing `{column_name}`."
        output = f"\n### Tables containing `{column_name}`\nFound in {len(matches)} tables:\n\n"
        for _, row in matches.iterrows():
            output += f"- `{row['full_table_name']}`\n"
        return output

    if intent == "column_usage":
        extraction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the column name. Reply with just the column name, lowercase, nothing else."},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        column_name = extraction.choices[0].message.content.strip().lower()
        tables_with_col = col_df[col_df["column_name"].str.lower() == column_name]["full_table_name"].tolist()
        if not tables_with_col:
            return f"No queries found using `{column_name}`."
        matches = df_tables[df_tables["tables"].isin(tables_with_col)]
        if matches.empty:
            return f"No queries found using `{column_name}`."
        recent_ids = matches.sort_values("START_TIME", ascending=False)["QUERY_ID"].unique()[:5]
        results = df[df["QUERY_ID"].isin(recent_ids)]
        output = f"\n### Queries Using `{column_name}`\n"
        for i, (_, row) in enumerate(results.iterrows(), 1):
            status = " ⚠️ (no longer at Angi)" if row["TERMINATED"] == "Y" else ""
            output += f"\n{i}. **{row['EMPLOYEE_NAME']}**{status}\n```sql\n{row['QUERY_TEXT']}\n```\n"
        return output

    if intent == "team_onboarding":
        extraction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the manager or supervisor name. Reply with just the full name, nothing else."},
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        supervisor_name = extraction.choices[0].message.content.strip()
        team_members = df[df["SUPERVISOR_NAME"].str.lower() == supervisor_name.lower()][
            ["USER_NAME", "EMPLOYEE_NAME", "TERMINATED"]
        ].drop_duplicates()

        if team_members.empty:
            return f"No team found for `{supervisor_name}`. Try using their full name."

        deterministic_output = f"\n### {supervisor_name}'s Team\n\n"
        structured_context = f"TEAM SUMMARY FOR: {supervisor_name}\n\n"

        for _, member in team_members.iterrows():
            status = " ⚠️ (no longer at Angi)" if member["TERMINATED"] == "Y" else ""
            top_tables = get_user_top_tables(member["USER_NAME"], user_tables, top_k=5)
            deterministic_output += f"**{member['EMPLOYEE_NAME']}**{status}\n"
            structured_context += f"### {member['EMPLOYEE_NAME']}{status}\n"
            if top_tables is not None:
                for _, row in top_tables.iterrows():
                    deterministic_output += f"  - `{row['tables']}` ({row['query_count']} queries)\n"
                    structured_context += f"  - {row['tables']} ({row['query_count']} queries)\n"
            else:
                deterministic_output += "  - No query history found\n"
                structured_context += "  - No query history found\n"
            user_recent = df[df["USER_NAME"] == member["USER_NAME"]].sort_values("START_TIME", ascending=False).head(3)
            structured_context += "\nRECENT QUERIES:\n"
            for i, (_, row) in enumerate(user_recent.iterrows(), 1):
                structured_context += f"Query {i}:\n{row['QUERY_TEXT']}\n"
            deterministic_output += "\n"
            structured_context += "\n"

        synthesis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize what this analytics team works on. Identify main domains, call out distinct focuses, note anyone no longer at Angi. 3-5 sentences, no table lists."},
                {"role": "user", "content": f"Summarize:\n\n{structured_context}"}
            ],
            temperature=0.1
        )
        return deterministic_output + "---\n\n**Theme Summary:**\n" + synthesis.choices[0].message.content

    # user onboarding + semantic + general
    structured_context = ""
    detected_user = extract_username_from_question(question, df)

    if detected_user:
        match = df[df["USER_NAME"] == detected_user].iloc[0]
        status_note = " (no longer at Angi)" if match["TERMINATED"] == "Y" else ""
        top_tables = get_user_top_tables(detected_user, user_tables)
        if top_tables is not None:
            structured_context += f"USER SUMMARY: {match['EMPLOYEE_NAME']}{status_note}\nSUPERVISOR: {match['SUPERVISOR_NAME']}\n\nTOP TABLES:\n"
            for _, row in top_tables.iterrows():
                structured_context += f"- {row['tables']} (used {row['query_count']} times)\n"
        user_recent = df[df["USER_NAME"] == detected_user].sort_values("START_TIME", ascending=False).head(5)
        structured_context += "\nRECENT QUERIES:\n"
        for i, (_, row) in enumerate(user_recent.iterrows(), 1):
            structured_context += f"\nQuery {i}:\n{row['QUERY_TEXT']}\n"

    if not detected_user or intent in ("general", "sql_check"):
        similar = search_similar_queries(question, df_emb, matrix, client, top_k=5)
        structured_context += "\nRELEVANT HISTORICAL QUERIES:\n"
        for _, row in similar.iterrows():
            status_note = " (no longer at Angi)" if row["TERMINATED"] == "Y" else ""
            structured_context += f"\nEmployee: {row['EMPLOYEE_NAME']}{status_note}\nSupervisor: {row['SUPERVISOR_NAME']}\n{row['QUERY_TEXT']}\n---\n"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"User Question:\n{question}\n\nOrganizational Context:\n{structured_context}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.1
    )
    return response.choices[0].message.content


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔍 AngiLens")
        st.caption("*Search how we query. Know how we think.*")
        st.divider()

        if st.button("+ New Chat", use_container_width=True):
            st.session_state["conversation_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("**This session**")
        messages = st.session_state.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            st.caption("No messages yet.")
        else:
            for msg in user_messages:
                st.caption(f"› {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")


def render_chat():
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("## 🔍 AngiLens")
    st.caption("Search how we query. Know how we think.")

    if not st.session_state["messages"]:
        st.divider()
        st.markdown("**Try asking:**")
        examples = [
            "Who should I ask about autodialer data?",
            "What does Patrick McCormack's team work on?",
            "What columns are in rpt.reports.f_sp?",
            "Who uses rpt.reports.f_lead the most?",
            "I just joined Makia's team, what tables do I need to know?",
            "What columns do people usually use with rpt.reports.f_lead?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()
        st.divider()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prefill = st.session_state.pop("_prefill", None)
    question = st.chat_input("Ask a question...") or prefill

    if question:
        conv_id = st.session_state["conversation_id"]
        st.session_state["messages"].append({"role": "user", "content": question})
        log_message(conv_id, "user", question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = st.session_state["messages"][:-1]
                result = ask_ai(question, chat_history=history)
            st.markdown(result)

        st.session_state["messages"].append({"role": "assistant", "content": result})
        log_message(conv_id, "assistant", result)


render_sidebar()
render_chat()
