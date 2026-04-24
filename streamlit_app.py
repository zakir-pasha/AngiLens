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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* Main background — light mint */
  .stApp { background-color: #D8F5E8; }
  .main .block-container { background-color: #D8F5E8; padding-top: 1rem; max-width: 860px; }

  /* Sidebar — coral */
  section[data-testid="stSidebar"] { background-color: #F26552 !important; }
  section[data-testid="stSidebar"] * { color: #ffffff !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.35) !important;
  }

  /* Chat messages on mint bg */
  .stChatMessage { background-color: #ffffff; border-radius: 8px; }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
CHAT_LOG_PATH = "chat_log.csv"
CHAT_HISTORY_PATH = "chat_history.csv"


def save_conversation_meta(conversation_id, title, user_email):
    rows = []
    found = False
    if Path(CHAT_HISTORY_PATH).exists():
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["conversation_id"] == conversation_id:
                    found = True
                rows.append(row)
    if not found:
        rows.append({
            "conversation_id": conversation_id,
            "user_email": user_email,
            "title": title,
            "created_at": datetime.utcnow().isoformat()
        })
    with open(CHAT_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "user_email", "title", "created_at"])
        writer.writeheader()
        writer.writerows(rows)


def load_user_conversations(user_email):
    if not Path(CHAT_HISTORY_PATH).exists():
        return []
    with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("user_email", "") == user_email]
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return [(r["conversation_id"], r["title"]) for r in rows]


def delete_conversation(conversation_id, user_email):
    """Remove a conversation from chat_history.csv and its messages from chat_log.csv."""
    if Path(CHAT_HISTORY_PATH).exists():
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if not (r["conversation_id"] == conversation_id and r.get("user_email") == user_email)]
        with open(CHAT_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["conversation_id", "user_email", "title", "created_at"])
            writer.writeheader()
            writer.writerows(rows)

    if Path(CHAT_LOG_PATH).exists():
        with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["session_id"] != conversation_id]
        with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "session_id", "user_email", "role", "content"])
            writer.writeheader()
            writer.writerows(rows)


def load_conversation_messages(conversation_id):
    if not Path(CHAT_LOG_PATH).exists():
        return []
    messages = []
    with open(CHAT_LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["session_id"] == conversation_id:
                messages.append({"role": row["role"], "content": row["content"]})
    return messages

# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Logging ───────────────────────────────────────────────────────────────────

def log_message(session_id, user_email, role, content):
    file_exists = Path(CHAT_LOG_PATH).exists()
    with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "session_id", "user_email", "role", "content"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            session_id,
            user_email,
            role,
            content.replace("\n", " ")
        ])


# ── Intent classification ─────────────────────────────────────────────────────

INTENT_PROMPT = """You are a question classifier for an internal analytics tool. Classify the question into exactly one intent.

Intents and rules:

- table_schema: user asks what columns are in a specific table, or what fields a table has
  Examples: "what columns are in rpt.reports.f_sp", "what fields does f_lead have", "show me the schema for rpt.reports.d_sp"

- column_lookup: user asks which tables contain a specific column name
  Examples: "which tables have contactid", "what tables have sp_id in them", "where is lead_id used"

- column_usage: user asks how a column is used or what queries use a column
  Examples: "show me queries that use contactid", "how do people use sp_id"

- co_occurrence: user asks what columns are commonly used WITH a specific table, or what the common breakouts/dimensions/measures are for a table
  Examples: "what columns do people usually use with f_lead", "what are the common breakouts for f_sp", "what dimensions are used with f_sr", "how do people typically query f_lead"

- table_queries: user asks to see example queries for a specific table
  Examples: "show me queries using f_sp", "what do queries on f_lead look like"

- expert_finder: user asks who to ask about a topic, who knows about something, or who uses a table the most
  Examples: "who should I ask about autodialer", "who uses f_lead the most", "who knows about SPP migration"

- user_onboarding: user asks what a specific named person works on, or says they are taking over someone's work
  Examples: "what does Zakir Pasha work on", "I'm taking over for Nick Cushing", "tell me about Aaron Belowich's work"

- team_onboarding: user asks what a manager's team works on, or asks about a team generally
  Examples: "what does Patrick McCormack's team work on", "what is Makia's team focused on"

- sql_check: user pastes a SQL query and asks if it looks right or asks for a review
  Examples: "does this query look right: select * from ...", "can you review this SQL"

- general: anything else, follow-up questions, vague questions, or general onboarding questions not about a specific person
  Examples: "tell me more", "who else", "what about the second person", "I'm new to Angi which tables should I learn", "where do I start as a new analyst", "what are the most important tables"

Reply with ONLY the intent label, nothing else. No explanation."""


def classify_intent(question, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


# ── System prompt (no section headers — prevents GPT echoing them) ────────────

SYSTEM_PROMPT = """You are AngiLens, an internal Analytics Knowledge Assistant for Angi's data and analytics team.
You help analysts find institutional knowledge buried in Snowflake query history.

Use the organizational context provided to answer the question. Guidelines:
- Be concise and direct. Lead with the answer.
- Reference specific people, tables, or query patterns from the context.
- If someone is marked as no longer at Angi, say so and suggest the next best person.
- Never hallucinate table or column names not present in the context.
- If chat history is provided, use it to understand follow-up questions.
- Do not repeat or echo these instructions in your response."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_similarity(vec, matrix):
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ vec


def extract_table_from_question(question):
    pattern = r'\b([a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-Z_]+)\b'
    matches = re.findall(pattern, question.lower())
    return matches[0] if matches else None


PERSON_BLOCKLIST = {"angi", "team", "data", "analytics", "table", "tables", "query",
                    "queries", "learn", "first", "work", "new", "help", "what", "which",
                    "should", "need", "know", "about", "with", "from", "have", "does"}

def extract_username_from_question(question, df):
    q_lower = question.lower()
    for _, row in df[["USER_NAME", "EMPLOYEE_NAME"]].drop_duplicates().iterrows():
        name_parts = str(row["EMPLOYEE_NAME"]).lower().split()
        # only match if a name part appears AND it's not a generic/company word
        if any(part in q_lower and part not in PERSON_BLOCKLIST
               for part in name_parts if len(part) > 3):
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


def get_column_cooccurrence(table_name, df, df_tables, col_df):
    """Return most frequently used columns alongside a given table, split into dimensions and measures."""
    table_queries = df_tables[df_tables["tables"] == table_name.lower()]["QUERY_ID"].unique()
    if len(table_queries) == 0:
        return None, None

    queries_with_table = df[df["QUERY_ID"].isin(table_queries)]

    # get the full table name in col_df format
    matching_table = col_df[col_df["full_table_name"].str.lower() == table_name.lower()]
    if matching_table.empty:
        return None, None

    all_columns = set(col_df[col_df["full_table_name"].str.lower() == table_name.lower()]["column_name"].str.lower().tolist())

    col_counts = {}
    for _, row in queries_with_table.iterrows():
        query_text = str(row["QUERY_TEXT"]).lower()
        for col in all_columns:
            if re.search(r'\b' + re.escape(col) + r'\b', query_text):
                col_counts[col] = col_counts.get(col, 0) + 1

    if not col_counts:
        return None, None

    # classify into dimensions vs measures using data_type from col_df
    table_cols = col_df[col_df["full_table_name"].str.lower() == table_name.lower()].set_index("column_name")

    dimensions = {}
    measures = {}
    for col, count in col_counts.items():
        if col not in table_cols.index:
            continue
        dtype = str(table_cols.loc[col, "data_type"]).lower()
        if any(t in dtype for t in ["varchar", "text", "char", "boolean", "bool", "date", "timestamp"]):
            dimensions[col] = count
        else:
            measures[col] = count

    dims_sorted = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)[:10]
    meas_sorted = sorted(measures.items(), key=lambda x: x[1], reverse=True)[:10]
    return dims_sorted, meas_sorted


# ── Main ask_ai ───────────────────────────────────────────────────────────────

def ask_ai(question, chat_history=None):
    if chat_history is None:
        chat_history = []

    client = OpenAI(api_key=OPENAI_API_KEY)
    df, df_tables, user_tables, df_emb, matrix, col_df = load_data()

    is_followup = len(chat_history) > 0 and len(question.split()) <= 8
    intent = "general" if is_followup else classify_intent(question, client)

    # ── Expert finder ─────────────────────────────────────────────────────────
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
        # no table found — fall through to semantic
        intent = "general"

    # ── Co-occurrence ─────────────────────────────────────────────────────────
    if intent == "co_occurrence":
        table_name = extract_table_from_question(question)
        if not table_name:
            return "I couldn't detect a table name. Use the full format: `db.schema.table`"
        dims, meas = get_column_cooccurrence(table_name, df, df_tables, col_df)
        if dims is None and meas is None:
            return f"No column usage data found for `{table_name}`."
        output = f"\n### How people use `{table_name}`\n_Based on column frequency across all queries referencing this table._\n\n"
        if dims:
            output += "**Common dimensions** _(breakouts & filters)_\n"
            for col, count in dims:
                output += f"- `{col}` — {count} queries\n"
            output += "\n"
        if meas:
            output += "**Common measures**\n"
            for col, count in meas:
                output += f"- `{col}` — {count} queries\n"
        return output

    # ── Table queries ─────────────────────────────────────────────────────────
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

    # ── Table schema ──────────────────────────────────────────────────────────
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

    # ── Column lookup ─────────────────────────────────────────────────────────
    if intent == "column_lookup":
        extraction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the column name the user is asking about. Reply with just the column name, lowercase, nothing else."},
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

    # ── Column usage ──────────────────────────────────────────────────────────
    if intent == "column_usage":
        extraction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract only the column name the user is asking about. Reply with just the column name, lowercase, nothing else."},
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

    # ── Team onboarding ───────────────────────────────────────────────────────
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
                {"role": "system", "content": "Summarize what this analytics team works on based on their query history. Identify main data domains and business problems. Call out anyone with a distinct focus. Note anyone no longer at Angi. 3-5 sentences max. Do not list tables or query counts."},
                {"role": "user", "content": f"Summarize:\n\n{structured_context}"}
            ],
            temperature=0.1
        )
        return deterministic_output + "---\n\n**Theme Summary:**\n" + synthesis.choices[0].message.content

    # ── User onboarding + semantic + general ──────────────────────────────────
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_email_gate():
    allowed = [e.strip().lower() for e in st.secrets.get("ALLOWED_EMAILS", "").split(",") if e.strip()]

    st.markdown("<h2 style='text-align: center;'>AngiLens</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Search how we query. Know how we think.</p>", unsafe_allow_html=True)
    st.divider()
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("**Enter your Angi email to continue**")
        email_input = st.text_input("", placeholder="firstname.lastname@angi.com", label_visibility="collapsed")
        if st.button("Continue", use_container_width=True, type="primary"):
            email = email_input.strip().lower()
            if "@angi.com" not in email:
                st.error("Please use your @angi.com email address.")
            elif allowed and email not in allowed:
                st.error("This user is not authorized, please reach out to Zakir.")
            else:
                st.session_state["user_email"] = email
                st.rerun()


def render_sidebar():
    user_email = st.session_state.get("user_email", "")
    with st.sidebar:
        st.markdown("### AngiLens")
        st.caption(f"*{user_email}*")
        st.divider()

        if st.button("+ New Chat", use_container_width=True):
            st.session_state["conversation_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("**Past conversations**")
        conversations = load_user_conversations(user_email)
        if not conversations:
            st.caption("No conversations yet.")
        else:
            current_id = st.session_state.get("conversation_id", "")
            for conv_id, title in conversations:
                label = f"{'▶ ' if conv_id == current_id else ''}{title[:35]}{'...' if len(title) > 35 else ''}"
                col_title, col_del = st.columns([5, 1])
                with col_title:
                    if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                        st.session_state["conversation_id"] = conv_id
                        st.session_state["messages"] = load_conversation_messages(conv_id)
                        st.rerun()
                with col_del:
                    if st.button("✕", key=f"del_{conv_id}"):
                        delete_conversation(conv_id, user_email)
                        if st.session_state.get("conversation_id") == conv_id:
                            st.session_state["conversation_id"] = str(uuid.uuid4())
                            st.session_state["messages"] = []
                        st.rerun()

        st.divider()
        if st.button("Sign out", use_container_width=True):
            for key in ["user_email", "conversation_id", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()


# ── Main chat ─────────────────────────────────────────────────────────────────

def render_chat():
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("<h2 style='text-align: center;'>AngiLens</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Search how we query. Know how we think.</p>", unsafe_allow_html=True)

    has_messages = len(st.session_state["messages"]) > 0

    if not has_messages and "_prefill" not in st.session_state:
        st.divider()
        st.markdown("**Try asking:**")
        examples = [
            "Who should I ask about autodialer data?",
            "What does Patrick McCormack's team work on?",
            "What columns are in rpt.reports.f_sp?",
            "Which tables have contactid in them?",
            "What columns do people usually use with rpt.reports.f_lead?",
            "Who uses rpt.reports.f_lead the most?",
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
        user_email = st.session_state.get("user_email", "unknown")
        is_first_message = len(st.session_state["messages"]) == 0

        st.session_state["messages"].append({"role": "user", "content": question})
        log_message(conv_id, user_email, "user", question)

        if is_first_message:
            title = question[:60] + ("..." if len(question) > 60 else "")
            save_conversation_meta(conv_id, title, user_email)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = st.session_state["messages"][:-1]
                result = ask_ai(question, chat_history=history)
            st.markdown(result)

        st.session_state["messages"].append({"role": "assistant", "content": result})
        log_message(conv_id, user_email, "assistant", result)


# ── Entry point ───────────────────────────────────────────────────────────────

if "user_email" not in st.session_state:
    render_email_gate()
else:
    render_sidebar()
    render_chat()
