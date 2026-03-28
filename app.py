"""
Conversational BI Agent — Streamlit UI
CSVs are loaded automatically from ./data folder — no upload needed.
"""

import streamlit as st
import os
import sys
import logging

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_APP_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Instacart BI Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp { background: linear-gradient(180deg, #06080f 0%, #0a0e18 40%, #080b14 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0f1c 0%, #0d1120 100%); border-right: 1px solid #1e2740; }
[data-testid="stSidebar"] * { color: #a8b0c4 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #e8ecff !important; font-family: 'IBM Plex Mono', monospace !important; }

/* Chat thread */
[data-testid="stVerticalBlock"] > [data-testid="stChatMessageContainer"] {
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}
[data-testid="stChatMessage"] {
  background: rgba(13, 17, 32, 0.85) !important;
  border: 1px solid #1e2a48 !important;
  border-radius: 16px !important;
  padding: 4px 8px !important;
  margin-bottom: 12px !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  border-color: #2a4080 !important;
  background: linear-gradient(145deg, #121a2e 0%, #0f1628 100%) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-color: #1a4a3a !important;
  background: linear-gradient(145deg, #0f1c24 0%, #0c1820 100%) !important;
}

/* Chat input dock */
[data-testid="stChatInput"] {
  border-radius: 14px !important;
  border: 1px solid #2a3f6f !important;
  background: #0a1020 !important;
}
[data-testid="stChatInput"] textarea {
  color: #e0e8ff !important;
  font-size: 1rem !important;
}

.bi-header {
  background: linear-gradient(135deg, #0c1224 0%, #101a32 50%, #0a1628 100%);
  border: 1px solid #1e3a5f;
  border-radius: 16px;
  padding: 18px 24px;
  margin-bottom: 8px;
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}
.bi-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem; font-weight: 600; color: #5ba3ff; margin: 0; letter-spacing: -0.3px; }
.bi-header p { color: #7a8aad; margin: 6px 0 0 0; font-size: 0.88rem; }

.welcome-wrap {
  max-width: 900px;
  margin: 0 auto 16px auto;
  padding: 28px 20px;
  text-align: center;
  border: 1px dashed #243050;
  border-radius: 16px;
  background: rgba(10, 14, 28, 0.5);
}
.welcome-wrap .emoji { font-size: 2.5rem; margin-bottom: 8px; }
.welcome-wrap .title { font-family: 'IBM Plex Mono', monospace; color: #6b7fa3; font-size: 0.95rem; }
.welcome-wrap .sub { color: #3d4f78; font-size: 0.8rem; margin-top: 6px; }

.meta-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.badge { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; font-weight: 500; }
.badge-intent { background: #0d2b1f; color: #3ecf8e; border: 1px solid #1a5c40; }
.badge-time { background: #1a1f0d; color: #9ecf3e; border: 1px solid #3a5c1a; }
.badge-cache { background: #1a0d2b; color: #cf3e8e; border: 1px solid #5c1a40; }
.badge-followup { background: #0d1a2b; color: #3e8ecf; border: 1px solid #1a405c; }
.badge-retry { background: #2b1a0d; color: #cf8e3e; border: 1px solid #5c401a; }

.insight-text { color: #c5d4f0; font-size: 0.98rem; line-height: 1.65; }

[data-testid="stDataFrame"] { border: 1px solid #243050 !important; border-radius: 10px !important; }
.status-ok { color: #3ecf8e; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }
.status-err { color: #e84c4c; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080b14; }
::-webkit-scrollbar-thumb { background: #2a3550; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


def init_session():
    defaults = {
        "messages": [],
        "orchestrator": None,
        "data_loaded": False,
        "load_status": {},
        "show_sql": True,
        "initialized": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

DATA_DIR = os.path.join(_APP_DIR, "data")


@st.cache_resource(show_spinner=False)
def load_backend(data_dir: str):
    from utils.llm_client import get_llm_client
    from utils.data_loader import DataLoader
    from agents.orchestrator import QueryOrchestrator

    llm = get_llm_client()
    loader = DataLoader(data_dir=data_dir)
    status = loader.load_all()
    orchestrator = QueryOrchestrator(llm=llm, data_loader=loader)
    return orchestrator, status


if not st.session_state.initialized:
    with st.spinner(
        "🔄 Connecting to OpenRouter and registering tables… "
        "(Large CSVs use lazy views — first heavy query on big tables may take longer.)"
    ):
        try:
            orchestrator, load_status = load_backend(DATA_DIR)
            st.session_state.orchestrator = orchestrator
            st.session_state.load_status = load_status
            st.session_state.data_loaded = True
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"❌ Startup error: {e}")
            st.session_state.data_loaded = False


with st.sidebar:
    st.markdown("## 🛒 BI Agent")
    st.markdown("*Instacart · Claude (OpenRouter) · DuckDB*")
    st.markdown("---")

    st.markdown(
        """
<div style="font-family:'IBM Plex Mono';font-size:0.7rem;color:#4A6080;
background:#0A0F1E;border:1px solid #1E2440;border-radius:6px;padding:8px 12px;margin-bottom:12px;word-break:break-all;">
📂 data/ (auto-loaded)
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("⚡ Reload Data", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.initialized = False
        st.session_state.data_loaded = False
        st.session_state.load_status = {}
        st.rerun()

    if st.session_state.load_status:
        st.markdown("**Dataset Status:**")
        for table, info in st.session_state.load_status.items():
            short_name = table.replace("order_products_", "op_")
            if info.get("loaded"):
                rows = info.get("rows", 0)
                lazy = info.get("lazy", False)
                row_note = f"~{rows:,} rows (lazy)" if lazy else f"{rows:,} rows"
                st.markdown(
                    f'<span class="status-ok">✓ {short_name}</span> '
                    f'<span style="color:#4A5A7A;font-size:0.72rem;font-family:\'IBM Plex Mono\'">'
                    f"{row_note}</span>",
                    unsafe_allow_html=True,
                )
            else:
                err = info.get("error", "missing")
                st.markdown(
                    f'<span class="status-err">✗ {short_name}</span> '
                    f'<span style="color:#7A4A4A;font-size:0.72rem">{str(err)[:30]}</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("**Settings**")
    st.session_state.show_sql = st.toggle("Show generated SQL", value=True)

    if st.button("🔄 Clear chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.orchestrator:
            st.session_state.orchestrator.reset_context()
        st.rerun()

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        """
<div style="font-family:'IBM Plex Mono';font-size:0.72rem;color:#4A6080;line-height:1.8;">
Intent → Context → SQL → DuckDB → Chart
</div>
""",
        unsafe_allow_html=True,
    )


SUGGESTED = [
    "Top 10 most ordered products",
    "Busiest hours of the day",
    "Reorder rate by department",
    "Which day of week has most orders?",
    "Top 5 aisles by order count",
    "Average basket size per order",
    "Most reordered products in produce",
    "Departments by first-time purchase %",
]


def render_assistant_turn(msg: dict):
    """Content inside st.chat_message('assistant')."""
    st.markdown(msg["content"])

    badges = []
    intent = msg.get("intent", "")
    if intent:
        badges.append(f'<span class="badge badge-intent">{intent}</span>')
    elapsed = msg.get("elapsed_ms", 0)
    if elapsed:
        badges.append(f'<span class="badge badge-time">⏱ {elapsed}ms</span>')
    if msg.get("from_cache"):
        badges.append('<span class="badge badge-cache">cached</span>')
    if msg.get("is_followup"):
        badges.append('<span class="badge badge-followup">follow-up</span>')
    attempts = msg.get("attempts", 0)
    if attempts > 1:
        badges.append(f'<span class="badge badge-retry">{attempts} attempts</span>')
    if badges:
        st.markdown(f'<div class="meta-row">{"".join(badges)}</div>', unsafe_allow_html=True)

    if msg.get("merged_query"):
        st.caption(f"Interpreted as: _{msg['merged_query']}_")

    if msg.get("sql") and st.session_state.show_sql:
        with st.expander("Generated SQL", expanded=False):
            st.code(msg["sql"], language="sql")

    if msg.get("chart_fig") is not None:
        st.plotly_chart(
            msg["chart_fig"],
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
        )

    df = msg.get("df")
    if df is not None and not df.empty:
        _n = len(df)
        _label = "Result (1 row)" if _n == 1 else f"Data ({_n:,} rows)"
        with st.expander(
            _label,
            expanded=(msg.get("chart_type") == "table"),
        ):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(420, 56 + min(len(df), 12) * 35),
            )

    if msg.get("error"):
        st.error(f"SQL: {msg['error']}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="bi-header">
    <h1>🛒 Instacart BI Agent</h1>
    <p>Chat below · Claude + DuckDB · plain-English questions</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🛒"):
            render_assistant_turn(msg)

# ── Empty state + starter chips ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
<div class="welcome-wrap">
    <div class="emoji">💬</div>
    <div class="title">Ask anything about orders, products, aisles, or reorder behavior</div>
    <div class="sub">Press Enter to send · Your message clears from the box after sending</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("**Or try:**")
    c = st.columns(4)
    for i, suggestion in enumerate(SUGGESTED):
        with c[i % 4]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()

# ── Chat input (clears after each send — native Streamlit behavior) ─────────
prompt = st.chat_input("Ask about the Instacart data…")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    text = prompt.strip()
    if text:
        st.session_state.messages.append({"role": "user", "content": text})

        if not st.session_state.data_loaded or st.session_state.orchestrator is None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Data is not ready yet. Wait for startup or use **Reload Data** in the sidebar.",
                    "intent": "error",
                }
            )
        else:
            with st.spinner("Analyzing…"):
                try:
                    response = st.session_state.orchestrator.process(text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.user_message,
                            "intent": response.intent,
                            "sql": response.sql,
                            "df": response.df,
                            "chart_fig": response.chart_fig,
                            "chart_type": response.chart_type,
                            "is_followup": response.is_followup,
                            "merged_query": response.merged_query,
                            "attempts": response.attempts,
                            "from_cache": response.from_cache,
                            "elapsed_ms": response.elapsed_ms,
                            "error": response.error,
                        }
                    )
                except Exception as e:
                    logger.error(f"Orchestrator error: {e}", exc_info=True)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Something went wrong: {str(e)}",
                            "intent": "error",
                        }
                    )
        st.rerun()
