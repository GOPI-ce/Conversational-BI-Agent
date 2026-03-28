"""
Query Agent Orchestrator
Central pipeline that mirrors the QueryAgentService from the PDF.

Flow (adapted for BI):
1. Receive user query
2. Context Analysis → merge with prior query if follow-up
3. Intent Classification → data_query / greeting / definition / etc.
4. SQL Generation → DuckDB SQL with schema context
5. Execution → DuckDB query on 6 CSVs
6. Visualization → auto-selected Plotly chart
7. Return structured response
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from utils.llm_client import LLMClient
from utils.data_loader import DataLoader
from utils.visualization import VisualizationService
from agents.intent_classifier import IntentClassifier
from agents.context_analyzer import ContextAnalyzer
from agents.conversation_router import ConversationRouter
from agents.sql_generator import SQLGeneratorAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response from the Query Orchestrator."""
    intent: str
    user_message: str                      # Display message for the user
    sql: Optional[str] = None             # Generated SQL (if data query)
    df: Optional[pd.DataFrame] = None     # Result dataframe
    chart_fig: object = None              # Plotly figure (or None)
    chart_type: str = "table"
    insight: str = ""
    is_followup: bool = False
    merged_query: str = ""
    attempts: int = 0
    from_cache: bool = False
    elapsed_ms: int = 0
    error: Optional[str] = None


class QueryOrchestrator:
    """
    Central pipeline — equivalent to QueryAgentService from the Query Agent PDF.
    Delegates to specialized agents: IntentClassifier, ContextAnalyzer, SQLGeneratorAgent.
    """

    def __init__(self, llm: LLMClient, data_loader: DataLoader):
        self.llm = llm
        self.data_loader = data_loader
        self.viz_service = VisualizationService()

        # Initialize sub-agents
        self.intent_classifier = IntentClassifier(llm)
        self.context_analyzer = ContextAnalyzer(llm)
        self.conversation_router = ConversationRouter(llm)
        self.sql_generator = SQLGeneratorAgent(llm, data_loader)

        # Session state — mirrors ChatHistoryManager + ContextManager from PDF
        self.conversation_history = []
        self.last_query = None
        self.last_sql = None
        self.last_result_summary = None

    def process(self, user_query: str) -> AgentResponse:
        """
        Main entry point. Runs the full agent pipeline.
        """
        t_start = time.time()
        logger.info(f"Processing query: {user_query}")

        # ── Step 1–2: Context + intent in ONE LLM call (fallback: two legacy calls) ─
        try:
            intent_result = self.conversation_router.route(
                current_query=user_query,
                previous_query=self.last_query,
                previous_sql=self.last_sql,
                previous_result_summary=self.last_result_summary,
                conversation_history=self.conversation_history,
            )
            effective_query = intent_result["merged_query"]
            is_followup = intent_result["is_followup"]
        except Exception as e:
            logger.warning(f"Combined router failed ({e}), using legacy context+intent")
            context_result = self.context_analyzer.analyze(
                current_query=user_query,
                previous_query=self.last_query,
                previous_sql=self.last_sql,
                previous_result_summary=self.last_result_summary,
            )
            effective_query = context_result["merged_query"]
            is_followup = context_result["is_followup"]
            intent_result = self.intent_classifier.classify(
                user_query=effective_query,
                conversation_history=self.conversation_history,
            )
        intent = intent_result.get("intent", "data_query")

        # ── Step 3: Route by Intent ───────────────────────────────────────────

        # Greeting
        if intent == "greeting":
            return self._respond(
                intent=intent,
                message="👋 Hello! I'm your Instacart BI Agent. Ask me anything about the grocery data — top products, reorder rates, busiest hours, department breakdowns, and more!",
                elapsed_ms=int((time.time() - t_start) * 1000),
            )

        # Definition / explanation
        if intent == "definition":
            explanation = self._explain(effective_query)
            return self._respond(
                intent=intent,
                message=explanation,
                elapsed_ms=int((time.time() - t_start) * 1000),
            )

        # Out of scope
        if intent == "out_of_scope":
            return self._respond(
                intent=intent,
                message="⚠️ That question doesn't seem related to the Instacart grocery dataset. I can answer questions about orders, products, aisles, departments, reorder behavior, and shopping patterns.",
                elapsed_ms=int((time.time() - t_start) * 1000),
            )

        # Clarification needed — still anchor thread so short replies ("department") merge next turn
        if intent == "clarification_needed":
            q = intent_result.get("clarification_question", "Could you please provide more details?")
            self.last_query = effective_query
            self._update_history(user_query, f"🤔 {q}")
            return self._respond(
                intent=intent,
                message=f"🤔 {q}",
                elapsed_ms=int((time.time() - t_start) * 1000),
            )

        # ── Step 4: Data Query → SQL Generation ───────────────────────────────
        tables_hint = intent_result.get("tables_needed", None)
        chart_type_hint = intent_result.get("chart_type", "auto")

        sql_result = self.sql_generator.generate(
            query=effective_query,
            tables_hint=tables_hint,
        )

        if not sql_result["success"]:
            return self._respond(
                intent=intent,
                message=f"❌ I couldn't generate a working SQL query after {sql_result['attempts']} attempts.\n\n**Error:** `{sql_result['error']}`\n\nTry rephrasing your question.",
                sql=sql_result["sql"],
                error=sql_result["error"],
                elapsed_ms=int((time.time() - t_start) * 1000),
            )

        df = sql_result["df"]

        # ── Step 5: Visualization ─────────────────────────────────────────────
        fig, chart_type, insight = self.viz_service.render(
            df=df,
            chart_type_hint=chart_type_hint,
            query=effective_query,
        )

        # ── Step 6: Update session state ──────────────────────────────────────
        self.last_query = effective_query
        self.last_sql = sql_result["sql"]
        self.last_result_summary = self._summarize_result(df)
        self._update_history(user_query, insight)

        return AgentResponse(
            intent=intent,
            user_message=insight,
            sql=sql_result["sql"],
            df=df,
            chart_fig=fig,
            chart_type=chart_type,
            insight=insight,
            is_followup=is_followup,
            merged_query=effective_query if is_followup else "",
            attempts=sql_result["attempts"],
            from_cache=sql_result.get("from_cache", False),
            elapsed_ms=int((time.time() - t_start) * 1000),
            error=None,
        )

    def _explain(self, query: str) -> str:
        """Use LLM to explain a term or concept from the dataset."""
        system = (
            "You are a data analyst explaining concepts from the Instacart grocery dataset. "
            "Give clear, concise explanations in 2-3 sentences. Reference the actual dataset columns where relevant."
        )
        return self.llm.complete(system, query, temperature=0.3)

    def _summarize_result(self, df: pd.DataFrame) -> str:
        """Create a brief text summary of the result for context carry-forward."""
        if df is None or df.empty:
            return "No results."
        return f"{len(df)} rows, columns: {', '.join(df.columns.tolist()[:5])}"

    def _update_history(self, query: str, response: str):
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response[:500]})
        # Keep last 10 turns
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _respond(self, intent: str, message: str, **kwargs) -> AgentResponse:
        return AgentResponse(intent=intent, user_message=message, **kwargs)

    def reset_context(self):
        """Clear conversation context — new session."""
        self.conversation_history = []
        self.last_query = None
        self.last_sql = None
        self.last_result_summary = None
