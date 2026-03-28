"""
Single LLM round-trip: context merge + intent classification.
Cuts latency vs separate ContextAnalyzer + IntentClassifier (saves one API call per question).
"""

import logging
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

COMBINED_SYSTEM_PROMPT = """You are the routing layer for an Instacart grocery BI chatbot (DuckDB). Answer in ONE JSON object (no markdown).

=== PART A: CONTEXT ===
If PREVIOUS QUERY is empty/null: set is_followup=false, merged_query = CURRENT QUERY exactly.

If PREVIOUS QUERY exists: decide if CURRENT is a follow-up. If yes, merged_query must be ONE self-contained question. Handle short replies after clarification ("department", "by aisle") by merging with PREVIOUS QUERY.

=== PART B: INTENT (apply to merged_query) ===
Classify merged_query:

"intent": one of "data_query", "greeting", "definition", "clarification_needed", "out_of_scope"

"tables_needed": subset of
  ["orders", "order_products_prior", "order_products_train", "products", "aisles", "departments"]

"chart_type": one of "bar", "line", "pie", "scatter", "table", "number", "auto"

"requires_multistep": true/false
"time_filter": string or null
"clarification_question": string or null — ONLY if intent is clarification_needed
"confidence": "high" or "low"
"reasoning": one short sentence

Intent rules:
- data_query: wants numbers, rankings, charts, trends from the dataset
- greeting: hi, hello, thanks
- definition: what does X mean
- clarification_needed: STILL ambiguous after merge — but NOT when user is answering a prior clarify with a short dimension (then data_query)
- out_of_scope: unrelated to grocery data

OUTPUT JSON SHAPE (all keys required):
{
  "is_followup": false,
  "merged_query": "...",
  "changes_made": "none or brief",
  "previous_sql_still_valid": false,
  "intent": "data_query",
  "tables_needed": ["orders"],
  "chart_type": "auto",
  "requires_multistep": false,
  "time_filter": null,
  "clarification_question": null,
  "confidence": "high",
  "reasoning": "..."
}
"""


class ConversationRouter:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def route(
        self,
        current_query: str,
        previous_query: str | None,
        previous_sql: str | None,
        previous_result_summary: str | None,
        conversation_history: list | None,
    ) -> dict:
        """Returns merged context + intent in one structure. On JSON failure, raises to trigger legacy path."""
        ctx = ""
        if conversation_history:
            recent = conversation_history[-8:]
            ctx = "\nRECENT CONVERSATION:\n" + "\n".join(
                f"{t['role'].upper()}: {t['content'][:400]}" for t in recent
            )

        pq = previous_query or ""
        parts = [ctx, f"PREVIOUS QUERY: {pq or '(none — first turn)'}", f"CURRENT QUERY: {current_query}"]
        if previous_sql and pq:
            parts.append(f"PREVIOUS SQL:\n{previous_sql[:1200]}")
        if previous_result_summary and pq:
            parts.append(f"PREVIOUS RESULT: {previous_result_summary}")

        user_msg = "\n\n".join(parts)

        result = self.llm.complete_json(
            system_prompt=COMBINED_SYSTEM_PROMPT,
            user_message=user_msg,
            temperature=0.0,
        )

        if result.get("error") == "json_parse_failed" or "intent" not in result:
            raise ValueError("router JSON parse failed or missing intent")

        merged = (result.get("merged_query") or current_query).strip()
        is_fu = bool(result.get("is_followup"))

        # Short-reply merge fallback (same as ContextAnalyzer)
        if (
            previous_query
            and not is_fu
            and merged == current_query.strip()
            and len(current_query.split()) <= 5
        ):
            merged = (
                f"{previous_query} The user wants this broken down by: {current_query.strip()}. "
                "Apply top 5 (or as requested) with a visualization if charts were mentioned."
            )
            result["is_followup"] = True
            result["merged_query"] = merged
            logger.info("Router: applied short-reply merge fallback")

        result["merged_query"] = merged
        result["is_followup"] = result.get("is_followup", False)

        if not result.get("intent"):
            result["intent"] = "data_query"
        if not result.get("tables_needed"):
            result["tables_needed"] = [
                "orders",
                "order_products_prior",
                "products",
                "aisles",
                "departments",
            ]
        if not result.get("chart_type"):
            result["chart_type"] = "auto"

        logger.info(
            f"Router: intent={result.get('intent')} followup={result.get('is_followup')} "
            f"| merged_len={len(merged)}"
        )
        return result
