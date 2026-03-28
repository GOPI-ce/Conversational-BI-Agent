"""
Agent 1: Intent & Scope Classifier
Mirrors the 'AI Intent Layer' from the Query Agent PDF.

Responsibilities:
- Determine if query is BI-related or out of scope
- Classify query type: data_query, greeting, definition, clarification_needed
- Extract key entities: tables needed, chart type hint, time filters
- Flag if query needs multi-step reasoning
"""

import logging
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are the Intent Classifier for a Business Intelligence agent built on the Instacart grocery dataset.

Your job is to analyze the user's question and return a JSON object with:

{
  "intent": one of ["data_query", "greeting", "definition", "clarification_needed", "out_of_scope"],
  "tables_needed": list of table names needed — choose from:
    ["orders", "order_products_prior", "order_products_train", "products", "aisles", "departments"],
  "chart_type": suggested chart — one of ["bar", "line", "pie", "scatter", "table", "number", "auto"],
  "requires_multistep": true/false — true if query needs intermediate results or correlation analysis,
  "time_filter": any time-related filter mentioned (e.g., "weekdays only", "morning hours") or null,
  "clarification_question": if intent is clarification_needed, what to ask the user, else null,
  "confidence": "high" or "low",
  "reasoning": one sentence explaining your classification
}

DATASET CONTEXT:
- orders: 3.4M orders with user_id, day of week, hour of day, days_since_prior_order
- order_products_prior: 32M product line items from historical orders (reordered flag)
- order_products_train: 1.4M product line items from users' most recent orders
- products: 50K products with names, linked to aisles and departments
- aisles: 134 aisles (e.g., yogurt, fresh vegetables)
- departments: 21 departments (e.g., produce, dairy eggs, beverages)

CLASSIFICATION RULES:
- "data_query" → user wants data, charts, counts, rankings, trends from the dataset
- "greeting" → hello, hi, how are you, thanks
- "definition" → user asks what something means (reorder rate, aisle, etc.)
- "clarification_needed" → ONLY if the query is STILL ambiguous after considering RECENT CONVERSATION. If the assistant just asked for a missing detail (dimension, time range, etc.) and the user replies with a short answer ("department", "by aisle", "hour of day"), that completes the request — classify "data_query" (the Context Analyzer already merged the full question into CURRENT QUERY).
- "out_of_scope" → nothing to do with grocery/instacart data

For chart_type:
- Rankings/comparisons → "bar"
- Trends over time/sequences → "line"  
- Proportions/share → "pie"
- Distributions/correlations → "scatter"
- Single metric → "number"
- Raw data preview → "table"
- Let LLM decide → "auto"
"""


class IntentClassifier:
    """
    Agent 1: Classifies user intent and extracts query metadata.
    Equivalent to the AI Intent Layer in the Query Agent PDF.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(self, user_query: str, conversation_history: list = None) -> dict:
        """
        Classify the user's intent.
        Returns structured dict with intent, tables, chart type, etc.
        """
        context = ""
        if conversation_history:
            recent = conversation_history[-8:]  # enough for clarify → reply threads
            context = "\nRECENT CONVERSATION:\n" + "\n".join(
                [f"{t['role'].upper()}: {t['content'][:400]}" for t in recent]
            )

        user_msg = f"{context}\n\nCURRENT QUERY: {user_query}"

        result = self.llm.complete_json(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_message=user_msg,
        )

        # Fallback if JSON parse failed
        if "error" in result:
            logger.warning("Intent classification JSON parse failed, using fallback")
            return {
                "intent": "data_query",
                "tables_needed": [
                    "orders",
                    "order_products_prior",
                    "products",
                    "aisles",
                    "departments",
                ],
                "chart_type": "auto",
                "requires_multistep": False,
                "time_filter": None,
                "clarification_question": None,
                "confidence": "low",
                "reasoning": "Fallback classification — JSON parse failed",
            }

        logger.info(f"Intent: {result.get('intent')} | Tables: {result.get('tables_needed')}")
        return result
