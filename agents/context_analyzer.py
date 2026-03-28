"""
Agent 2: Contextual Query Builder / Refiner
Mirrors the 'Context Analyzer' from the Query Agent PDF.

Responsibilities:
- Detect if current query is a follow-up to the previous one
- Merge/enrich the query with context from prior conversation
- Resolve pronouns and implicit references ("that", "those", "the same")
- Handle filter additions: "now filter that to organic products only"
"""

import logging
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM_PROMPT = """You are the Context Analyzer for a BI chatbot on Instacart grocery data.

Your job: determine if the current query is a follow-up to the previous query, and if so, 
produce a MERGED, self-contained query that captures full intent.

Return JSON:
{
  "is_followup": true/false,
  "merged_query": "The complete standalone question combining context and current query",
  "changes_made": "Brief description of what was merged/added, or 'none' if not a followup",
  "previous_sql_still_valid": true/false  
}

EXAMPLES:
Previous: "Show me top 10 products by order count"
Current: "Now filter that to only dairy products"
→ is_followup: true
→ merged_query: "Show me top 10 dairy products by order count"
→ previous_sql_still_valid: false

Previous: "What are the busiest hours of the day for orders?"
Current: "What about weekends only?"  
→ is_followup: true
→ merged_query: "What are the busiest hours of the day for orders on weekends (Saturday and Sunday)?"
→ previous_sql_still_valid: false

Previous: "Top 5 departments by total orders"
Current: "What is the reorder rate?"
→ is_followup: false (completely different topic)
→ merged_query: "What is the reorder rate?" (unchanged)
→ previous_sql_still_valid: false

CLARIFICATION FOLLOW-UPS (assistant asked for missing detail; user replies with a short answer):
Previous: "Give me top 5 order count and a visualization chart"
(Assistant had asked what dimension: by user, day, hour, department, aisle?)
Current: "department"
→ is_followup: true
→ merged_query: "Show top 5 departments by order count with a visualization chart"

Previous: "Top products by sales"
(Assistant asked: which time period?)
Current: "last month only"
→ is_followup: true
→ merged_query: "Top products by sales in the last month only"

Previous: "Show reorder rate"
Current: "by aisle"
→ is_followup: true
→ merged_query: "Show reorder rate broken down by aisle"

RULES:
- Pronouns like "that", "those", "it", "same", "also", "additionally" → likely followup
- Filter additions (only, just, excluding, filter to) → followup
- Short answers (one to a few words) right after a vague or underspecified question — almost always a followup: merge them into ONE self-contained data question
- Dimension names alone ("department", "aisle", "hour", "user", "day of week") after a ranking/top-N request → followup: interpret as "by that dimension"
- Completely different topics → NOT a followup
- If not a followup, merged_query = original current query unchanged
"""


class ContextAnalyzer:
    """
    Agent 2: Contextual Query Refiner.
    Equivalent to the Context Analyzer in the Query Agent PDF.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        current_query: str,
        previous_query: str = None,
        previous_sql: str = None,
        previous_result_summary: str = None,
    ) -> dict:
        """
        Analyze if current query is a follow-up and produce merged query.
        """
        if not previous_query:
            # First query — nothing to merge
            return {
                "is_followup": False,
                "merged_query": current_query,
                "changes_made": "none",
                "previous_sql_still_valid": False,
            }

        context_parts = [f"PREVIOUS QUERY: {previous_query}"]
        if previous_sql:
            context_parts.append(f"PREVIOUS SQL:\n{previous_sql}")
        if previous_result_summary:
            context_parts.append(f"PREVIOUS RESULT: {previous_result_summary}")
        context_parts.append(f"\nCURRENT QUERY: {current_query}")

        user_msg = "\n".join(context_parts)

        result = self.llm.complete_json(
            system_prompt=CONTEXT_SYSTEM_PROMPT,
            user_message=user_msg,
        )

        if "error" in result:
            logger.warning("Context analysis JSON parse failed, treating as new query")
            return {
                "is_followup": False,
                "merged_query": current_query,
                "changes_made": "none",
                "previous_sql_still_valid": False,
            }

        merged = (result.get("merged_query") or current_query).strip()
        is_fu = bool(result.get("is_followup"))

        # Fallback: model sometimes misses short answers ("department") after a clarify turn
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
            logger.info("Context: applied short-reply merge fallback")

        logger.info(
            f"Context: is_followup={result.get('is_followup')} | "
            f"changes={result.get('changes_made', 'none')}"
        )
        return result
