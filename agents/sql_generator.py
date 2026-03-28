"""
Agent 3: Natural Language → SQL Generator
Mirrors the 'SQL Service' from the Query Agent PDF.

Responsibilities:
- Generate DuckDB-compatible SQL from natural language
- Use schema context (like EmbeddingManager from PDF)
- Learn from previous failures (like FeedbackRetriever from PDF)
- Handle multi-step queries with CTEs
- Self-correct on execution failure (retry with error context)
"""

import logging
import re
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are Agent 3 — the SQL Generator for a BI chatbot on the Instacart grocery dataset stored in DuckDB.

Your ONLY job: convert the user's natural language question into a single, correct, executable DuckDB SQL query.

OUTPUT RULES:
- Return ONLY the SQL query. No explanation. No markdown. No ```sql fences. Just pure SQL.
- Always end with a semicolon.
- Always add LIMIT (default 20, unless user asks for more or query is an aggregation with few rows).
- Use CTEs (WITH clauses) for multi-step queries — they are cleaner and easier to debug.
- Column aliases must be human-readable (e.g., "total_orders" not "count(*)").

DUCKDB-SPECIFIC SYNTAX:
- Window function filtering: use QUALIFY instead of wrapping in subquery
  e.g.: SELECT *, ROW_NUMBER() OVER (...) rn FROM t QUALIFY rn <= 10
- Median: PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
- String formatting: PRINTF('%,.0f', num) for thousands separators
- Safe division: TRY_DIVIDE(a, b) or CASE WHEN b = 0 THEN 0 ELSE a/b END
- ROUND(value, 2) for decimal places

{schema_context}

QUERY PATTERNS TO KNOW:
1. Top N products: JOIN order_products_prior + products + aisles/departments, GROUP BY, ORDER BY COUNT DESC, LIMIT N
2. Reorder rate: AVG(reordered) or SUM(reordered)/COUNT(*) from order_products_prior
3. Busiest hours/days: GROUP BY order_hour_of_day or order_dow from orders
4. Department/aisle breakdown: 3-table join: order_products_prior → products → departments/aisles
5. Purchase frequency: AVG(days_since_prior_order) from orders WHERE days_since_prior_order IS NOT NULL
6. First-time vs repeat: WHERE reordered = 0 or reordered = 1
7. Order size: COUNT(product_id) per order_id from order_products_prior
8. User behavior: aggregate at user_id level via orders JOIN order_products_prior

AVOID:
- SELECT * on large tables (order_products_prior has 32M rows)
- Cartesian joins (always use ON conditions)
- Missing NULL handling for days_since_prior_order
- Subqueries in FROM when a CTE would be cleaner

PERFORMANCE (critical — keep execution fast on multi-million-row tables):
- order_products_prior is ~32M rows. NEVER self-join the full table for pairs / "bought together" / baskets without a bounded subset first.
- DuckDB CTEs with USING SAMPLE must be **MATERIALIZED** if the same sampled rows are self-joined or read twice. Otherwise SAMPLE re-evaluates and joins return **zero rows**. Use:
  `WITH sampled_orders AS MATERIALIZED (SELECT order_id FROM orders USING SAMPLE 0.5 PERCENT),`
  `sampled_products AS MATERIALIZED (SELECT op.order_id, op.product_id FROM order_products_prior op INNER JOIN sampled_orders so ON op.order_id = so.order_id)`
  Then self-join sampled_products a/b with a.order_id = b.order_id AND a.product_id < b.product_id, GROUP BY, LIMIT 20.
- For pairs & associations: sample orders first (MATERIALIZED), join line items, then pair-join — never skip MATERIALIZED on sampled CTEs used in self-joins.
- Prefer GROUP BY + LIMIT 20; avoid exploding intermediate results.
"""

SQL_RETRY_SYSTEM_PROMPT = """You are Agent 3 — the SQL Generator for Instacart BI chatbot using DuckDB.

The previous SQL query FAILED with an error. Your job is to fix it.

PREVIOUS FAILED SQL:
{failed_sql}

ERROR MESSAGE:
{error_message}

ORIGINAL USER QUESTION:
{original_question}

{schema_context}

Analyze the error carefully and generate a corrected DuckDB SQL query.
OUTPUT: Only the corrected SQL. No explanation. No markdown fences.
"""


class SQLGeneratorAgent:
    """
    Agent 3: NL → SQL with self-correction.
    Equivalent to SQL Service in the Query Agent PDF.
    """

    def __init__(self, llm: LLMClient, data_loader):
        self.llm = llm
        self.data_loader = data_loader
        self._query_history = []  # Simple feedback store (mirrors FeedbackRetriever)

    def generate(self, query: str, tables_hint: list = None, max_retries: int = 2) -> dict:
        """
        Generate SQL for the user query. Retries on failure with error context.
        Returns: {"sql": str, "success": bool, "error": str|None, "attempts": int}
        """
        # Check feedback store first (exact match — mirrors FeedbackRetriever)
        cached = self._check_feedback_store(query)
        if cached:
            logger.info("Using cached SQL from feedback store")
            return {"sql": cached, "success": True, "error": None, "attempts": 0, "from_cache": True}

        # Get schema context for relevant tables
        schema_context = self.data_loader.get_schema_context(tables_hint)

        # First attempt
        sql = self._generate_sql(query, schema_context)
        result = self._try_execute(sql)

        if result["success"]:
            self._store_success(query, sql)
            return {**result, "sql": sql, "attempts": 1, "from_cache": False}

        # Retry with error context (Self-correction loop from PDF enhancement #3)
        for attempt in range(2, max_retries + 2):
            logger.info(f"SQL retry attempt {attempt}: {result['error']}")
            sql = self._retry_with_error(
                failed_sql=sql,
                error_message=result["error"],
                original_question=query,
                schema_context=schema_context,
            )
            result = self._try_execute(sql)
            if result["success"]:
                self._store_success(query, sql)
                return {**result, "sql": sql, "attempts": attempt, "from_cache": False}

        return {
            "sql": sql,
            "success": False,
            "error": result["error"],
            "df": None,
            "attempts": max_retries + 1,
            "from_cache": False,
        }

    def _generate_sql(self, query: str, schema_context: str) -> str:
        """First-pass SQL generation."""
        system = SQL_SYSTEM_PROMPT.format(schema_context=schema_context)
        raw = self.llm.complete(
            system_prompt=system,
            user_message=f"Generate DuckDB SQL for: {query}",
            temperature=0.05,
            max_tokens=1800,
        )
        return self._clean_sql(raw)

    def _retry_with_error(
        self, failed_sql: str, error_message: str, original_question: str, schema_context: str
    ) -> str:
        """Retry SQL generation with error feedback."""
        system = SQL_RETRY_SYSTEM_PROMPT.format(
            failed_sql=failed_sql,
            error_message=error_message,
            original_question=original_question,
            schema_context=schema_context,
        )
        raw = self.llm.complete(
            system_prompt=system,
            user_message="Generate the corrected SQL query.",
            temperature=0.1,
        )
        return self._clean_sql(raw)

    def _try_execute(self, sql: str) -> dict:
        """Try to execute SQL. Returns success flag and dataframe."""
        try:
            df = self.data_loader.execute_raw(sql)
            return {"success": True, "error": None, "df": df}
        except Exception as e:
            return {"success": False, "error": str(e), "df": None}

    def _clean_sql(self, raw: str) -> str:
        """Strip markdown fences and extra whitespace from LLM SQL output."""
        cleaned = re.sub(r"```(?:sql|SQL)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = cleaned.strip()
        # Remove any leading explanation text before SELECT/WITH
        match = re.search(r"(WITH|SELECT|INSERT|CREATE)\s", cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[match.start():]
        # Ensure semicolon
        if not cleaned.rstrip().endswith(";"):
            cleaned = cleaned.rstrip() + ";"
        return cleaned

    def _check_feedback_store(self, query: str) -> str | None:
        """
        Simple exact-match feedback store.
        Mirrors FeedbackRetriever from the Query Agent PDF.
        In production, this would use vector similarity search.
        """
        query_normalized = query.lower().strip()
        for item in self._query_history:
            if item["query"].lower().strip() == query_normalized:
                return item["sql"]
        return None

    def _store_success(self, query: str, sql: str):
        """Store successful query→SQL mapping."""
        self._query_history.append({"query": query, "sql": sql})
        # Keep last 50 successful queries
        if len(self._query_history) > 50:
            self._query_history = self._query_history[-50:]
