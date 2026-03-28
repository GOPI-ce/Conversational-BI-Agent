"""
Data Loader — DuckDB Engine
Loads all 6 Instacart CSVs into an in-memory DuckDB database.
DuckDB is chosen over pandas for the 32M-row prior orders table:
  - columnar format handles aggregations without loading everything into RAM
  - native SQL join engine, no pandas merge overhead
  - lazy evaluation via views by default (fast startup)

Set BI_AGENT_MATERIALIZE_LARGE=true in .env to fully load large CSVs as TABLEs
(slower startup, more RAM; queries can be faster). Same SQL works either way.
"""

import duckdb
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Large fact tables: register as VIEW over CSV unless BI_AGENT_MATERIALIZE_LARGE=true.
# Row counts below are approximate for lazy views only.
LAZY_VIEW_TABLES = frozenset({"orders", "order_products_prior", "order_products_train"})
APPROX_ROW_COUNTS = {
    "orders": 3_421_084,
    "order_products_prior": 32_434_490,
    "order_products_train": 1_384_618,
}


def _materialize_large_tables() -> bool:
    """If True, load LAZY_VIEW_TABLES as full TABLEs (copy CSV into DuckDB)."""
    return os.getenv("BI_AGENT_MATERIALIZE_LARGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


# Schema definitions used for SQL generation context (mirrors EmbeddingManager from PDF)
TABLE_SCHEMAS = {
    "orders": {
        "description": "Core order table. One row per order.",
        "columns": {
            "order_id": "INTEGER — unique order identifier",
            "user_id": "INTEGER — customer identifier",
            "eval_set": "VARCHAR — 'prior' or 'train' or 'test'. Use 'prior' for history analysis.",
            "order_number": "INTEGER — sequence number of this order for the user (1 = first order)",
            "order_dow": "INTEGER — day of week (0=Saturday, 1=Sunday, 2=Monday...6=Friday)",
            "order_hour_of_day": "INTEGER — hour of day (0-23)",
            "days_since_prior_order": "FLOAT — days since user's previous order. NULL for first order.",
        },
        "row_count_approx": "3.4M rows",
        "key": "order_id",
    },
    "order_products_prior": {
        "description": "Products in prior (historical) orders. ~32M rows — the largest table.",
        "columns": {
            "order_id": "INTEGER — foreign key to orders.order_id",
            "product_id": "INTEGER — foreign key to products.product_id",
            "add_to_cart_order": "INTEGER — position when added to cart (1 = first item added)",
            "reordered": "INTEGER — 1 if product was reordered, 0 if first time purchase",
        },
        "row_count_approx": "32M rows",
        "key": "order_id, product_id",
    },
    "order_products_train": {
        "description": "Products in training orders (the most recent order per user in train eval_set).",
        "columns": {
            "order_id": "INTEGER — foreign key to orders.order_id",
            "product_id": "INTEGER — foreign key to products.product_id",
            "add_to_cart_order": "INTEGER — position when added to cart",
            "reordered": "INTEGER — 1 if reordered, 0 if new",
        },
        "row_count_approx": "1.4M rows",
        "key": "order_id, product_id",
    },
    "products": {
        "description": "Product catalog. Bridge between orders and aisles/departments.",
        "columns": {
            "product_id": "INTEGER — unique product identifier",
            "product_name": "VARCHAR — full product name",
            "aisle_id": "INTEGER — foreign key to aisles.aisle_id",
            "department_id": "INTEGER — foreign key to departments.department_id",
        },
        "row_count_approx": "50K rows",
        "key": "product_id",
    },
    "aisles": {
        "description": "Aisle lookup table. 134 aisles.",
        "columns": {
            "aisle_id": "INTEGER — unique aisle identifier",
            "aisle": "VARCHAR — aisle name (e.g., 'yogurt', 'fresh vegetables')",
        },
        "row_count_approx": "134 rows",
        "key": "aisle_id",
    },
    "departments": {
        "description": "Department lookup table. 21 departments.",
        "columns": {
            "department_id": "INTEGER — unique department identifier",
            "department": "VARCHAR — department name (e.g., 'produce', 'dairy eggs')",
        },
        "row_count_approx": "21 rows",
        "key": "department_id",
    },
}

# Join relationships — used to guide SQL generation
JOIN_HINTS = """
KEY JOIN RELATIONSHIPS:
- order_products_prior.order_id → orders.order_id
- order_products_train.order_id → orders.order_id
- order_products_prior.product_id → products.product_id
- order_products_train.product_id → products.product_id
- products.aisle_id → aisles.aisle_id
- products.department_id → departments.department_id

IMPORTANT RULES:
1. For historical/all-time analysis: use order_products_prior (32M rows, most complete)
2. For most-recent-order analysis: use order_products_train
3. For full product analysis combining both: UNION ALL order_products_prior and order_products_train
4. orders.days_since_prior_order is NULL for a user's very first order — always use COALESCE or filter
5. order_dow: 0=Saturday, 1=Sunday, 2=Monday, 3=Tuesday, 4=Wednesday, 5=Thursday, 6=Friday
6. Always LIMIT results to reasonable sizes (top 10, top 20) unless user asks for all
7. DuckDB syntax: use QUALIFY for window function filtering, use PERCENTILE_CONT for medians
"""


class DataLoader:
    """
    Loads CSVs into DuckDB and provides query execution interface.
    Single instance shared across agent lifetime (singleton pattern).
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.con = duckdb.connect(database=":memory:")
        self._loaded = False
        self._load_status = {}

    def load_all(self) -> dict:
        """
        Load all 6 CSVs. Returns status dict for each table.
        Uses DuckDB's native CSV reader — memory-efficient columnar storage.
        """
        csv_map = {
            "orders": "orders.csv",
            "order_products_prior": "order_products__prior.csv",
            "order_products_train": "order_products__train.csv",
            "products": "products.csv",
            "aisles": "aisles.csv",
            "departments": "departments.csv",
        }

        status = {}
        for table_name, filename in csv_map.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                # Try alternate naming (some kaggle downloads drop double underscore)
                alt = self.data_dir / filename.replace("__", "_")
                if alt.exists():
                    filepath = alt
                else:
                    status[table_name] = {
                        "loaded": False,
                        "error": f"File not found: {filename}",
                        "rows": 0,
                    }
                    logger.warning(f"Missing CSV: {filepath}")
                    continue

            try:
                path_sql = str(filepath).replace("\\", "/")
                use_lazy_view = table_name in LAZY_VIEW_TABLES and not _materialize_large_tables()
                if use_lazy_view:
                    self.con.execute(
                        f"""
                        CREATE OR REPLACE VIEW {table_name} AS
                        SELECT * FROM read_csv_auto('{path_sql}', header=True, null_padding=True)
                        """
                    )
                    row_count = APPROX_ROW_COUNTS.get(table_name, 0)
                    status[table_name] = {
                        "loaded": True,
                        "rows": row_count,
                        "file": str(filepath),
                        "lazy": True,
                    }
                    logger.info(
                        f"Registered {table_name} as lazy view (~{row_count:,} rows est.)"
                    )
                else:
                    self.con.execute(
                        f"""
                        CREATE OR REPLACE TABLE {table_name} AS
                        SELECT * FROM read_csv_auto('{path_sql}', header=True, null_padding=True)
                        """
                    )
                    row_count = self.con.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]
                    status[table_name] = {
                        "loaded": True,
                        "rows": row_count,
                        "file": str(filepath),
                        "lazy": False,
                    }
                    logger.info(f"Loaded {table_name}: {row_count:,} rows")
            except Exception as e:
                status[table_name] = {"loaded": False, "error": str(e), "rows": 0}
                logger.error(f"Failed to load {table_name}: {e}")

        self._load_status = status
        self._loaded = any(v["loaded"] for v in status.values())
        return status

    def execute(self, sql: str, limit: int = 10000):
        """
        Execute SQL and return result as pandas DataFrame.
        Safety limit applied to prevent memory blowout.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_all() first.")

        # Safety: wrap in subquery with limit if not already present
        sql_stripped = sql.strip().rstrip(";")
        if "LIMIT" not in sql_stripped.upper():
            sql_stripped = f"SELECT * FROM ({sql_stripped}) _q LIMIT {limit}"

        return self.con.execute(sql_stripped).df()

    def execute_raw(self, sql: str):
        """Execute without auto-limit (used for aggregations that are already bounded)."""
        return self.con.execute(sql.strip().rstrip(";")).df()

    def get_schema_context(self, tables: list[str] = None) -> str:
        """
        Returns schema description string for LLM context.
        Equivalent to EmbeddingManager.get_relevant_tables() in the Query Agent PDF.
        """
        if tables is None:
            tables = list(TABLE_SCHEMAS.keys())

        lines = ["DATABASE SCHEMA (DuckDB SQL):\n"]
        for t in tables:
            if t not in TABLE_SCHEMAS:
                continue
            schema = TABLE_SCHEMAS[t]
            lines.append(f"TABLE: {t}")
            lines.append(f"  Description: {schema['description']}")
            lines.append(f"  Size: {schema['row_count_approx']}")
            lines.append("  Columns:")
            for col, desc in schema["columns"].items():
                lines.append(f"    - {col}: {desc}")
            lines.append("")

        lines.append(JOIN_HINTS)
        return "\n".join(lines)

    @property
    def load_status(self):
        return self._load_status

    @property
    def is_loaded(self):
        return self._loaded
