"""
Visualization Service
Enhancement #1 from the Query Agent PDF: "Adding Graphical Images to the Tables"

Analyzes query result structure and auto-selects the best chart type.
Uses Plotly for interactive charts in Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def _scalar_to_float(val) -> float:
    """DuckDB/pandas often return numpy scalars; Plotly needs a Python float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _format_scalar(val) -> str:
    """Human-readable number/string for one cell (handles numpy int64/float64)."""
    try:
        f = float(val)
        if f.is_integer() or abs(f - round(f)) < 1e-9:
            return f"{int(round(f)):,}"
        return f"{f:,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(val)

# Color palette — modern data viz aesthetic
COLORS = [
    "#2D6BE4", "#E84C4C", "#27AE60", "#F39C12", "#8E44AD",
    "#16A085", "#D35400", "#2980B9", "#C0392B", "#1ABC9C",
]

BACKGROUND_COLOR = "#0F1117"
PAPER_COLOR = "#1A1D27"
FONT_COLOR = "#E8EAF0"
GRID_COLOR = "#2A2D3A"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BACKGROUND_COLOR,
    font=dict(color=FONT_COLOR, family="'IBM Plex Mono', monospace", size=12),
    title_font=dict(size=16, color=FONT_COLOR),
    legend=dict(bgcolor=PAPER_COLOR, bordercolor=GRID_COLOR, borderwidth=1),
    margin=dict(l=40, r=40, t=60, b=40),
    colorway=COLORS,
)


class VisualizationService:
    """
    Converts a pandas DataFrame + query metadata into the best Plotly chart.
    Implements Enhancement #1 from the Query Agent PDF architecture.
    """

    def render(
        self,
        df: pd.DataFrame,
        chart_type_hint: str = "auto",
        query: str = "",
        title: str = "",
    ) -> tuple:
        """
        Returns (fig, chart_type_used, insight_text).
        fig is None if only a table should be shown.
        """
        if df is None or df.empty:
            return None, "table", "No data returned."

        rows, cols = df.shape

        # Determine actual chart type
        chart_type = self._decide_chart_type(df, chart_type_hint, query)
        logger.info(f"Chart type: {chart_type} (hint was: {chart_type_hint})")

        fig = None
        if chart_type == "bar":
            fig = self._bar_chart(df, title or query)
        elif chart_type == "line":
            fig = self._line_chart(df, title or query)
        elif chart_type == "pie":
            fig = self._pie_chart(df, title or query)
        elif chart_type == "scatter":
            fig = self._scatter_chart(df, title or query)
        elif chart_type == "number":
            fig = self._single_number(df, title or query)
        # "table" → return None fig, show as st.dataframe

        insight = self._generate_insight(df, chart_type)
        return fig, chart_type, insight

    def _decide_chart_type(self, df: pd.DataFrame, hint: str, query: str) -> str:
        """
        Auto-detect best chart type from data shape and query keywords.
        """
        if hint not in ("auto", None, ""):
            return hint

        rows, cols = df.shape
        query_lower = query.lower()

        # Single value → number card
        if rows == 1 and cols == 1:
            return "number"

        # Time series indicators in column names
        time_cols = {"hour", "day", "week", "month", "dow", "hour_of_day", "order_number"}
        col_names = {c.lower() for c in df.columns}
        if time_cols & col_names:
            return "line"

        # Distribution / proportion keywords
        if any(kw in query_lower for kw in ["proportion", "share", "percentage", "breakdown", "distribution", "%"]):
            if rows <= 10:
                return "pie"

        # Correlation / two numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) >= 2 and cols <= 3 and rows > 5:
            if any(kw in query_lower for kw in ["correlat", "vs", "versus", "compare", "relation"]):
                return "scatter"

        # Ranking / top N
        if any(kw in query_lower for kw in ["top", "most", "least", "rank", "best", "worst", "highest", "lowest"]):
            return "bar"

        # Small result → bar; large → table
        if rows <= 30 and cols <= 5:
            return "bar"

        return "table"

    def _bar_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not numeric_cols:
            return None

        x_col = cat_cols[0] if cat_cols else cols[0]
        y_col = numeric_cols[0]

        # Truncate long labels
        df = df.copy()
        if df[x_col].dtype == object:
            df[x_col] = df[x_col].astype(str).str[:30]

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color_discrete_sequence=COLORS,
        )
        fig.update_traces(
            marker_line_width=0,
            opacity=0.9,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(
            gridcolor=GRID_COLOR,
            tickangle=-35 if len(df) > 8 else 0,
        )
        fig.update_yaxes(gridcolor=GRID_COLOR)
        return fig

    def _line_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 1:
            return self._bar_chart(df, title)

        x_col = cols[0]
        y_cols = numeric_cols if numeric_cols[0] != x_col else numeric_cols[1:]

        if not y_cols:
            return self._bar_chart(df, title)

        fig = go.Figure()
        for i, y_col in enumerate(y_cols[:4]):  # max 4 lines
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    name=y_col.replace("_", " ").title(),
                    mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                    marker=dict(size=6),
                )
            )
        fig.update_layout(title=title, **PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        return fig

    def _pie_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not numeric_cols or not cat_cols:
            return self._bar_chart(df, title)

        labels = df[cat_cols[0]].astype(str).str[:25]
        values = df[numeric_cols[0]]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,  # donut style
                marker=dict(colors=COLORS, line=dict(color=BACKGROUND_COLOR, width=2)),
                textfont=dict(color=FONT_COLOR),
            )
        )
        fig.update_layout(title=title, **PLOTLY_LAYOUT)
        return fig

    def _scatter_chart(self, df: pd.DataFrame, title: str) -> go.Figure:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            return self._bar_chart(df, title)

        x_col, y_col = numeric_cols[0], numeric_cols[1]
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        color_col = cat_cols[0] if cat_cols else None

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            color_discrete_sequence=COLORS,
            opacity=0.75,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        return fig

    def _single_number(self, df: pd.DataFrame, title: str) -> go.Figure:
        val = df.iloc[0, 0]
        col_name = df.columns[0].replace("_", " ").title()
        num = _scalar_to_float(val)

        fig = go.Figure(
            go.Indicator(
                mode="number",
                value=num,
                title={"text": col_name, "font": {"size": 18, "color": FONT_COLOR}},
                number={
                    "font": {"size": 56, "color": COLORS[0]},
                    "valueformat": ",.0f" if abs(num - round(num)) < 1e-9 else ",.4f",
                },
            )
        )
        fig.update_layout(
            paper_bgcolor=PAPER_COLOR,
            font=dict(color=FONT_COLOR),
            margin=dict(l=20, r=20, t=60, b=20),
            height=200,
        )
        return fig

    def _generate_insight(self, df: pd.DataFrame, chart_type: str) -> str:
        """Generate a quick textual insight from the result."""
        if df is None or df.empty:
            return "No data returned."

        rows, cols = df.shape
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # Single-row results (COUNT, SUM, etc.): show the actual values — not "1 rows" (misread as table size)
        if rows == 1:
            parts = []
            for col in df.columns:
                v = df.iloc[0][col]
                label = str(col).replace("_", " ")
                parts.append(f"**{label}:** `{_format_scalar(v)}`")
            parts.append(
                "*(One result row: a single aggregate. The numbers above are the metric values.)*"
            )
            if chart_type == "table":
                parts.append("Showing as table — too many rows/columns for a chart.")
            return " ".join(parts)

        parts = [f"**{rows:,}** rows in this query result."]

        if numeric_cols and rows > 1:
            col = numeric_cols[0]
            col_label = col.replace("_", " ")
            top_val = df[col].max()
            parts.append(f"Highest **{col_label}:** `{_format_scalar(top_val)}`")

        if chart_type == "table":
            parts.append("Showing as table — too many rows/columns for a chart.")

        return " · ".join(parts)
