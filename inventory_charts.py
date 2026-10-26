"""Stock-health visualisations for the Inventory Analysis tab.

Both charts take the SKU-level ``master`` frame produced by
``build_inventory_table`` (one row per SKU with ``Status``, ``On hand``,
``Units sold``, ``Est. stock value`` and the product attributes) and return a
single-view or layered Altair chart, which is what lets Streamlit stretch them
to the container width. Concatenated or faceted specs cannot be made
responsive, so side-by-side panels are done with ``st.columns`` instead.

Usage inside the tab::

    from inventory_charts import stock_health_breakdown, stock_health_by_group, FULL_WIDTH

    st.altair_chart(stock_health_breakdown(filtered), **FULL_WIDTH)
    st.altair_chart(stock_health_by_group(filtered, "Supplier"), **FULL_WIDTH)
"""
from __future__ import annotations

from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Shared vocabulary: one status order and one colour per status, used by every
# chart so a colour means the same thing everywhere on the dashboard.
# Colours are from the Okabe–Ito colour-blind-safe palette.
# -----------------------------------------------------------------------------
STATUS_ORDER = ["Healthy", "Stock-out risk", "Slow moving", "Dead stock", "Out of stock", "Inactive"]
STATUS_COLOURS = {
    "Healthy": "#009e73",
    "Stock-out risk": "#cc79a7",
    "Slow moving": "#e69f00",
    "Dead stock": "#d55e00",
    "Out of stock": "#0072b2",
    "Inactive": "#9aa0a6",
}
STATUS_RANK = {status: rank for rank, status in enumerate(STATUS_ORDER)}

Measure = Literal["units", "skus", "value"]
MEASURES = {
    # key: (aggregated column, axis title, D3 axis format, D3 label/tooltip format)
    "units": ("units", "Units on hand", "~s", ",.0f"),
    "skus": ("skus", "SKUs", "~s", ",.0f"),
    "value": ("value", "Stock value", "$~s", "$,.0f"),
}


def python_format(value: float, d3_format: str) -> str:
    """Format a number in Python the way D3 would with ``d3_format`` ('$,.0f' → '$1,234')."""
    prefix = "$" if d3_format.startswith("$") else ""
    return f"{prefix}{format(value, d3_format.lstrip('$'))}"


def _streamlit_version() -> tuple[int, ...]:
    parts = []
    for piece in st.__version__.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Streamlit 1.51+ replaced use_container_width=True with width="stretch".
FULL_WIDTH = {"width": "stretch"} if _streamlit_version() >= (1, 51) else {"use_container_width": True}


# -----------------------------------------------------------------------------
# Axis / scale helpers
# -----------------------------------------------------------------------------

def category_axis(**overrides) -> alt.Axis:
    """Axis for a nominal field: labels only, no grid, ticks or domain line."""
    settings = dict(grid=False, ticks=False, domain=False, labelPadding=6, labelLimit=200)
    settings.update(overrides)
    return alt.Axis(**settings)


def value_axis(fmt: str, **overrides) -> alt.Axis:
    """Axis for a quantitative field: light dashed grid, no domain line or ticks."""
    settings = dict(
        format=fmt, grid=True, gridOpacity=0.35, gridDash=[2, 3], tickCount=5,
        ticks=False, domain=False, titleFontWeight="normal", titlePadding=8,
    )
    settings.update(overrides)
    return alt.Axis(**settings)


def status_colour(statuses: list[str], legend: alt.Legend | None = None) -> alt.Color:
    """Colour encoding with the fixed status palette, restricted to the statuses present."""
    present = [status for status in STATUS_ORDER if status in statuses]
    return alt.Color(
        "Status:N",
        scale=alt.Scale(domain=present, range=[STATUS_COLOURS[status] for status in present]),
        sort=present,
        legend=legend,
        title=None,
    )


def top_legend() -> alt.Legend:
    return alt.Legend(title=None, orient="top", direction="horizontal", symbolType="square", padding=0, offset=6, labelFontSize=11)


def row_height(rows: int, per_row: int = 30, minimum: int = 120, extra: int = 30) -> int:
    """Chart height that grows with the number of bars so bars never become slivers."""
    return max(minimum, per_row * rows + extra)


# -----------------------------------------------------------------------------
# 1. Stock health breakdown
# -----------------------------------------------------------------------------

def _status_summary(master: pd.DataFrame) -> pd.DataFrame:
    summary = (
        master.groupby("Status")
        .agg(units=("On hand", "sum"), skus=("SKU", "nunique"), value=("Est. stock value", "sum"))
        .reindex([status for status in STATUS_ORDER if status in set(master["Status"])])
        .fillna(0.0)
        .reset_index()
    )
    total_units = summary["units"].sum()
    total_skus = summary["skus"].sum()
    summary["units_share"] = summary["units"] / total_units if total_units else 0.0
    summary["skus_share"] = summary["skus"] / total_skus if total_skus else 0.0
    return summary


def stock_health_breakdown(master: pd.DataFrame, measure: Measure = "units", bar_size: int = 22) -> alt.LayerChart:
    """Horizontal bars by status for one measure, labelled with that measure and the SKU count.

    ``measure`` selects the bar length: "units" (on hand), "skus" or "value".
    Tooltips always carry units, SKUs, stock value and shares.
    """
    summary = _status_summary(master)
    column, axis_title, axis_fmt, number_fmt = MEASURES[measure]
    other_column, _, _, other_fmt = MEASURES["skus" if measure != "skus" else "units"]
    other_word = "SKUs" if measure != "skus" else "units"

    summary["label"] = [
        f"{python_format(row[column], number_fmt)} · {python_format(row[other_column], other_fmt)} {other_word}"
        for _, row in summary.iterrows()
    ]
    statuses = summary["Status"].tolist()
    # Leave ~35% headroom on the x scale so the end-of-bar labels fit inside the plot.
    x_max = float(summary[column].max() or 1.0) * 1.35

    tooltip = [
        alt.Tooltip("Status:N"),
        alt.Tooltip("units:Q", title="Units on hand", format=",.0f"),
        alt.Tooltip("units_share:Q", title="Share of units", format=".0%"),
        alt.Tooltip("skus:Q", title="SKUs", format=",.0f"),
        alt.Tooltip("skus_share:Q", title="Share of SKUs", format=".0%"),
        alt.Tooltip("value:Q", title="Stock value", format="$,.0f"),
    ]

    bars = (
        alt.Chart(summary)
        .mark_bar(size=bar_size, cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Status:N", sort=statuses, title=None, axis=category_axis(labelFontSize=12)),
            x=alt.X(f"{column}:Q", title=axis_title, scale=alt.Scale(domain=[0, x_max], nice=False), axis=value_axis(axis_fmt)),
            color=status_colour(statuses, legend=None),
            tooltip=tooltip,
        )
    )
    # Text inherits the bar colour (theme-safe in light and dark mode).
    labels = bars.mark_text(align="left", baseline="middle", dx=6, fontSize=11, fontWeight="bold").encode(
        text="label:N"
    )
    return alt.layer(bars, labels).properties(height=row_height(len(summary))).configure_view(strokeWidth=0)


# -----------------------------------------------------------------------------
# 2. Stock health by product attribute (stacked)
# -----------------------------------------------------------------------------

def stock_health_by_group(
    master: pd.DataFrame,
    group_col: str = "Supplier",
    measure: Measure = "units",
    top_n: int = 15,
    normalize: bool = False,
    bar_size: int = 18,
) -> alt.LayerChart:
    """Stacked horizontal bars: one bar per ``group_col`` value, segmented by status.

    Groups are sorted by their total of ``measure`` (largest at the top) and
    limited to ``top_n``. ``normalize=True`` shows each group's status mix as a
    share of 100% instead of absolute values. Segments stack in STATUS_ORDER so
    the problem statuses always sit at the right-hand end of the bar.
    """
    column, axis_title, axis_fmt, number_fmt = MEASURES[measure]
    frame = master.copy()
    frame[group_col] = frame[group_col].fillna("Unknown").astype(str).replace({"": "Unknown", "nan": "Unknown"})

    agg = (
        frame.groupby([group_col, "Status"])
        .agg(units=("On hand", "sum"), skus=("SKU", "nunique"), value=("Est. stock value", "sum"))
        .reset_index()
    )
    agg = agg.loc[agg[column] > 0]
    totals = agg.groupby(group_col)[column].sum().sort_values(ascending=False)
    order = totals.index[:top_n].tolist()
    agg = agg.loc[agg[group_col].isin(order)].copy()
    agg["group_total"] = agg[group_col].map(totals)
    agg["share"] = agg[column] / agg["group_total"]
    agg["status_rank"] = agg["Status"].map(STATUS_RANK)
    statuses = agg["Status"].unique().tolist()

    if normalize:
        x = alt.X(f"{column}:Q", stack="normalize", title=f"Share of {axis_title.lower()}", axis=value_axis(".0%"))
    else:
        # Headroom on the x scale so the total labels never clip at the right edge.
        x_max = float(totals.iloc[0]) * 1.15 if len(totals) else 1.0
        x = alt.X(
            f"{column}:Q", stack="zero", title=axis_title,
            scale=alt.Scale(domain=[0, x_max], nice=False), axis=value_axis(axis_fmt),
        )

    bars = (
        alt.Chart(agg)
        .mark_bar(size=bar_size)
        .encode(
            y=alt.Y(f"{group_col}:N", sort=order, title=None, axis=category_axis(labelFontSize=11)),
            x=x,
            color=status_colour(statuses, legend=top_legend()),
            order=alt.Order("status_rank:Q"),
            tooltip=[
                alt.Tooltip(f"{group_col}:N"),
                alt.Tooltip("Status:N"),
                alt.Tooltip("units:Q", title="Units on hand", format=",.0f"),
                alt.Tooltip("skus:Q", title="SKUs", format=",.0f"),
                alt.Tooltip("value:Q", title="Stock value", format="$,.0f"),
                alt.Tooltip("share:Q", title=f"Share of group {axis_title.lower()}", format=".0%"),
                alt.Tooltip("group_total:Q", title=f"Group {axis_title.lower()}", format=number_fmt),
            ],
        )
    )
    layers = [bars]
    if not normalize:
        # Total at the end of each bar; aggregated from the same frame so it needs no extra data.
        totals_text = (
            alt.Chart(agg)
            .mark_text(align="left", baseline="middle", dx=5, fontSize=11, color="#6b7280")
            .encode(
                y=alt.Y(f"{group_col}:N", sort=order),
                x=alt.X(f"sum({column}):Q"),
                text=alt.Text(f"sum({column}):Q", format=number_fmt),
            )
        )
        layers.append(totals_text)
    chart = alt.layer(*layers).properties(height=row_height(len(order), per_row=26, extra=44))
    if not normalize:
        chart = chart.resolve_scale(x="shared")
    return chart.configure_view(strokeWidth=0)
