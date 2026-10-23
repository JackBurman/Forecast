from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import forecast_engine  # type: ignore

# App setup
st.set_page_config(page_title="Inventory & Purchasing System", layout="wide")


def _streamlit_version() -> tuple[int, ...]:
    parts: list[int] = []
    for piece in st.__version__.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Streamlit 1.51+ replaced use_container_width=True with width="stretch" and
# shows a deprecation banner for the old argument. Older versions only know
# the old argument. Pass **FULL_WIDTH to buttons, tables and charts.
FULL_WIDTH = (
    {"width": "stretch"}
    if _streamlit_version() >= (1, 51)
    else {"use_container_width": True}
)

MONTH_NUMBER_TO_NAME = dict(enumerate(forecast_engine.MONTHS, start=1))

# Data paths and cached loaders
base_dir = Path(__file__).resolve().parent
raw_data_path = base_dir / "Raw_data"
csv_path = raw_data_path / "options_data.csv"


@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"SKU": "string", "Stock Code": "string", "Model": "string"},
        low_memory=False,
    )


def to_number(series: pd.Series) -> pd.Series:
    """Coerce values like '1,234.50', '$1,234', '45%', '(12)' or blanks to floats.

    Anything that is not numeric (including repeated header rows that survive a
    multi-file merge) becomes NaN instead of raising.
    """
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[,$%\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(text, errors="coerce").astype(float)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a Date column that may mix Excel serials, real datetimes and text.

    Text dates are read day-first (23/10/2026), which is what Australian
    exports produce; ISO dates (2026-10-23) are unaffected.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    is_serial = series.map(
        lambda value: (pd.api.types.is_integer(value) or pd.api.types.is_float(value))
        and pd.notna(value)
    ).astype(bool)
    if is_serial.any():
        days = series[is_serial].astype(float).astype(int)
        result.loc[is_serial] = pd.to_datetime("1899-12-30") + pd.to_timedelta(
            days, unit="D"
        )

    is_text = ~is_serial & series.notna()
    if is_text.any():
        result.loc[is_text] = pd.to_datetime(
            series[is_text], dayfirst=True, errors="coerce", format="mixed"
        )

    return result


def normalize_sku(value):
    """Preserve numeric SKUs as integers and alphanumeric SKUs as text."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    integer_text = text[:-2] if text.endswith(".0") else text
    if integer_text.isdigit():
        return int(integer_text)
    return text


@st.cache_data
def clean_sales_data(df_sales: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
    """Apply the Power Query-style transformation to the sales sample file."""
    sales = df_sales.copy()

    # Clean and standardize column names
    sales.columns = [str(col).strip() for col in sales.columns]

    rename_map = {
        "Inv#": "Inv #",
        "Stock code": "Stock Code",
        "GP %": "GP%",
        "GP doller": "GP $",
    }
    sales = sales.rename(columns={k: v for k, v in rename_map.items() if k in sales.columns})

    expected_columns = [
        "Inv #",
        "Date",
        "Stock Code",
        "Description",
        "Qty",
        "Total",
        "Cost",
        "GP%",
        "GP $",
    ]
    if not {"Inv #", "Date", "Stock Code"}.issubset(sales.columns):
        data_columns = [col for col in sales.columns if col != "_SourceFile"]
        if len(data_columns) != len(expected_columns):
            raise ValueError(
                "Sales export columns were not recognised: "
                + ", ".join(map(str, sales.columns))
            )
        sales = sales.rename(columns=dict(zip(data_columns, expected_columns)))

    sales = sales.dropna(how="all").reset_index(drop=True)

    # Carry location labels down the sales export.
    sales["Location"] = pd.NA
    location_mask = (
        sales["Inv #"].astype(str).str.upper().str.startswith("LOCATION:", na=False)
    )
    sales.loc[location_mask, "Location"] = (
        sales.loc[location_mask, "Inv #"]
        .astype(str)
        .str.replace("LOCATION:", "", regex=False, case=False)
        .str.strip()
    )
    if "_SourceFile" in sales.columns:
        sales["Location"] = sales.groupby("_SourceFile", sort=False)["Location"].ffill()
    else:
        sales["Location"] = sales["Location"].ffill()

    sales["SKU"] = sales["Stock Code"].map(normalize_sku)

    grand_total_mask = sales["Date"].astype(str).str.upper().str.contains(
        "GRAND TOTAL", na=False
    )
    sales = sales.loc[sales["GP%"].notna() & ~grand_total_mask].copy()

    sales["Date"] = parse_dates(sales["Date"])
    sales["Stock Code"] = (
        sales["Stock Code"].astype(str).str.strip().str.replace("\r", "")
    )

    # Enrich sales rows with product metadata.
    options_lookup = df_options.copy()
    options_lookup["SKU"] = options_lookup["Stock Code"].map(normalize_sku)
    options_lookup = options_lookup[
        ["SKU", "Supplier", "Class", "Fourth Product Group"]
    ].drop_duplicates(subset=["SKU"])

    sales = sales.merge(options_lookup, on="SKU", how="left", suffixes=("", "_Options"))
    sales = sales.rename(
        columns={
            "Supplier": "OptionsData.Supplier",
            "Class": "OptionsData.Class",
            "Fourth Product Group": "OptionsData.Fourth Product Group",
        }
    )

    city_suffixes = ["Perth", "Sydney", "Brisbane", "Melbourne", "Adelaide", "Oxford St"]

    def location_text(value) -> str:
        return "" if pd.isna(value) else str(value).strip()

    def get_city(location: str) -> str:
        for city in city_suffixes:
            if location.endswith(city):
                return city
        return "Other"

    locations = [location_text(value) for value in sales["Location"]]
    cities = [get_city(location) for location in locations]
    sales["City"] = cities
    sales["Sales Channel"] = [
        location if city == "Other" else location.replace(city, "").strip()
        for location, city in zip(locations, cities)
    ]

    sales["Month"] = sales["Date"].dt.month_name()
    sales["Year"] = sales["Date"].dt.year

    # Convert every numeric column once, before any filtering relies on it.
    sales = sales.rename(columns={"Total": "Sale"})
    for column in ["Qty", "Sale", "Cost", "GP%", "GP $"]:
        if column in sales.columns:
            sales[column] = to_number(sales[column])
    sales["Inv #"] = pd.to_numeric(sales["Inv #"], errors="coerce")

    sales = sales.loc[sales["Qty"].gt(0) & sales["GP $"].gt(0)].copy()

    def sales_channel_net(value) -> str:
        text = str(value or "").upper()
        if "A&D" in text:
            return "Retail"
        if "ECOMMERCE SALES" in text:
            return "Retail"
        if "RETAIL" in text:
            return "Retail"
        if "WAREHOUSE" in text:
            return "Warehouse/Sale"
        if "CLEARANCE SALE" in text:
            return "Warehouse/Sale"
        if "LOAN STOCK" in text:
            return "Warehouse/Sale"
        if "WARRANTY" in text:
            return "Warranty"
        if "RELIVE" in text:
            return "Relive"
        if "CONTRACT" in text:
            return "Contract"
        return "Other"

    sales["Sales Channel Net"] = sales["Sales Channel"].map(sales_channel_net)

    sales = sales.drop_duplicates(subset=["Inv #", "SKU", "Sale"]).copy()

    final_columns = [
        "City",
        "Sales Channel",
        "Sales Channel Net",
        "Location",
        "Inv #",
        "Date",
        "Month",
        "Year",
        "SKU",
        "OptionsData.Supplier",
        "OptionsData.Class",
        "Description",
        "OptionsData.Fourth Product Group",
        "Qty",
        "Sale",
        "Cost",
        "GP%",
        "GP $",
    ]

    final_columns = [col for col in final_columns if col in sales.columns]
    sales = sales[final_columns]
    if "SKU" in sales.columns:
        sales["SKU"] = sales["SKU"].astype("string")
    sales = sales.sort_values("Date", ascending=False).reset_index(drop=True)
    return sales


def raw_folder_manifest(folder: Path) -> tuple[tuple[str, int, int], ...]:
    supported_extensions = {".csv", ".xls", ".xlsx"}
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    )
    return tuple(
        (
            str(path.relative_to(folder)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in files
    )


@st.cache_resource(show_spinner="Merging daily and historical files...")
def load_raw_folder_data(
    folder: str,
    manifest: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    sources, source_names = forecast_engine.read_raw_data_folder(folder)
    formatted_sales = clean_sales_data(sources.pop("Sales Raw"), sources["Options Data"])
    forecast_inputs = {
        "OptionsData": sources["Options Data"],
        "Current Inventory": sources["Current Inventory"],
        "Purchase Orders": sources["Purchase Orders"],
        "Sales Data": formatted_sales,
    }
    return forecast_engine.prepare_forecast_sheets(forecast_inputs), source_names


# ---------------------------------------------------------------------------
# Forecasting
#
# This-year forecast  : additive Holt-Winters (ETS) fitted to monthly history
#                       before 1 January, blended with a year-on-year trend
#                       projection (same month last year × last year's growth).
# Next-year forecast  : this year's actual month × this year's growth where a
#                       complete month exists, otherwise this year's forecast.
# Rolling 12 months   : the 12 months after the current month, summed.
# Historical wMAPE    : the same method back-tested as a weighted MAPE on
#                       the previous 12 complete months.
# ---------------------------------------------------------------------------

SEASON = 12
GROWTH_FLOOR, GROWTH_CAP = 0.5, 1.2
DEFAULT_GROWTH = 1.05  # used for next year when there is no prior-year base


def monthly_series(sales: pd.DataFrame, metric: str, before: pd.Timestamp | None = None) -> pd.Series:
    """Aggregate dated sales into a gap-free monthly series (PeriodIndex)."""
    if metric not in sales.columns or "Date" not in sales.columns:
        return pd.Series(dtype=float)
    dated = sales.dropna(subset=["Date"])
    if before is not None:
        dated = dated.loc[dated["Date"] < before]
    if dated.empty:
        return pd.Series(dtype=float)
    grouped = dated.groupby(dated["Date"].dt.to_period("M"))[metric]
    series = grouped.mean() if metric == "GP%" else grouped.sum()
    full_range = pd.period_range(series.index.min(), series.index.max(), freq="M")
    series = series.reindex(full_range)
    series = series.ffill().bfill() if metric == "GP%" else series.fillna(0.0)
    return series.astype(float)


def _holt_winters_run(y: list[float], alpha: float, beta: float, gamma: float, phi: float, horizon: int):
    """One pass of additive Holt-Winters with a damped trend; returns (forecast, SSE)."""
    n = len(y)
    level = sum(y[:SEASON]) / SEASON
    trend = (sum(y[SEASON : 2 * SEASON]) / SEASON - level) / SEASON
    seasonal = [value - level for value in y[:SEASON]]
    sse = 0.0
    for t in range(n):
        index = t % SEASON
        s = seasonal[index]
        predicted = level + phi * trend + s
        if t >= SEASON:  # skip the initialisation season
            error = y[t] - predicted
            sse += error * error
        new_level = alpha * (y[t] - s) + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        seasonal[index] = gamma * (y[t] - new_level) + (1 - gamma) * s
        level = new_level
    forecast = []
    damped = 0.0
    for h in range(1, horizon + 1):
        damped += phi**h
        forecast.append(level + damped * trend + seasonal[(n + h - 1) % SEASON])
    return forecast, sse


@st.cache_data(show_spinner=False)
def holt_winters(values: tuple[float, ...], horizon: int) -> tuple[float, ...]:
    """ETS forecast `horizon` months past the end of `values`.

    Smoothing parameters are chosen by grid search on one-step-ahead error.
    With fewer than two seasons of history it falls back to seasonal naive,
    and with less than one season to the mean.
    """
    y = [float(value) for value in values]
    n = len(y)
    if n == 0:
        return tuple([float("nan")] * horizon)
    if n < SEASON:
        mean = sum(y) / n
        return tuple([mean] * horizon)
    if n < 2 * SEASON:
        return tuple(y[n - SEASON + ((h - 1) % SEASON)] for h in range(1, horizon + 1))

    best_sse = None
    best_forecast = None
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for beta in (0.0, 0.05, 0.1, 0.2, 0.3):
            for gamma in (0.05, 0.1, 0.2, 0.3, 0.5):
                for phi in (0.9, 1.0):
                    forecast, sse = _holt_winters_run(y, alpha, beta, gamma, phi, horizon)
                    if best_sse is None or sse < best_sse:
                        best_sse, best_forecast = sse, forecast
    return tuple(best_forecast)


def clamp_growth(ratio: float) -> float:
    return min(max(ratio, GROWTH_FLOOR), GROWTH_CAP)


def yoy_growth(history: pd.Series) -> float | None:
    """Last 12 months over the 12 before, clamped; None without two full years."""
    if len(history) < 2 * SEASON:
        return None
    recent = float(history.iloc[-SEASON:].sum())
    prior = float(history.iloc[-2 * SEASON : -SEASON].sum())
    if recent <= 0 or prior <= 0:
        return None
    return clamp_growth(recent / prior)


def blended_forecast(
    history: pd.Series,
    start: pd.Period,
    horizon: int,
    yoy_weight: float,
    bias: float,
    clip_at_zero: bool,
) -> pd.Series:
    """ETS forecast from `start`, blended with the YoY trend projection.

    forecast[m] = (1 - w) × ETS[m] + w × (actual[m - 12 months] × growth),
    using the ETS value alone where no prior-year month exists.
    """
    periods = pd.period_range(start, periods=horizon, freq="M")
    history = history[history.index < start]
    if history.empty:
        return pd.Series(np.nan, index=periods, dtype=float)

    full_range = pd.period_range(history.index.min(), start - 1, freq="M")
    history = history.reindex(full_range)
    history = history.fillna(0.0) if clip_at_zero else history.ffill().bfill()

    ets_values = holt_winters(tuple(history.tolist()), horizon)
    forecast = pd.Series(ets_values, index=periods, dtype=float)

    growth = yoy_growth(history)
    if growth is not None and yoy_weight > 0:
        for period in periods:
            prior = period - SEASON
            if prior in history.index:
                yoy_value = history[prior] * growth
                forecast[period] = (1 - yoy_weight) * forecast[period] + yoy_weight * yoy_value

    forecast = forecast * bias
    if clip_at_zero:
        forecast = forecast.clip(lower=0.0)
    return forecast


def next_year_forecast(
    actual: pd.Series,
    this_year_forecast: pd.Series,
    this_year: int,
    last_complete: pd.Period,
    bias: float,
) -> pd.Series:
    """Next year by month: this year's complete actual × growth, else this year's forecast."""
    this_year_months = [
        period
        for period in pd.period_range(f"{this_year}-01", periods=SEASON, freq="M")
        if period <= last_complete and period in actual.index
    ]
    recent = sum(float(actual[period]) for period in this_year_months)
    prior = sum(
        float(actual[period - SEASON])
        for period in this_year_months
        if (period - SEASON) in actual.index
    )
    growth = clamp_growth(recent / prior) if recent > 0 and prior > 0 else DEFAULT_GROWTH

    values = {}
    for period in pd.period_range(f"{this_year + 1}-01", periods=SEASON, freq="M"):
        base_period = period - SEASON
        actual_value = (
            float(actual[base_period])
            if base_period <= last_complete and base_period in actual.index
            else np.nan
        )
        if pd.notna(actual_value) and actual_value > 0:
            values[period] = actual_value * growth * bias
        else:
            values[period] = (
                float(this_year_forecast[base_period])
                if base_period in this_year_forecast.index
                else np.nan
            )
    return pd.Series(values, dtype=float)


def rolling_wmape(
    actual: pd.Series, last_complete: pd.Period, yoy_weight: float, bias: float, clip_at_zero: bool
) -> tuple[float | None, pd.PeriodIndex]:
    """Back-test using Weighted MAPE (wMAPE) over the previous 12 complete months."""
    window = pd.period_range(last_complete - (SEASON - 1), periods=SEASON, freq="M")
    predicted = blended_forecast(actual, window[0], SEASON, yoy_weight, bias, clip_at_zero)
    observed = actual.reindex(window)
    valid = observed.notna() & observed.ne(0) & predicted.notna()
    if not valid.any():
        return None, window
    
    abs_errors = (observed[valid] - predicted[valid]).abs()
    total_actual = observed[valid].abs()
    
    if total_actual.sum() == 0:
        return 0.0, window
        
    return float(abs_errors.sum() / total_actual.sum()), window


def build_forecasts(
    selected_sales: pd.DataFrame,
    metric: str,
    yoy_weight: float,
    bias: float,
    today: pd.Timestamp | None = None,
) -> dict:
    """Everything the Forecast tab needs, computed from the filtered sales."""
    today = (today or pd.Timestamp.today()).normalize()
    current_period = today.to_period("M")
    current_month_start = current_period.to_timestamp()
    last_complete = current_period - 1
    this_year = today.year
    clip_at_zero = metric != "GP%"

    actual = monthly_series(selected_sales, metric, before=current_month_start)

    this_start = pd.Period(f"{this_year}-01", freq="M")
    forecast_this = blended_forecast(actual, this_start, SEASON, yoy_weight, bias, clip_at_zero)
    forecast_next = next_year_forecast(actual, forecast_this, this_year, last_complete, bias)

    if metric == "GP%":
        forecast_this[:] = forecast_this.mean()
        forecast_next[:] = forecast_next.mean()

    rolling_window = pd.period_range(current_period + 1, periods=SEASON, freq="M")
    rolling = pd.concat([forecast_this, forecast_next]).reindex(rolling_window)
    if rolling.notna().any():
        rolling_total = float(rolling.mean() if metric == "GP%" else rolling.sum())
    else:
        rolling_total = None

    mape, mape_window = rolling_wmape(actual, last_complete, yoy_weight, bias, clip_at_zero)

    def as_frame(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {"Date": series.index.to_timestamp(), "Forecast": series.to_numpy(dtype=float)}
        )

    return {
        "actual": actual,
        "this_year": this_year,
        "forecast_this": as_frame(forecast_this),
        "forecast_next": as_frame(forecast_next),
        "rolling_window": rolling_window,
        "rolling_total": rolling_total,
        "mape": mape,
        "mape_window": mape_window,
    }


def window_label(window: pd.PeriodIndex) -> str:
    return f"{window[0].strftime('%b %Y')} – {window[-1].strftime('%b %Y')}"


# Top-right manual refresh button container
top_left, top_right = st.columns([8, 2])
with top_left:
    st.title("📦 Inventory & Purchasing System Migration")
    st.subheader("Options Data Integration")
with top_right:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data", **FULL_WIDTH):
        load_csv.clear()
        load_raw_folder_data.clear()
        st.rerun()

st.sidebar.markdown(
    "<div style='text-align: center; margin-top: -60px;'>"
    "<img src='https://livingedge.com.au/on/demandware.static/-/Library-Sites-shared_library/default/dw1e07d773/logo.svg' width='100'>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Source data refreshes automatically every day at 7:00 AM.")


@st.fragment(run_every="1h")
def automatic_source_refresh():
    current_time = pd.Timestamp.now()
    last_refresh_date = st.session_state.get("_last_auto_refresh_date")
    if current_time.hour >= 7 and (
        last_refresh_date is None or last_refresh_date != current_time.date()
    ):
        st.session_state["_last_auto_refresh_date"] = current_time.date()
        load_csv.clear()
        load_raw_folder_data.clear()
        st.rerun(scope="app")


automatic_source_refresh()


def filter_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Render Options Data filters and return the filtered dataframe."""
    filtered = df.copy()

    with st.sidebar.expander("Options Data Filters", expanded=True):
        st.caption("Leave a filter empty to include all values.")

        for index, column in enumerate(df.columns):
            key_base = f"options_filter_{index}"
            non_null = df[column].dropna()
            unique_values = sorted(
                {str(value).strip() for value in non_null if str(value).strip()},
                key=str.casefold,
            )

            if len(unique_values) <= 100:
                selected_values = st.multiselect(
                    column,
                    unique_values,
                    default=[],
                    key=f"{key_base}_select",
                )
                if selected_values:
                    filtered = filtered[
                        filtered[column].astype("string").str.strip().isin(selected_values)
                    ]
            else:
                search_value = st.text_input(
                    f"{column} contains",
                    key=f"{key_base}_text",
                ).strip()
                if search_value:
                    filtered = filtered[
                        filtered[column]
                        .astype("string")
                        .str.contains(search_value, case=False, na=False, regex=False)
                    ]

    return filtered


# Tab-style navigation.
selected_tab = st.radio(
    "View",
    ["Options Data", "Forecast", "Formatted Data"],
    horizontal=True,
    label_visibility="collapsed",
)

prepared_data = None
source_names = []

if selected_tab == "Options Data":
    try:
        df_options = load_csv(str(csv_path))
        st.success(f"Options data loaded successfully from {csv_path.name}!")

        df_options_filtered = filter_options_data(df_options)

        st.subheader("Options Data Table:")
        st.caption(
            f"Showing {min(1000, len(df_options_filtered)):,} of "
            f"{len(df_options_filtered):,} filtered rows "
            f"({len(df_options):,} total options)"
        )
        st.data_editor(
            df_options_filtered.head(1000),
            num_rows="dynamic",
            **FULL_WIDTH,
        )

        st.download_button(
            "Download full options data",
            df_options.to_csv(index=False).encode("utf-8"),
            file_name="options_data.csv",
            mime="text/csv",
        )

    except FileNotFoundError:
        st.warning(
            "⚠️ 'options_data.csv' not found. "
            "Please make sure your CSV file is in the Raw_data folder."
        )
    except Exception as e:
        st.error(f"Something went wrong while loading the data: {e}")

elif selected_tab == "Forecast":
    try:
        prepared_data, source_names = load_raw_folder_data(
            str(raw_data_path), raw_folder_manifest(raw_data_path)
        )
    except Exception as e:
        st.error(f"Could not load files from Raw_data: {e}")

    if prepared_data is None:
        st.info(
            "Add Inventory_report.xlsx, PODashboard.xlsx, options_data.csv, and the "
            "historical/weekly sales workbooks under Raw_data/Sales Data."
        )
    else:
        try:
            sheets = prepared_data
            sales_data = sheets["Sales Data"]
            source_times = [
                (raw_data_path / name).stat().st_mtime
                for name in source_names
                if (raw_data_path / name).exists()
            ]
            latest_source_label = (
                f"{pd.Timestamp(max(source_times), unit='s'):%d-%m-%Y %H:%M}"
                if source_times
                else "n/a"
            )
            planner_col, source_col = st.columns([2, 1])
            with planner_col:
                st.subheader("Forecast Planner")
            with source_col:
                st.caption(
                    f"Loaded {len(source_names)} source files · "
                    f"{len(sales_data):,} formatted sales rows · "
                    f"latest source: {latest_source_label}"
                )

            def choices_for(series: pd.Series) -> list[str]:
                values = sorted(
                    {str(value).strip() for value in series.dropna() if str(value).strip()},
                    key=str.casefold,
                )
                return ["TOTAL", *values]

            def filter_selectbox(
                label: str, series: pd.Series, default: str = "TOTAL"
            ) -> str:
                if series.nunique(dropna=True) > 1500:
                    starting_value = "" if default == "TOTAL" else default
                    typed_value = st.text_input(
                        f"{label} contains",
                        value=starting_value,
                        key=f"forecast_{label}",
                    ).strip()
                    return typed_value or "TOTAL"
                choices = choices_for(series)
                index = choices.index(default) if default in choices else 0
                return st.selectbox(label, choices, index=index, key=f"forecast_{label}")

            with st.sidebar:
                st.header("Forecast filters")
                st.subheader("Sales")
                selected_location = filter_selectbox("Location", sales_data["City"])
                selected_channel = filter_selectbox(
                    "Sales channel", sales_data["Sales Channel Net"]
                )

                st.subheader("Product")
                selected_classification = filter_selectbox(
                    "Classification", sales_data["OptionsData.Fourth Product Group"]
                )
                selected_supplier = filter_selectbox(
                    "Supplier", sales_data["Supplier Brand"]
                )
                selected_class = filter_selectbox("Class", sales_data["OptionsData.Class"])
                selected_model = filter_selectbox("Model", sales_data["Model"])
                selected_sku = filter_selectbox("SKU", sales_data["SKU Key"])

                st.subheader("Forecast settings")
                metric_options = ["Qty", "Sale", "Cost", "GP%", "GP $"]
                metric = st.selectbox("View", metric_options, index=0)
                history_from_year = st.number_input(
                    "History from Year",
                    min_value=2000,
                    max_value=2100,
                    value=2024,
                    step=1,
                    help="First year of actuals shown on the chart.",
                )
                forecast_from = pd.Timestamp(f"{history_from_year}-01-01")
                bias = st.number_input(
                    "Forecast bias",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.05,
                    help="Multiplier applied to every forecast month.",
                )
                yoy_weight = st.slider(
                    "YoY trend weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="0 = pure ETS, 1 = pure year-on-year trend projection.",
                )

            filters = {
                "Location": selected_location,
                "Sales Channel": selected_channel,
                "Classification": selected_classification,
                "Supplier": selected_supplier,
                "Class": selected_class,
                "Model": selected_model,
                "SKU": selected_sku,
            }
            selected_sales = forecast_engine.filter_sales(sales_data, filters).copy()
            if metric == "GP%" and not selected_sales.empty and "GP%" in selected_sales.columns:
                selected_sales["GP%"] = selected_sales["GP%"].mean()

            result = build_forecasts(selected_sales, metric, yoy_weight, bias)
            this_year = result["this_year"]
            forecast_this_year = result["forecast_this"]
            forecast_next_year = result["forecast_next"]
            rolling_window = result["rolling_window"]

            purchase_months = forecast_engine.monthly_purchase_orders(
                sheets["Purchase Orders"], filters
            )
            on_hand, free_stock = forecast_engine.inventory_totals(
                sheets["Current Inventory"], filters
            )

            if result["actual"].empty:
                st.warning("No completed months of dated sales match these selections.")
            else:
                # Open POs due by the end of the rolling window, including overdue ones.
                total_po = 0.0
                if not purchase_months.empty and {"Date", "Open PO Qty"} <= set(
                    purchase_months.columns
                ):
                    po_periods = pd.to_datetime(
                        purchase_months["Date"], errors="coerce"
                    ).dt.to_period("M")
                    po_by_period = purchase_months.groupby(po_periods)["Open PO Qty"].sum()
                    total_po = float(po_by_period[po_by_period.index <= rolling_window[-1]].sum())

                rolling_total = result["rolling_total"]
                mape = result["mape"]
                summary = st.columns(5)
                forecast_label_prefix = "$" if metric in ["Sale", "Cost", "GP $"] else ""
                if rolling_total is None:
                    rolling_display = "n/a"
                elif metric == "GP%":
                    rolling_display = f"{rolling_total:.1f}%"
                else:
                    rolling_display = f"{forecast_label_prefix}{rolling_total:,.0f}"
                summary[0].metric(
                    "Rolling 12-Month Forecast",
                    rolling_display,
                    help=f"{window_label(rolling_window)} · "
                    + ("average GP%" if metric == "GP%" else f"total {metric}"),
                )
                summary[1].metric("On hand · all sites", f"{on_hand:,.0f}")
                summary[2].metric("Free stock · all sites", f"{free_stock:,.0f}")
                summary[3].metric(
                    "Open PO quantity",
                    f"{total_po:,.0f}",
                    help=f"Open POs due by {rolling_window[-1].strftime('%b %Y')}, including overdue.",
                )
                summary[4].metric(
                    "Historical wMAPE",
                    f"{mape:.1%}" if mape is not None else "n/a",
                    help=f"Rolling previous 12 months (Weighted) · {window_label(result['mape_window'])}",
                )

                annual_profile = forecast_engine.yearly_monthly_profile(
                    selected_sales,
                    pd.DataFrame(columns=["Date", "Forecast"]),
                    metric,
                    forecast_from,
                )
                if metric == "GP%" and not annual_profile.empty:
                    numeric_cols = [c for c in annual_profile.columns if c != "Month"]
                    for col in numeric_cols:
                        annual_profile[col] = annual_profile[col].mean()

                # Year columns can come back as ints or strings; keep the
                # originals for lookups and use strings for everything drawn.
                year_columns = sorted(
                    (column for column in annual_profile.columns if column != "Month"),
                    key=lambda value: int(value),
                )
                year_labels = [str(column) for column in year_columns]

                st.subheader("Monthly comparison by year")
                actual_chart_data = annual_profile.melt(
                    id_vars="Month", var_name="Year", value_name="Value"
                ).dropna(subset=["Value"])
                actual_chart_data["Series"] = actual_chart_data["Year"].astype(str)
                actual_chart_data["Line Type"] = "Actual"

                forecast_frames = [
                    frame.dropna(subset=["Forecast"])
                    for frame in (forecast_this_year, forecast_next_year)
                    if not frame.empty
                ]
                forecast_frames = [frame for frame in forecast_frames if not frame.empty]
                if forecast_frames:
                    forecast_chart_data = pd.concat(forecast_frames, ignore_index=True)
                    forecast_chart_data["Year"] = (
                        forecast_chart_data["Date"].dt.year.astype(str)
                    )
                    forecast_chart_data["Month"] = (
                        forecast_chart_data["Date"].dt.month.map(MONTH_NUMBER_TO_NAME)
                    )
                else:
                    forecast_chart_data = pd.DataFrame(
                        columns=["Date", "Forecast", "Year", "Month"]
                    )
                forecast_plot_data = pd.DataFrame(
                    {
                        "Month": forecast_chart_data["Month"],
                        "Series": "Forecast " + forecast_chart_data["Year"].astype(str),
                        "Line Type": "Forecast",
                        "Value": forecast_chart_data["Forecast"],
                    }
                )
                chart_data = pd.concat(
                    [
                        actual_chart_data[["Month", "Series", "Line Type", "Value"]],
                        forecast_plot_data,
                    ],
                    ignore_index=True,
                )
                forecast_series = sorted(
                    forecast_plot_data["Series"].unique(),
                    key=lambda series: int(series.removeprefix("Forecast ")),
                )
                series_order = list(dict.fromkeys(year_labels + forecast_series))
                palette = [
                    "#0072b2",
                    "#e69f00",
                    "#009e73",
                    "#d55e00",
                    "#cc79a7",
                    "#56b4e9",
                    "#f0e442",
                    "#000000",
                ]
                series_colors = [
                    palette[index % len(palette)] for index in range(len(series_order))
                ]
                value_format = ",.0f" if metric != "GP%" else ".1f"
                legend_selection = alt.selection_point(
                    name="series_legend",
                    fields=["Series"],
                    bind="legend",
                    toggle="true",
                    clear=False,
                )
                yearly_chart = (
                    alt.Chart(chart_data)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Month:N", sort=forecast_engine.MONTHS, title="Month"),
                        y=alt.Y("Value:Q", title=metric),
                        color=alt.Color(
                            "Series:N",
                            sort=series_order,
                            scale=alt.Scale(domain=series_order, range=series_colors),
                            title="Year",
                        ),
                        strokeDash=alt.StrokeDash(
                            "Line Type:N",
                            scale=alt.Scale(
                                domain=["Actual", "Forecast"],
                                range=[[1, 0], [7, 4]],
                            ),
                            legend=None,
                        ),
                        detail=alt.Detail("Series:N"),
                        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0)),
                        tooltip=[
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Month:N", title="Month"),
                            alt.Tooltip("Value:Q", title=metric, format=value_format),
                        ],
                    )
                    .add_params(legend_selection)
                )

                month_hover_charts = []
                for month in forecast_engine.MONTHS:
                    rows = annual_profile.loc[annual_profile["Month"].eq(month)]
                    tooltip_row = {"Month": month}
                    tooltip_fields = []
                    for year in year_columns:
                        value = rows.iloc[0][year] if not rows.empty else pd.NA
                        if pd.notna(value):
                            field = f"year_{year}"
                            tooltip_row[field] = float(value)
                            tooltip_fields.append(
                                alt.Tooltip(f"{field}:Q", title=str(year), format=value_format)
                            )
                    month_forecasts = forecast_chart_data.loc[
                        forecast_chart_data["Month"].eq(month)
                    ]
                    for year in sorted(month_forecasts["Year"].unique()):
                        forecast_value = month_forecasts.loc[
                            month_forecasts["Year"].eq(year), "Forecast"
                        ]
                        if forecast_value.empty or pd.isna(forecast_value.iloc[0]):
                            continue
                        field = f"forecast_{year}"
                        tooltip_row[field] = float(forecast_value.iloc[0])
                        tooltip_fields.append(
                            alt.Tooltip(
                                f"{field}:Q", title=f"Forecast {year}", format=value_format
                            )
                        )
                    if tooltip_fields:
                        month_hover_selection = alt.selection_point(
                            name=f"hover_{month.lower()}",
                            fields=["Month"],
                            on="pointerover",
                            clear="pointerout",
                        )
                        month_hover_charts.append(
                            alt.Chart(pd.DataFrame([tooltip_row]))
                            .mark_rule(color="#667085", strokeWidth=12)
                            .encode(
                                x=alt.X("Month:N", sort=forecast_engine.MONTHS),
                                opacity=alt.condition(
                                    month_hover_selection,
                                    alt.value(0.18),
                                    alt.value(0),
                                    empty=False,
                                ),
                                tooltip=tooltip_fields,
                            )
                            .add_params(month_hover_selection)
                        )
                st.altair_chart(alt.layer(yearly_chart, *month_hover_charts), **FULL_WIDTH)
                st.caption(
                    f"Solid lines: actual by year · Dashed lines: forecast · "
                    f"Forecast {this_year} = ETS fitted to history before Jan {this_year}, "
                    f"blended with the year-on-year trend · "
                    f"Forecast {this_year + 1} = {this_year} actual × growth where the month is "
                    f"complete, otherwise the {this_year} forecast."
                )

                graph_table = (
                    chart_data.drop_duplicates(subset=["Series", "Month"], keep="last")
                    .pivot(index="Series", columns="Month", values="Value")
                    .reindex(index=series_order, columns=forecast_engine.MONTHS)
                    .reset_index()
                )
                graph_table.columns.name = None
                formatted_graph_table = graph_table.copy()
                for col in forecast_engine.MONTHS:
                    if col in formatted_graph_table.columns:
                        if metric in ["Sale", "Cost", "GP $"]:
                            formatted_graph_table[col] = formatted_graph_table[col].apply(
                                lambda x: f"${x:,.0f}" if pd.notna(x) else ""
                            )
                        elif metric == "GP%":
                            formatted_graph_table[col] = formatted_graph_table[col].apply(
                                lambda x: f"{x:.1f}%" if pd.notna(x) else ""
                            )
                        else:
                            formatted_graph_table[col] = formatted_graph_table[col].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
                            )

                st.dataframe(
                    formatted_graph_table,
                    hide_index=True,
                    height=220,
                    **FULL_WIDTH,
                )
        except Exception as e:
            st.error(f"Something went wrong while rendering the forecast: {e}")

elif selected_tab == "Formatted Data":
    try:
        prepared_data, source_names = load_raw_folder_data(
            str(raw_data_path), raw_folder_manifest(raw_data_path)
        )
    except Exception as e:
        st.error(f"Could not load files from Raw_data: {e}")

    if prepared_data is None:
        st.info("Add required source files under Raw_data to view formatted tables.")
    else:
        st.subheader("Formatted Data Tables")
        sheet_names = list(prepared_data.keys())
        selected_sheet = st.selectbox("Select table", sheet_names)
        df_sheet = prepared_data[selected_sheet]

        st.caption(f"Showing {min(1000, len(df_sheet)):,} of {len(df_sheet):,} rows")
        st.data_editor(
            df_sheet.head(1000),
            num_rows="dynamic",
            **FULL_WIDTH,
        )

        st.download_button(
            f"Download {selected_sheet} CSV",
            df_sheet.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_sheet.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
