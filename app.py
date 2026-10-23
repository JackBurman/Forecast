from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import forecast_engine  # type: ignore

# App setup
st.set_page_config(page_title="Inventory & Purchasing System", layout="wide")

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


@st.cache_data
def clean_sales_data(df_sales: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
    """Apply the Power Query-style transformation to the sales sample file."""
    sales = df_sales.copy()

    # Clean and standardize column names
    sales.columns = [str(col).strip() for col in sales.columns]

    rename_map = {
        "Inv #": "Inv #",
        "Inv # ": "Inv #",
        "Inv#": "Inv #",
        "Date": "Date",
        "Stock Code": "Stock Code",
        "Stock code": "Stock Code",
        "Description": "Description",
        "Qty": "Qty",
        "Total": "Total",
        "Cost": "Cost",
        "GP%": "GP%",
        "GP %": "GP%",
        "GP $": "GP $",
        "GP $ ": "GP $",
        "GP doller": "GP $",
    }
    sales = sales.rename(columns={k: v for k, v in rename_map.items() if k in sales.columns})

    if "Inv #" not in sales.columns or "Date" not in sales.columns or "Stock Code" not in sales.columns:
        sales.columns = [
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

    sales = sales.dropna(how="all").reset_index(drop=True)

    # Carry location labels down the sales export.
    sales["Location"] = pd.NA
    location_mask = (
        sales["Inv #"].astype(str).str.upper().str.startswith("LOCATION:", na=False)
    )
    sales.loc[location_mask, "Location"] = (
        sales.loc[location_mask, "Inv #"]
        .astype(str)
        .str.replace("LOCATION:", "", regex=False)
        .str.strip()
    )
    if "_SourceFile" in sales.columns:
        sales["Location"] = sales.groupby("_SourceFile", sort=False)["Location"].ffill()
    else:
        sales["Location"] = sales["Location"].ffill()

    # Preserve numeric SKUs as integers and alphanumeric SKUs as text.
    def normalize_sku(value):
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if not text:
            return pd.NA
        integer_text = text[:-2] if text.endswith(".0") else text
        if integer_text.isdigit():
            return int(integer_text)
        return text

    sales["SKU"] = sales["Stock Code"].map(normalize_sku)

    grand_total_mask = sales["Date"].astype(str).str.upper().str.contains(
        "GRAND TOTAL", na=False
    )
    sales = sales.loc[sales["GP%"].notna() & ~grand_total_mask].copy()

    def parse_date(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float)):
            try:
                return pd.to_datetime("1899-12-30") + pd.to_timedelta(int(value), unit="D")
            except Exception:
                return pd.NaT
        txt = str(value).strip()
        if txt == "":
            return pd.NaT
        try:
            return pd.to_datetime(txt, errors="raise")
        except Exception:
            return pd.NaT

    sales["Date"] = sales["Date"].apply(parse_date)
    sales["Stock Code"] = (
        sales["Stock Code"].astype(str).str.strip().str.replace("\r", "")
    )

    # Enrich sales rows with product metadata.
    options_lookup = df_options.copy()
    options_lookup["SKU"] = options_lookup["Stock Code"].map(normalize_sku)
    options_lookup = options_lookup[["SKU", "Supplier", "Class", "Fourth Product Group"]].drop_duplicates()

    sales = sales.merge(
        options_lookup,
        on="SKU",
        how="left",
        suffixes=("", "_Options")
    )
    sales = sales.rename(columns={
        "Supplier": "OptionsData.Supplier",
        "Class": "OptionsData.Class",
        "Fourth Product Group": "OptionsData.Fourth Product Group",
    })

    city_map = {
        "Perth": "Perth",
        "Sydney": "Sydney",
        "Brisbane": "Brisbane",
        "Melbourne": "Melbourne",
        "Adelaide": "Adelaide",
        "Oxford St": "Oxford St",
    }

    def get_city(location):
        loc = str(location or "").strip()
        for suffix, city in city_map.items():
            if loc.endswith(suffix):
                return city
        return "Other"

    sales["City"] = sales["Location"].apply(get_city)
    sales["Sales Channel"] = sales.apply(
        lambda row: (
            row["Location"]
            if row["City"] == "Other"
            else str(row["Location"]).replace(row["City"], "").strip()
        ),
        axis=1,
    )

    sales["Month"] = sales["Date"].dt.month_name()
    sales["Year"] = sales["Date"].dt.year

    sales = sales.rename(columns={"Total": "Sale"})
    positive_sales = sales["Qty"].astype(float).gt(0)
    positive_margin = (
        sales["GP $"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
        .gt(0)
    )
    sales = sales.loc[positive_sales & positive_margin].copy()

    def sales_channel_net(value):
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

    sales["Sales Channel Net"] = sales["Sales Channel"].apply(sales_channel_net)

    sales["Inv #"] = pd.to_numeric(sales["Inv #"], errors="coerce")
    sales["Sale"] = pd.to_numeric(
        sales["Sale"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    sales["Cost"] = pd.to_numeric(
        sales["Cost"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    sales["Qty"] = pd.to_numeric(sales["Qty"], errors="coerce")
    sales["GP%"] = pd.to_numeric(sales["GP%"], errors="coerce")
    sales["GP $"] = pd.to_numeric(
        sales["GP $"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )

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


# Top-right manual refresh button container
top_left, top_right = st.columns([8, 2])
with top_left:
    st.title("📦 Inventory & Purchasing System Migration")
    st.subheader("Options Data Integration")
with top_right:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data", use_container_width=True):
        load_csv.clear()
        load_raw_folder_data.clear()
        st.rerun()

st.sidebar.caption("Source data refreshes automatically every day at 7:00 AM.")

st.sidebar.markdown(
    "<div style='text-align: center; margin-top: -60px;'><img src='https://livingedge.com.au/on/demandware.static/-/Library-Sites-shared_library/default/dw1e07d773/logo.svg' width='100'></div>",
    unsafe_allow_html=True,
)


# Automatic source-data refresh limitation note:
# Streamlit does not natively support direct cron/time-of-day scheduling (like 7:00 AM) inside 
# st.fragment without an active browser session ticking it. Using st.fragment(run_every="1h") 
# checks periodically to clear caches and refresh when crossed into a new day or when active.
@st.fragment(run_every="1h")
def automatic_source_refresh():
    current_time = pd.Timestamp.now()
    last_refresh_date = st.session_state.get("_last_auto_refresh_date")
    
    # Check if it's past 7:00 AM local time and hasn't refreshed today yet
    if current_time.hour >= 7 and (last_refresh_date is None or last_refresh_date != current_time.date()):
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
                        filtered[column].astype("string").isin(selected_values)
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
# Streamlit's st.tabs() renders all tab code on each run and does not expose
# the active tab to Python, so a horizontal radio control is used here to
# allow the sidebar to change with the selected view.
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
            use_container_width=True,
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
            latest_source_time = pd.Timestamp(
                max((raw_data_path / name).stat().st_mtime for name in source_names),
                unit="s",
            )
            planner_col, source_col = st.columns([2, 1])
            with planner_col:
                st.subheader("Forecast Planner")
            with source_col:
                st.caption(
                    f"Loaded {len(source_names)} source files · "
                    f"{len(sales_data):,} formatted sales rows · "
                    f"latest source: {latest_source_time:%d-%m-%Y %H:%M}"
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
                selected_location = filter_selectbox(
                    "Location", sales_data["City"]
                )
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
                selected_class = filter_selectbox(
                    "Class", sales_data["OptionsData.Class"]
                )
                selected_model = filter_selectbox(
                    "Model", sales_data["Model"]
                )
                selected_sku = filter_selectbox(
                    "SKU", sales_data["SKU Key"]
                )

                st.subheader("Forecast settings")
                metric_options = ["Qty", "Sale", "Cost", "GP%", "GP $"]
                metric = st.selectbox(
                    "View",
                    metric_options,
                    index=0,
                )
                forecast_from = st.date_input(
                    "Forecast from",
                    value=pd.Timestamp("2024-01-01").date(),
                )
                bias = st.number_input(
                    "Forecast bias", min_value=0.0, max_value=5.0,
                    value=1.0,
                    step=0.05,
                )
                confidence_alpha = st.number_input(
                    "Confidence alpha", min_value=0.05, max_value=0.9,
                    value=0.35,
                    step=0.05,
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
            selected_sales = forecast_engine.filter_sales(sales_data, filters)
            actual, forecast, mape = forecast_engine.calculate_forecast(
                selected_sales,
                metric,
                pd.Timestamp(forecast_from),
                bias,
                confidence_alpha,
            )
            purchase_months = forecast_engine.monthly_purchase_orders(sheets["Purchase Orders"], filters)
            on_hand, free_stock = forecast_engine.inventory_totals(sheets["Current Inventory"], filters)
            # Filter sales strictly to months prior to the current month for historical MAPE evaluation
            current_month_start = pd.Timestamp.today().normalize().replace(day=1)
            historical_sales_for_mape = selected_sales.loc[selected_sales["Date"] < current_month_start]

            actual, forecast, mape = forecast_engine.calculate_forecast(
                historical_sales_for_mape if not historical_sales_for_mape.empty else selected_sales,
                metric,
                pd.Timestamp(forecast_from),
                bias,
                confidence_alpha,
            )

            if actual.empty:
                st.warning("No dated sales match these selections and forecast start date.")
            else:
                # Summary statistics are shown above the monthly comparison chart.
                displayed_forecast = forecast.head(12).copy()
                po_by_month = (
                    purchase_months.set_index("Date")["Open PO Qty"]
                    if not purchase_months.empty
                    else pd.Series(dtype=float)
                )
                displayed_forecast["Open PO Qty"] = (
                    displayed_forecast["Date"].map(po_by_month).fillna(0)
                )
                total_forecast = float(displayed_forecast["Forecast"].sum())
                total_po = float(displayed_forecast["Open PO Qty"].sum())
                summary = st.columns(5)
                summary[0].metric("12-month forecast", f"{total_forecast:,.0f}")
                summary[1].metric("On hand · all sites", f"{on_hand:,.0f}")
                summary[2].metric("Free stock · all sites", f"{free_stock:,.0f}")
                summary[3].metric("Open PO quantity", f"{total_po:,.0f}")
                summary[4].metric("Historical MAPE", f"{mape:.1%}" if mape is not None else "n/a")

                annual_profile = forecast_engine.yearly_monthly_profile(
                    selected_sales,
                    forecast.iloc[0:0],
                    metric,
                    pd.Timestamp(forecast_from),
                )
                year_columns = [column for column in annual_profile.columns if column != "Month"]
                st.subheader("Monthly comparison by year")
                actual_chart_data = (
                    annual_profile.melt(
                        id_vars="Month", var_name="Year", value_name=metric
                    )
                    .dropna(subset=[metric])
                )
                forecast_year = pd.Timestamp.today().year
                full_year_forecast = forecast_engine.forecast_calendar_year(
                    selected_sales,
                    metric,
                    forecast_year,
                    pd.Timestamp(forecast_from),
                    bias,
                    confidence_alpha,
                )
                forecast_chart_data = pd.concat(
                    [
                        forecast.loc[forecast["Date"].dt.year.gt(forecast_year)],
                        full_year_forecast,
                    ],
                    ignore_index=True,
                )
                forecast_chart_data["Year"] = forecast_chart_data["Date"].dt.year.astype(str)
                forecast_chart_data["Month"] = (
                    forecast_chart_data["Date"]
                    .dt.month.map(dict(enumerate(forecast_engine.MONTHS, start=1)))
                )
                actual_chart_data["Series"] = actual_chart_data["Year"]
                actual_chart_data["Line Type"] = "Actual"
                actual_chart_data = actual_chart_data.rename(columns={metric: "Value"})
                forecast_plot_data = forecast_chart_data.assign(
                    Series="Forecast " + forecast_chart_data["Year"],
                    **{"Line Type": "Forecast", "Value": forecast_chart_data["Forecast"]},
                )[["Month", "Series", "Line Type", "Value"]]
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
                series_order = sorted(year_columns, key=int) + forecast_series
                series_colors = [
                    "#0072b2",
                    "#e69f00",
                    "#009e73",
                    "#d55e00",
                    "#cc79a7",
                ]
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
                            scale=alt.Scale(
                                domain=series_order,
                                range=series_colors[: len(series_order)],
                            ),
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
                            alt.Tooltip("Value:Q", title=metric, format=",.0f"),
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
                                alt.Tooltip(f"{field}:Q", title=year, format=",.0f")
                            )
                    month_forecasts = forecast_chart_data.loc[
                        forecast_chart_data["Month"].eq(month)
                    ]
                    for year in sorted(month_forecasts["Year"].unique()):
                        forecast_value = month_forecasts.loc[
                            month_forecasts["Year"].eq(year), "Forecast"
                        ]
                        field = f"forecast_{year}"
                        tooltip_row[field] = float(forecast_value.iloc[0])
                        tooltip_fields.append(
                            alt.Tooltip(
                                f"{field}:Q", title=f"Forecast {year}", format=",.0f"
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
                st.altair_chart(
                    alt.layer(yearly_chart, *month_hover_charts),
                    use_container_width=True,
                )
                st.caption("Solid lines: actual by year · Dashed line: forecast")

                graph_table = (
                    chart_data.pivot(index="Series", columns="Month", values="Value")
                    .reindex(index=series_order, columns=forecast_engine.MONTHS)
                    .reset_index()
                )
                graph_table.columns.name = None
                st.dataframe(
                    graph_table,
                    use_container_width=True,
                    hide_index=True,
                    height=220,
                )

                with st.expander("Forward forecast and overrides"):
                    displayed_forecast["Manual Override"] = pd.Series(
                        [pd.NA] * len(displayed_forecast), dtype="Float64"
                    )
                    edited_forecast = st.data_editor(
                        displayed_forecast,
                        use_container_width=True,
                        hide_index=True,
                        disabled=["Date", "Forecast", "Lower", "Upper", "Open PO Qty"],
                        column_config={
                            "Date": st.column_config.DateColumn(format="MMM YYYY"),
                            "Forecast": st.column_config.NumberColumn(format="%.0f"),
                            "Lower": st.column_config.NumberColumn(format="%.0f"),
                            "Upper": st.column_config.NumberColumn(format="%.0f"),
                            "Open PO Qty": st.column_config.NumberColumn(format="%.0f"),
                            "Manual Override": st.column_config.NumberColumn(format="%.0f"),
                        },
                        key=f"forecast_editor_{selected_sku}_{metric}",
                    )
                    st.download_button(
                        "Download forecast CSV",
                        edited_forecast.to_csv(index=False).encode("utf-8"),
                        file_name="forecast.csv",
                        mime="text/csv",
                    )
                edited_forecast["Final Forecast"] = pd.to_numeric(
                    edited_forecast["Manual Override"], errors="coerce"
                ).fillna(edited_forecast["Forecast"])
                edited_forecast["Projected Free Stock"] = (
                    free_stock
                    + edited_forecast["Open PO Qty"].cumsum()
                    - edited_forecast["Final Forecast"].cumsum()
                )

                st.caption(
                    "Forecast uses 12-month seasonal Holt-Winters, blended 70/30 with recent monthly average, "
                    "adjusted by confidence alpha and forecast bias."
                )
        except Exception as e:
            st.error(f"Error rendering forecast sheet: {e}")
