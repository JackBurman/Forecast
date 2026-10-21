from io import BytesIO
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


HEADER_ROWS = {
    "Current Inventory": 1,
    "Purchase Orders": 1,
    "Data Validation": 1,
}
CALCULATION_SHEETS = {
    "FORCASTING",
    "Current Inventory",
    "Purchase Orders",
    "OptionsData",
    "Sales Data",
    "Vendor Codes",
}
CALCULATION_COLUMNS = {
    "Current Inventory": [
        "Location",
        "Supplier",
        "Product Code",
        "Loc Qty on hand",
        "Qty Alloc",
        "Qty Free",
        "Group 4",
    ],
    "Purchase Orders": [
        "Supplier",
        "Stock Code",
        "Qty Ordered",
        "ETA Date",
        "Model",
        "Class",
        "Group4",
    ],
    "OptionsData": [
        "SKU",
        "Stock Code",
        "Supplier",
        "Class",
        "Fourth Product Group",
        "Model",
    ],
    "Sales Data": [
        "City",
        "Sales Channel Net",
        "Date",
        "Year",
        "SKU",
        "Model",
        "OptionsData.Supplier",
        "Vendor Codes.Brand__r.Name",
        "OptionsData.Class",
        "OptionsData.Fourth Product Group",
        "Qty",
        "Sale",
        "Cost",
        "GP%",
        "GP $",
    ],
    "Vendor Codes": ["Options_ID__c", "Brand__r.Name"],
}
METRICS = {
    "Qty": "Qty",
    "Sale": "Sale",
    "Cost": "Cost",
    "GP%": "GP%",
    "GP $": "GP $",
}
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
FILTER_COLUMNS = {
    "Location": "City",
    "Sales Channel": "Sales Channel Net",
    "Classification": "OptionsData.Fourth Product Group",
    "Supplier": "Supplier Brand",
    "Class": "OptionsData.Class",
    "Model": "Model",
    "SKU": "SKU Key",
}


def _key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def read_forecast_workbook(
    source: str | BytesIO,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    workbook = pd.ExcelFile(source, engine="calamine")
    sheets = {}
    for sheet_name in workbook.sheet_names:
        if sheet_name not in CALCULATION_SHEETS:
            continue
        header = None if sheet_name == "FORCASTING" else HEADER_ROWS.get(sheet_name, 0)
        frame = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=header,
            usecols=CALCULATION_COLUMNS.get(sheet_name),
        )
        frame.columns = [str(column).strip() for column in frame.columns]
        sheets[sheet_name] = frame
    return workbook.sheet_names, sheets


def read_forecast_sheet(source: str | BytesIO, sheet_name: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(source, engine="calamine")
    header = None if sheet_name == "FORCASTING" else HEADER_ROWS.get(sheet_name, 0)
    frame = pd.read_excel(workbook, sheet_name=sheet_name, header=header)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def read_raw_data_folder(raw_data_dir: str | Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    raw_dir = Path(raw_data_dir)
    sales_dir = raw_dir / "Sales Data"
    required_files = [
        raw_dir / "Inventory_report.xlsx",
        raw_dir / "PODashboard.xlsx",
        raw_dir / "options_data.csv",
    ]
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required Raw_data files: {', '.join(missing_files)}")
    if not sales_dir.is_dir():
        raise FileNotFoundError(f"Sales Data folder not found: {sales_dir}")

    sales_files = sorted(
        path
        for path in sales_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".xls", ".xlsx"}
    )
    if not sales_files:
        raise FileNotFoundError(f"No .xls or .xlsx sales files found in {sales_dir}")

    options = pd.read_csv(
        raw_dir / "options_data.csv",
        dtype={"Stock Code": "string", "Supplier Code": "string"},
        low_memory=False,
    )
    options.columns = [str(column).strip() for column in options.columns]
    options = options.loc[
        :, ~options.columns.astype(str).str.match(r"^Unnamed:")
    ].copy()
    if "SKU" not in options.columns:
        options["SKU"] = options["Stock Code"]

    inventory = pd.read_excel(
        required_files[0], sheet_name=0, header=1, engine="calamine"
    )
    purchase_orders = pd.read_excel(
        required_files[1], sheet_name=0, header=0, engine="calamine"
    )

    sales_frames = []
    for path in sales_files:
        frame = pd.read_excel(
            path,
            sheet_name=0,
            header=0,
            engine="calamine",
            dtype={"Stock Code": "string"},
        )
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.loc[
            :, ~frame.columns.astype(str).str.match(r"^Unnamed:")
        ].copy()
        frame["_SourceFile"] = path.name
        sales_frames.append(frame)

    sources = {
        "Options Data": options,
        "Current Inventory": inventory,
        "Purchase Orders": purchase_orders,
        "Sales Raw": pd.concat(sales_frames, ignore_index=True, sort=False),
    }
    source_names = [path.name for path in required_files] + [
        f"Sales Data/{path.name}" for path in sales_files
    ]
    return sources, source_names


def prepare_forecast_sheets(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared = dict(sheets)
    options = prepared["OptionsData"]
    vendors = prepared.get("Vendor Codes", pd.DataFrame())

    vendor_brands = {}
    if {"Options_ID__c", "Brand__r.Name"}.issubset(vendors.columns):
        vendor_brands = dict(
            zip(vendors["Options_ID__c"].map(_key), vendors["Brand__r.Name"].fillna(""))
        )

    option_lookup = options[[
        "SKU",
        "Stock Code",
        "Supplier",
        "Class",
        "Fourth Product Group",
        "Model",
    ]].copy()
    option_lookup["SKU Key"] = option_lookup["SKU"].map(_key)
    option_lookup["Stock Key"] = option_lookup["Stock Code"].map(_key)
    option_lookup["Supplier Brand"] = option_lookup["Supplier"].map(
        lambda value: vendor_brands.get(_key(value), value)
    )
    option_by_sku = option_lookup.drop_duplicates("SKU Key").set_index("SKU Key")
    option_by_stock = option_lookup.drop_duplicates("Stock Key").set_index("Stock Key")
    stock_to_sku = option_by_stock["SKU Key"]

    def enrich(
        frame: pd.DataFrame,
        key_column: str,
        lookup: pd.DataFrame,
        existing: dict[str, str],
    ) -> pd.DataFrame:
        frame = frame.copy()
        frame["SKU Key"] = frame[key_column].map(_key)
        for column, detail_name in existing.items():
            if column not in frame.columns:
                frame[column] = pd.NA
            formula_mask = frame[column].map(
                lambda value: isinstance(value, str) and value.startswith("=")
            )
            frame.loc[formula_mask, column] = pd.NA
            missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
            mapped = frame["SKU Key"].map(lookup[detail_name])
            frame.loc[missing, column] = mapped.loc[missing]
        return frame

    sales = prepared["Sales Data"].copy()
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sales = enrich(
        sales,
        "SKU",
        option_by_sku,
        {
            "Model": "Model",
            "OptionsData.Class": "Class",
            "OptionsData.Fourth Product Group": "Fourth Product Group",
            "OptionsData.Supplier": "Supplier Brand",
        },
    )
    sales["Supplier Brand"] = sales["OptionsData.Supplier"].map(
        lambda value: vendor_brands.get(_key(value), value)
    )
    for column in METRICS.values():
        if column in sales.columns:
            sales[column] = pd.to_numeric(
                sales[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            ).fillna(0)
    prepared["Sales Data"] = sales

    purchase_orders = prepared["Purchase Orders"].copy()
    purchase_orders["ETA Date"] = pd.to_datetime(
        purchase_orders["ETA Date"], format="mixed", errors="coerce"
    )
    purchase_orders = enrich(
        purchase_orders,
        "Stock Code",
        option_by_stock,
        {
            "Model": "Model",
            "Class": "Class",
            "Group4": "Fourth Product Group",
        },
    )
    purchase_orders["SKU Key"] = purchase_orders["SKU Key"].map(stock_to_sku).fillna(purchase_orders["SKU Key"])
    purchase_orders["Supplier Brand"] = purchase_orders["Supplier"].map(
        lambda value: vendor_brands.get(_key(value), value)
    )
    purchase_orders["Qty Ordered"] = pd.to_numeric(purchase_orders["Qty Ordered"], errors="coerce").fillna(0)
    prepared["Purchase Orders"] = purchase_orders

    inventory = prepared["Current Inventory"].copy()
    inventory = enrich(
        inventory,
        "Product Code",
        option_by_stock,
        {
            "Model": "Model",
            "Class": "Class",
            "Group 4": "Fourth Product Group",
        },
    )
    inventory["SKU Key"] = inventory["SKU Key"].map(stock_to_sku).fillna(inventory["SKU Key"])
    inventory["Supplier Brand"] = inventory["Supplier"].map(
        lambda value: vendor_brands.get(_key(value), value)
    )
    for column in ["Loc Qty on hand", "Qty Alloc", "Qty Free"]:
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce").fillna(0)
    prepared["Current Inventory"] = inventory
    return prepared


def filter_sales(sales: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    filtered = sales
    for label, value in filters.items():
        column = FILTER_COLUMNS[label]
        if value == "TOTAL" or column not in filtered.columns:
            continue
        values = filtered[column].map(_key)
        if label in {"Model", "SKU"}:
            filtered = filtered.loc[values.str.contains(_key(value), case=False, regex=False)]
        else:
            filtered = filtered.loc[values.eq(_key(value))]
    return filtered


def filter_supporting_data(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    filtered = frame
    columns = {
        "Classification": "Group4" if "Group4" in frame.columns else "Group 4",
        "Supplier": "Supplier Brand",
        "Class": "Class" if "Class" in frame.columns else "OptionsData.Class",
        "Model": "Model",
        "SKU": "SKU Key",
    }
    for label, column in columns.items():
        value = filters.get(label, "TOTAL")
        if value != "TOTAL" and column in filtered.columns:
            values = filtered[column].map(_key)
            if label in {"Model", "SKU"}:
                filtered = filtered.loc[
                    values.str.contains(_key(value), case=False, regex=False)
                ]
            else:
                filtered = filtered.loc[values.eq(_key(value))]
    return filtered


def calculate_forecast(
    sales: pd.DataFrame,
    metric: str,
    forecast_from: pd.Timestamp,
    bias: float,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    metric_column = METRICS[metric]
    selected = sales.loc[sales["Date"].notna() & (sales["Date"] >= forecast_from)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["Date", "Actual"]), pd.DataFrame(), None

    selected["Month Start"] = selected["Date"].dt.to_period("M").dt.to_timestamp()
    actual = selected.groupby("Month Start")[metric_column].sum().sort_index()
    actual = actual.asfreq("MS", fill_value=0)
    history = actual.astype(float)
    final_month = max(history.index.max(), pd.Timestamp.today().to_period("M").to_timestamp())
    forecast_end = pd.Timestamp(year=final_month.year + 1, month=12, day=1)
    forecast_dates = pd.date_range(history.index.max() + pd.offsets.MonthBegin(1), forecast_end, freq="MS")
    if forecast_dates.empty:
        return actual.rename("Actual").rename_axis("Date").reset_index(), pd.DataFrame(), None

    seasonal_model = None
    if len(history) >= 24 and history.abs().sum() > 0:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            seasonal_model = ExponentialSmoothing(
                history,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
            raw_forecast = seasonal_model.forecast(len(forecast_dates)).to_numpy(dtype=float)
            residuals = (history - seasonal_model.fittedvalues).dropna().tail(24)
            error_scale = float(np.sqrt(np.mean(np.square(residuals)))) if not residuals.empty else 0.0
            fitted = seasonal_model.fittedvalues
            mape_actual = history.loc[fitted.index].tail(12)
            mape_fitted = fitted.tail(12)
            valid = mape_actual.ne(0)
            mape = float(((mape_actual.loc[valid] - mape_fitted.loc[valid]).abs() / mape_actual.loc[valid].abs()).mean()) if valid.any() else None
        except Exception:
            seasonal_model = None
            mape = None
            error_scale = float(history.tail(12).std(ddof=0)) if len(history) > 1 else 0.0
    else:
        mape = None
        error_scale = float(history.tail(12).std(ddof=0)) if len(history) > 1 else 0.0

    if seasonal_model is None:
        seasonal_means = history.groupby(history.index.month).mean()
        raw_forecast = np.array(
            [seasonal_means.get(date.month, history.mean()) for date in forecast_dates],
            dtype=float,
        )

    smooth_by_year = {}
    for year in forecast_dates.year.unique():
        previous_november_december = pd.date_range(
            start=pd.Timestamp(year=int(year) - 1, month=11, day=1),
            periods=2,
            freq="MS",
        )
        smooth_by_year[year] = float(
            history.reindex(previous_november_december, fill_value=0).mean()
        )
    smoothing = np.array([smooth_by_year[year] for year in forecast_dates.year])
    blended = np.where(raw_forecast < 0, smoothing, raw_forecast * 0.7 + smoothing * 0.3)
    values = np.rint(np.maximum(0, blended * bias))
    z_score = NormalDist().inv_cdf(1 - alpha / 2)
    steps = np.arange(1, len(values) + 1, dtype=float)
    margin = z_score * error_scale * np.sqrt(steps)
    forecast = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast": values,
            "Lower": np.maximum(0, values - margin),
            "Upper": values + margin,
        }
    )
    actual_frame = actual.rename("Actual").rename_axis("Date").reset_index()
    return actual_frame, forecast, mape


def forecast_calendar_year(
    sales: pd.DataFrame,
    metric: str,
    forecast_year: int,
    forecast_from: pd.Timestamp,
    bias: float,
    alpha: float,
) -> pd.DataFrame:
    year_start = pd.Timestamp(year=forecast_year, month=1, day=1)
    training_sales = sales.loc[sales["Date"].notna() & (sales["Date"] < year_start)]
    _, forecast, _ = calculate_forecast(
        training_sales, metric, forecast_from, bias, alpha
    )
    return forecast.loc[forecast["Date"].dt.year.eq(forecast_year)].reset_index(drop=True)


def yearly_monthly_profile(
    sales: pd.DataFrame,
    forecast: pd.DataFrame,
    metric: str,
    forecast_from: pd.Timestamp,
) -> pd.DataFrame:
    metric_column = METRICS[metric]
    selected = sales.loc[sales["Date"].notna() & (sales["Date"] >= forecast_from)].copy()
    selected["Year"] = selected["Date"].dt.year
    selected["Month Number"] = selected["Date"].dt.month
    actual_values = selected.groupby(["Year", "Month Number"])[metric_column].sum()

    forecast_values = pd.Series(dtype=float)
    if not forecast.empty:
        future = forecast.copy()
        future["Year"] = future["Date"].dt.year
        future["Month Number"] = future["Date"].dt.month
        forecast_values = future.set_index(["Year", "Month Number"])["Forecast"]

    if forecast_values.empty:
        values = actual_values
    elif actual_values.empty:
        values = forecast_values
    else:
        values = actual_values.combine_first(forecast_values)
    values = values.rename("Value")
    if values.empty:
        return pd.DataFrame({"Month": MONTHS})

    monthly = values.reset_index()
    monthly["Month"] = monthly["Month Number"].map(dict(enumerate(MONTHS, start=1)))
    profile = monthly.pivot(index="Month", columns="Year", values="Value")
    profile = profile.reindex(MONTHS).reset_index()
    profile.columns.name = None
    profile.columns = [str(column) if column != "Month" else column for column in profile.columns]
    return profile


def monthly_purchase_orders(purchase_orders: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    selected = filter_supporting_data(purchase_orders, filters)
    selected = selected.loc[selected["ETA Date"].notna()].copy()
    if selected.empty:
        return pd.DataFrame(columns=["Date", "Open PO Qty"])
    selected["Date"] = selected["ETA Date"].dt.to_period("M").dt.to_timestamp()
    return selected.groupby("Date", as_index=False)["Qty Ordered"].sum().rename(columns={"Qty Ordered": "Open PO Qty"})


def inventory_totals(inventory: pd.DataFrame, filters: dict[str, str]) -> tuple[float, float]:
    selected = filter_supporting_data(inventory, filters)
    return float(selected["Loc Qty on hand"].sum()), float(selected["Qty Free"].sum())