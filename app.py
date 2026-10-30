from datetime import datetime
from pathlib import Path
import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import forecast_engine  # type: ignore
from inventory_charts import stock_health_breakdown, stock_health_by_group

# =============================================================================
# App setup
# =============================================================================
st.set_page_config(page_title="Forecasting & Purchasing System", layout="wide")

# Layout-only styling: tighter page frame, compact metric tiles, card headings.
st.markdown(
    """
    <style>
        .block-container { padding-top: 3.2rem; padding-bottom: 1rem; }
        [data-testid="stVerticalBlock"] { gap: 0.6rem; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
        h1 { font-size: 1.8rem !important; padding-top: 0 !important; padding-bottom: 0.2rem !important; }
        h2, h3 { padding-top: 0.3rem !important; padding-bottom: 0.2rem !important; }
        [data-testid="stMetric"] { padding: 0.1rem 0; }
        [data-testid="stMetricLabel"] p { font-size: 0.78rem; line-height: 1.2; white-space: normal; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; line-height: 1.2; }
        [data-testid="stMetricDelta"] { font-size: 0.78rem; }
        .card-title {
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; opacity: 0.65; margin: 0 0 0.15rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _streamlit_version() -> tuple[int, ...]:
    parts: list[int] = []
    for piece in st.__version__.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# Streamlit 1.51+ replaced use_container_width=True with width="stretch".
FULL_WIDTH = (
    {"width": "stretch"}
    if _streamlit_version() >= (1, 51)
    else {"use_container_width": True}
)

MONTHS = list(forecast_engine.MONTHS)
MONTH_NUMBER_TO_NAME = dict(enumerate(MONTHS, start=1))
MONTH_NAME_TO_NUMBER = {name: number for number, name in MONTH_NUMBER_TO_NAME.items()}
FY_MONTHS = MONTHS[6:] + MONTHS[:6]

# Data paths
base_dir = Path(__file__).resolve().parent
raw_data_path = base_dir / "Raw_data"
csv_path = raw_data_path / "options_data.csv"
inventory_folder_path = raw_data_path / "inventory"
podash_folder_path = raw_data_path / "PODash"
loan_folder_path = raw_data_path / "loan"

# Files forecast_engine.read_raw_data_folder expects to find. When the live
# inventory / PO exports live in the sub-folders instead, zero-row
# placeholders are written for the duration of the read and removed again.
ENGINE_PLACEHOLDERS = {
    "Inventory_report.xlsx": ["Stock Code", "Product Code", "On Hand", "Free Stock"],
    "PODashboard.xlsx": ["Stock Code", "Product Code", "Date", "Open PO Qty"],
}
TABULAR_EXTENSIONS = {".csv", ".xls", ".xlsx"}


# =============================================================================
# Column detection
# =============================================================================

SKU_HEADERS = ["sku", "sku key", "stock code", "stockcode", "item code", "item number", "product code", "item", "code"]
ON_HAND_HEADERS = ["on hand", "onhand", "qty on hand", "quantity on hand", "stock on hand", "soh", "on hand qty", "qoh", "in stock", "stock qty", "stock", "quantity", "qty"]
FREE_HEADERS = ["free", "free stock", "free qty", "qty free", "available", "available qty", "qty available", "unallocated"]
OPEN_PO_HEADERS = ["open po qty", "open qty", "outstanding qty", "qty outstanding", "qty open", "balance qty", "po qty", "quantity", "qty"]
DATE_HEADERS = ["date", "due date", "expected date", "eta", "po date", "order date"]

INVENTORY_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "sku": SKU_HEADERS,
    "on_hand": ON_HAND_HEADERS,
    "free": FREE_HEADERS,
    "site": ["site", "location", "warehouse", "branch", "store", "showroom"],
    "value": ["stock value", "inventory value", "on hand value", "total cost", "cost value", "extended cost", "value"],
    "cost": ["average option cost", "avg option cost", "average cost", "avg cost", "option cost", "unit cost", "cost per unit", "cost price", "standard cost", "cost"],
    "description": ["description", "item description", "product description", "product name", "name", "product"],
}
INVENTORY_ROLE_LABELS = {
    "sku": "SKU column",
    "on_hand": "On-hand quantity column",
    "free": "Free stock column",
    "site": "Site / location column",
    "value": "Stock value column",
    "cost": "Average option cost column",
    "description": "Description column",
}


def normalise_header(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def detect_column(columns, candidates: list[str], exclude: tuple = ()) -> str | None:
    """Best-matching column for a role: exact normalised match first, then substring."""
    normalised = {col: normalise_header(col) for col in columns if col not in exclude}
    for candidate in candidates:
        for col, norm in normalised.items():
            if norm == candidate:
                return col
    for candidate in candidates:
        for col, norm in normalised.items():
            if candidate in norm:
                return col
    return None


def detect_inventory_columns(inventory: pd.DataFrame) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    used: list[str] = []
    for role, candidates in INVENTORY_COLUMN_CANDIDATES.items():
        col = detect_column(inventory.columns, candidates, exclude=tuple(used))
        found[role] = col
        if col:
            used.append(col)
    return found


# =============================================================================
# Data loading and cleaning
# =============================================================================

@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"SKU": "string", "Stock Code": "string", "Model": "string"},
        low_memory=False,
    )


def to_number(series: pd.Series) -> pd.Series:
    """Coerce '1,234.50', '$1,234', '45%', '(12)' or blanks to floats; junk becomes NaN."""
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[,$%\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    return pd.to_numeric(text, errors="coerce").astype(float)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column that may mix Excel serials, real datetimes and day-first text."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if series.empty:
        return result

    kinds = series.map(type)
    is_serial = series.notna() & kinds.isin([int, float, np.int64, np.float64, np.int32, np.float32]) & ~kinds.eq(bool)
    if is_serial.any():
        days = series[is_serial].astype(float).astype(int)
        result.loc[is_serial] = pd.to_datetime("1899-12-30") + pd.to_timedelta(days, unit="D")

    is_stamp = series.notna() & kinds.isin([pd.Timestamp, datetime, np.datetime64])
    if is_stamp.any():
        result.loc[is_stamp] = pd.to_datetime(series[is_stamp], errors="coerce")

    is_text = series.notna() & ~is_serial & ~is_stamp
    if is_text.any():
        text = series[is_text].astype(str).str.strip()
        fast = pd.to_datetime(text, dayfirst=True, errors="coerce")
        missing = fast.isna() & text.ne("")
        if missing.any():
            fast.loc[missing] = pd.to_datetime(text[missing], dayfirst=True, errors="coerce", format="mixed")
        result.loc[is_text] = fast

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


def sku_text(value) -> str:
    normalised = normalize_sku(value)
    return "" if pd.isna(normalised) else str(normalised)


def map_unique(series: pd.Series, func) -> pd.Series:
    uniques = series.unique()
    lookup = {value: func(value) for value in uniques}
    return series.map(lookup)


def frame_cache_key(frame: pd.DataFrame):
    token = frame.attrs.get("cache_token")
    if token is not None:
        return (token, frame.shape)
    return pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()


BY_TOKEN = {"hash_funcs": {pd.DataFrame: frame_cache_key, pd.Series: lambda s: frame_cache_key(s.to_frame())}}


def tag_frames(sheets: dict, token: str) -> dict:
    for name, frame in sheets.items():
        if isinstance(frame, pd.DataFrame):
            frame.attrs["cache_token"] = f"{token}:{name}"
    return sheets


@st.cache_data(show_spinner=False, max_entries=64, **BY_TOKEN)
def column_choices(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = frame[column].dropna().unique()
    cleaned = {str(value).strip() for value in values}
    cleaned.discard("")
    cleaned.discard("nan")
    return sorted(cleaned, key=str.casefold)


def clean_sales_data(df_sales: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
    sales = df_sales.copy()
    sales.columns = [str(col).strip() for col in sales.columns]

    rename_map = {
        "Inv#": "Inv #",
        "Stock code": "Stock Code",
        "GP %": "GP%",
        "GP doller": "GP $",
    }
    sales = sales.rename(columns={k: v for k, v in rename_map.items() if k in sales.columns})

    expected_columns = ["Inv #", "Date", "Stock Code", "Description", "Qty", "Total", "Cost", "GP%", "GP $"]
    if not {"Inv #", "Date", "Stock Code"}.issubset(sales.columns):
        data_columns = [col for col in sales.columns if col != "_SourceFile"]
        if len(data_columns) != len(expected_columns):
            raise ValueError(
                "Sales export columns were not recognised: "
                + ", ".join(map(str, sales.columns))
            )
        sales = sales.rename(columns=dict(zip(data_columns, expected_columns)))

    sales = sales.dropna(how="all").reset_index(drop=True)

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

    sales["SKU"] = map_unique(sales["Stock Code"], normalize_sku)

    grand_total_mask = sales["Date"].astype(str).str.upper().str.contains(
        "GRAND TOTAL", na=False
    )
    sales = sales.loc[sales["GP%"].notna() & ~grand_total_mask].copy()

    sales["Date"] = parse_dates(sales["Date"])
    sales["Stock Code"] = (
        sales["Stock Code"].astype(str).str.strip().str.replace("\r", "")
    )

    options_lookup = df_options.copy()
    options_lookup["SKU"] = map_unique(options_lookup["Stock Code"], normalize_sku)
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

    sales = sales.rename(columns={"Total": "Sale"})
    for column in ["Qty", "Sale", "Cost", "GP%", "GP $"]:
        if column in sales.columns:
            sales[column] = to_number(sales[column])
    sales["Inv #"] = pd.to_numeric(sales["Inv #"], errors="coerce")

    sales = sales.loc[sales["Qty"].gt(0) & sales["GP $"].gt(0)].copy()

    def sales_channel_net(value) -> str:
        text = str(value or "").upper()
        if "A&D" in text or "ECOMMERCE SALES" in text or "RETAIL" in text:
            return "Retail"
        if "WAREHOUSE" in text or "CLEARANCE SALE" in text or "LOAN STOCK" in text:
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
        "City", "Sales Channel", "Sales Channel Net", "Location", "Inv #", "Date", "Month", "Year",
        "SKU", "OptionsData.Supplier", "OptionsData.Class", "Description",
        "OptionsData.Fourth Product Group", "Qty", "Sale", "Cost", "GP%", "GP $",
    ]
    final_columns = [col for col in final_columns if col in sales.columns]
    sales = sales[final_columns]
    if "SKU" in sales.columns:
        sales["SKU"] = sales["SKU"].astype("string")
    sales = sales.sort_values("Date", ascending=False).reset_index(drop=True)
    for column in ["City", "Sales Channel", "Sales Channel Net", "Location", "Month"]:
        if column in sales.columns:
            sales[column] = sales[column].astype("category")
    return sales


def raw_folder_manifest(folder: Path) -> tuple[tuple[str, int, int], ...]:
    if not folder.exists():
        return ()
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in TABULAR_EXTENSIONS and path.name not in ENGINE_PLACEHOLDERS
    )
    return tuple(
        (str(path.relative_to(folder)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    )


def newest_file(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


ZIP_MAGIC = b"PK\x03\x04"
BIFF_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
LFS_MAGIC = b"version https://git-lfs"
WORKBOOK_PROBLEMS = {
    "empty": "the file is empty (0 bytes)",
    "lfs-pointer": (
        "the file is a Git LFS pointer, not the workbook itself. Streamlit Community Cloud does not fetch "
        "LFS objects: commit the real file (git lfs untrack <file>; git add --renormalize <file>) "
        "or load the data from external storage"
    ),
    "unknown": "the file is not a recognisable workbook (neither an .xlsx package nor a legacy .xls)",
    "xls-no-xlrd": "the file is a legacy .xls workbook and the 'xlrd' package is not installed; add xlrd to requirements.txt",
}


def sniff_workbook(path: Path) -> str:
    if path.stat().st_size == 0:
        return "empty"
    with open(path, "rb") as handle:
        head = handle.read(512)
    if head.startswith(ZIP_MAGIC):
        return "xlsx"
    if head.startswith(BIFF_MAGIC):
        return "xls"
    if head.startswith(LFS_MAGIC):
        return "lfs-pointer"
    if head.lstrip()[:1] == b"<":
        return "html"
    try:
        head.decode("utf-8")
        return "csv"
    except UnicodeDecodeError:
        return "unknown"


def xlrd_available() -> bool:
    try:
        import xlrd  # noqa: F401
    except ImportError:
        return False
    return True


def calamine_available() -> bool:
    try:
        import python_calamine  # noqa: F401
    except ImportError:
        return False
    major, minor = (int(part) for part in pd.__version__.split(".")[:2])
    return (major, minor) >= (2, 2)


def workbook_problem(path: Path) -> str | None:
    kind = sniff_workbook(path)
    if kind == "xls" and not xlrd_available():
        kind = "xls-no-xlrd"
    return WORKBOOK_PROBLEMS.get(kind)


_pandas_read_excel = pd.read_excel


def read_workbook(path: Path, *args, **kwargs) -> pd.DataFrame:
    kind = sniff_workbook(path)
    if kind == "xlsx":
        return _pandas_read_excel(path, *args, engine="openpyxl", **kwargs)
    if kind == "xls":
        if not xlrd_available():
            raise ImportError(f"Cannot read '{path.name}': {WORKBOOK_PROBLEMS['xls-no-xlrd']}")
        return _pandas_read_excel(path, *args, engine="xlrd", **kwargs)
    if kind == "html":
        try:
            tables = pd.read_html(path, header=kwargs.get("header", 0))
        except ImportError as e:
            raise ImportError(
                f"'{path.name}' is an HTML table saved with an Excel extension; reading it needs the "
                "'lxml' package — add lxml to requirements.txt"
            ) from e
        if not tables:
            raise ValueError(f"Cannot read '{path.name}': the HTML export contains no table")
        frame = tables[0]
        skiprows = kwargs.get("skiprows")
        if isinstance(skiprows, int) and skiprows > 0:
            frame = frame.iloc[skiprows:].reset_index(drop=True)
        return frame.head(kwargs["nrows"]) if kwargs.get("nrows") else frame
    if kind == "csv":
        return pd.read_csv(
            path, header=kwargs.get("header", 0), skiprows=kwargs.get("skiprows"), nrows=kwargs.get("nrows"),
            encoding="utf-8-sig", encoding_errors="replace", low_memory=False,
        )
    raise ValueError(f"Cannot read '{path.name}': {WORKBOOK_PROBLEMS[kind]}")


def _read_excel_by_content(io, *args, **kwargs):
    engine = kwargs.get("engine")
    if isinstance(io, (str, Path)):
        path = Path(io)
        if path.is_file() and path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
            if engine is None or (engine == "calamine" and not calamine_available()):
                kwargs.pop("engine", None)
                return _tolerant_read(read_workbook, path, *args, **kwargs)
            return _tolerant_read(_pandas_read_excel, path, *args, **kwargs)
    return _pandas_read_excel(io, *args, **kwargs)


def _tolerant_read(reader, path: Path, *args, **kwargs) -> pd.DataFrame:
    try:
        return reader(path, *args, **kwargs)
    except ValueError as e:
        message = str(e)
        if "Passed header" in message or "lines in file" in message:
            retry = {key: value for key, value in kwargs.items() if key not in ("header", "skiprows")}
            return reader(path, *args, header=0, **retry)
        raise


pd.read_excel = _read_excel_by_content


def workbook_problems(folder: Path) -> list[str]:
    problems = []
    if not folder.exists():
        return problems
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
            problem = workbook_problem(path)
            if problem:
                problems.append(f"{path.relative_to(folder)} — {problem}")
    return problems


def detect_header_row(preview: pd.DataFrame) -> int:
    if preview.empty:
        return 0
    best_row, best_count = 0, -1
    for index, row in preview.iterrows():
        cells = [normalise_header(value) for value in row if pd.notna(value) and str(value).strip()]
        if not cells:
            continue
        if any(cell in SKU_HEADERS or any(key in cell for key in ("stock code", "product code", "sku", "item code")) for cell in cells):
            return int(index)
        if len(cells) > best_count:
            best_row, best_count = int(index), len(cells)
    return best_row


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            lines = [handle.readline() for _ in range(20)]
        ragged = [line.rstrip("\r\n").split(",") for line in lines if line.strip()]
        width = max((len(row) for row in ragged), default=1)
        preview = pd.DataFrame([row + [None] * (width - len(row)) for row in ragged])
        header_row = detect_header_row(preview)
        frame = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    else:
        preview = read_workbook(path, header=None, nrows=20)
        header_row = detect_header_row(preview)
        try:
            frame = read_workbook(path, skiprows=header_row)
        except ValueError:
            frame = read_workbook(path, header=0)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame.dropna(how="all").reset_index(drop=True)


def add_alias_columns(frame: pd.DataFrame, aliases: dict[str, list[str]], numeric: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    for alias, candidates in aliases.items():
        if alias in frame.columns:
            continue
        source = detect_column(frame.columns, candidates)
        if source is None:
            frame[alias] = np.nan if alias in numeric else pd.Series(dtype=object)
        elif alias in numeric:
            frame[alias] = to_number(frame[source])
        else:
            frame[alias] = frame[source]
    return frame


def standardise_inventory(frame: pd.DataFrame | None) -> pd.DataFrame:
    frame = pd.DataFrame() if frame is None else frame
    frame = add_alias_columns(
        frame,
        {"Stock Code": SKU_HEADERS, "Product Code": SKU_HEADERS, "On Hand": ON_HAND_HEADERS, "Free Stock": FREE_HEADERS},
        numeric=("On Hand", "Free Stock"),
    )
    if frame["Free Stock"].isna().all():
        frame["Free Stock"] = frame["On Hand"]
    return frame


def standardise_purchase_orders(frame: pd.DataFrame | None) -> pd.DataFrame:
    frame = pd.DataFrame() if frame is None else frame
    return add_alias_columns(
        frame,
        {"Stock Code": SKU_HEADERS, "Product Code": SKU_HEADERS, "Date": DATE_HEADERS, "Open PO Qty": OPEN_PO_HEADERS},
        numeric=("Open PO Qty",),
    )


def write_engine_placeholders() -> list[Path]:
    created = []
    for name, columns in ENGINE_PLACEHOLDERS.items():
        path = raw_data_path / name
        if not path.exists() or path.stat().st_size == 0:
            pd.DataFrame([[f"{name} placeholder"] + [""] * (len(columns) - 1), columns]).to_excel(
                path, index=False, header=False
            )
            created.append(path)
    return created


def column_summary(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "missing"
    return f"{len(frame):,} rows · columns: {', '.join(map(str, frame.columns)) or '(none)'}"


def local_prepare_forecast_sheets(inputs: dict) -> dict:
    sales = inputs["Sales Data"].copy()
    options = inputs.get("OptionsData")
    sales["SKU Key"] = sales["SKU"].astype("string")
    sales["Supplier Brand"] = sales.get("OptionsData.Supplier")
    sales["Model"] = sales.get("Description")
    if options is not None and not options.empty:
        sku_col = detect_column(options.columns, ["stock code", "sku", "sku key"])
        if sku_col:
            lookup = options.assign(_key=options[sku_col].map(sku_text)).drop_duplicates("_key").set_index("_key")
            keys = sales["SKU"].map(sku_text)
            brand_col = detect_column(options.columns, ["supplier brand", "brand"], exclude=(sku_col,))
            model_col = detect_column(options.columns, ["model"], exclude=(sku_col,))
            if brand_col:
                sales["Supplier Brand"] = keys.map(lookup[brand_col]).fillna(sales["Supplier Brand"])
            if model_col:
                sales["Model"] = keys.map(lookup[model_col]).fillna(sales["Model"])
    return {
        "Sales Data": sales,
        "Current Inventory": inputs["Current Inventory"],
        "Purchase Orders": inputs["Purchase Orders"],
        "OptionsData": options,
    }


@st.cache_resource(show_spinner="Merging daily and historical files...")
def load_raw_folder_data(
    folder: str,
    manifest: tuple[tuple[str, int, int], ...],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    notes: list[str] = []

    placeholders = write_engine_placeholders()
    try:
        problems = workbook_problems(raw_data_path)
        if problems:
            raise RuntimeError(
                "These spreadsheet files under Raw_data cannot be read:\n\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )
        sources, source_names = forecast_engine.read_raw_data_folder(folder)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"forecast_engine.read_raw_data_folder failed for '{folder}': {e}") from e
    finally:
        for path in placeholders:
            try:
                path.unlink()
            except OSError:
                pass
    source_names = [name for name in source_names if Path(name).name not in ENGINE_PLACEHOLDERS]

    for key in ("Sales Raw", "Options Data"):
        if key not in sources:
            raise KeyError(f"forecast_engine.read_raw_data_folder did not return '{key}'.")

    inventory_file = newest_file(inventory_folder_path)
    if inventory_file is not None:
        try:
            sources["Current Inventory"] = read_tabular(inventory_file)
        except Exception as e:
            raise RuntimeError(f"Could not read inventory export '{inventory_file.name}': {e}") from e
        source_names.append(str(inventory_file.relative_to(raw_data_path)))
    sources["Current Inventory"] = standardise_inventory(sources.get("Current Inventory"))
    if sources["Current Inventory"].empty:
        notes.append(
            "No inventory export was found under Raw_data/inventory (or Raw_data/Inventory_report.xlsx), "
            "so stock on hand is reported as zero."
        )

    po_file = newest_file(podash_folder_path)
    if po_file is not None:
        try:
            sources["Purchase Orders"] = read_tabular(po_file)
        except Exception as e:
            raise RuntimeError(f"Could not read purchase order export '{po_file.name}': {e}") from e
        source_names.append(str(po_file.relative_to(raw_data_path)))
    sources["Purchase Orders"] = standardise_purchase_orders(sources.get("Purchase Orders"))
    if sources["Purchase Orders"].empty:
        notes.append(
            "No purchase order export was found under Raw_data/PODash (or Raw_data/PODashboard.xlsx), "
            "so open PO quantities are reported as zero."
        )

    formatted_sales = clean_sales_data(sources.pop("Sales Raw"), sources["Options Data"])
    forecast_inputs = {
        "OptionsData": sources["Options Data"],
        "Current Inventory": sources["Current Inventory"],
        "Purchase Orders": sources["Purchase Orders"],
        "Sales Data": formatted_sales,
    }

    try:
        sheets = forecast_engine.prepare_forecast_sheets(forecast_inputs)
    except Exception as e:
        sheets = local_prepare_forecast_sheets(forecast_inputs)
        notes.append(
            f"forecast_engine.prepare_forecast_sheets failed ({type(e).__name__}: {e}); the app joined the sheets itself. "
            f"Inventory sheet — {column_summary(forecast_inputs['Current Inventory'])}. "
            f"Purchase orders — {column_summary(forecast_inputs['Purchase Orders'])}."
        )

    loan_file = newest_file(loan_folder_path)
    if loan_file is not None:
        try:
            sheets["Loans"] = read_tabular(loan_file)
            source_names.append(str(loan_file.relative_to(raw_data_path)))
        except Exception as e:
            sheets["Loans"] = pd.DataFrame()
            notes.append(f"Could not read the loan file '{loan_file.name}': {e}")
    else:
        sheets["Loans"] = pd.DataFrame()
    token = f"{len(manifest)}:{hash(manifest) & 0xFFFFFFFFFFFF:x}"
    return tag_frames(sheets, token), source_names, notes


def load_prepared_data():
    try:
        sheets, names, notes = load_raw_folder_data(str(raw_data_path), raw_folder_manifest(raw_data_path))
    except Exception as e:
        st.error(f"Error loading Raw_data: {e}")
        return None, []
    for note in notes:
        st.warning(note)
    return sheets, names


# =============================================================================
# Engine calls with app-side fallbacks
# =============================================================================

FILTER_COLUMNS = {
    "Location": "City",
    "Sales Channel": "Sales Channel Net",
    "Classification": "OptionsData.Fourth Product Group",
    "Supplier": "Supplier Brand",
    "Class": "OptionsData.Class",
    "Model": "Model",
    "SKU": "SKU Key",
}


def engine_call(name: str, fallback, *args):
    func = getattr(forecast_engine, name, None)
    if func is None:
        return fallback(*args), f"forecast_engine has no {name}; the app calculated it instead."
    try:
        return func(*args), None
    except Exception as e:
        return fallback(*args), f"forecast_engine.{name} failed ({type(e).__name__}: {e}); the app calculated it instead."


def local_filter_sales(sales: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = sales
    for key, value in filters.items():
        column = FILTER_COLUMNS.get(key)
        if value == "TOTAL" or column not in out.columns:
            continue
        text = out[column].astype("string").str.strip()
        exact = text.eq(value)
        out = out.loc[exact] if exact.any() else out.loc[text.str.contains(value, case=False, regex=False, na=False)]
    return out


def _filter_by_sku(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    sku = filters.get("SKU", "TOTAL")
    if sku != "TOTAL" and "Stock Code" in frame.columns:
        return frame.loc[frame["Stock Code"].map(sku_text).eq(sku_text(sku))]
    return frame


def local_inventory_totals(inventory: pd.DataFrame, filters: dict) -> tuple[float, float]:
    frame = _filter_by_sku(inventory, filters)
    on_hand = to_number(frame["On Hand"]).sum() if "On Hand" in frame.columns else 0.0
    free = to_number(frame["Free Stock"]).sum() if "Free Stock" in frame.columns else on_hand
    return float(on_hand), float(free)


def local_monthly_purchase_orders(purchase_orders: pd.DataFrame, filters: dict) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Date", "Open PO Qty"])
    if purchase_orders is None or purchase_orders.empty or not {"Date", "Open PO Qty"} <= set(purchase_orders.columns):
        return empty
    frame = _filter_by_sku(purchase_orders, filters)
    dates = parse_dates(frame["Date"])
    quantities = to_number(frame["Open PO Qty"])
    monthly = quantities.groupby(dates.dt.to_period("M")).sum()
    if monthly.empty:
        return empty
    return pd.DataFrame({"Date": monthly.index.to_timestamp(), "Open PO Qty": monthly.to_numpy()})


def local_yearly_monthly_profile(sales: pd.DataFrame, _forecast, metric: str, start: pd.Timestamp) -> pd.DataFrame:
    dated = sales.dropna(subset=["Date"])
    dated = dated.loc[dated["Date"] >= start]
    if dated.empty or metric not in dated.columns:
        return pd.DataFrame({"Month": MONTHS})
    grouped = dated.groupby([dated["Date"].dt.month, dated["Date"].dt.year])[metric]
    profile = (grouped.mean() if metric == "GP%" else grouped.sum()).unstack()
    profile.index = [MONTH_NUMBER_TO_NAME[m] for m in profile.index]
    profile = profile.reindex(MONTHS)
    profile.columns.name = None
    profile.index.name = "Month"
    return profile.reset_index()


# =============================================================================
# Forecast math
# =============================================================================

SEASON = 12
GROWTH_FLOOR, GROWTH_CAP = 0.5, 1.2
DEFAULT_GROWTH = 1.05


def monthly_series(sales: pd.DataFrame, metric: str, before: pd.Timestamp | None = None) -> pd.Series:
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
    n = len(y)
    level = sum(y[:SEASON]) / SEASON
    trend = (sum(y[SEASON : 2 * SEASON]) / SEASON - level) / SEASON
    seasonal = [value - level for value in y[:SEASON]]
    sse = 0.0
    for t in range(n):
        index = t % SEASON
        s = seasonal[index]
        predicted = level + phi * trend + s
        if t >= SEASON:
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
    this_year: int,
    last_complete: pd.Period,
    yoy_weight: float,
    bias: float,
    clip_at_zero: bool,
) -> pd.Series:
    next_start = pd.Period(f"{this_year + 1}-01", freq="M")
    as_of_period = last_complete + 1
    months_ahead = (next_start - as_of_period).n
    horizon = months_ahead + SEASON

    forecast_series = blended_forecast(
        history=actual,
        start=as_of_period,
        horizon=horizon,
        yoy_weight=yoy_weight,
        bias=bias,
        clip_at_zero=clip_at_zero,
    )

    next_year_periods = pd.period_range(next_start, periods=SEASON, freq="M")
    return forecast_series.reindex(next_year_periods)


def rolling_wmape(
    actual: pd.Series, last_complete: pd.Period, yoy_weight: float, bias: float, clip_at_zero: bool
) -> tuple[float | None, pd.PeriodIndex]:
    window = pd.period_range(last_complete - (SEASON - 1), periods=SEASON, freq="M")
    predicted = blended_forecast(actual, window[0], SEASON, yoy_weight, bias, clip_at_zero)
    observed = actual.reindex(window)
    valid = observed.notna() & observed.ne(0) & predicted.notna()
    if valid.empty:
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
    history_from_year: int = 2024,
    today: pd.Timestamp | None = None,
) -> dict:
    today = (today or pd.Timestamp.today()).normalize()
    current_period = today.to_period("M")
    current_month_start = current_period.to_timestamp()
    last_complete = current_period - 1
    this_year = today.year
    clip_at_zero = metric != "GP%"

    actual = monthly_series(selected_sales, metric, before=current_month_start)

    this_start = pd.Period(f"{this_year}-01", freq="M")
    forecast_this = blended_forecast(actual, this_start, SEASON, yoy_weight, bias, clip_at_zero)
    forecast_next = next_year_forecast(actual, this_year, last_complete, yoy_weight, bias, clip_at_zero)

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
        "last_complete": last_complete,
    }


def window_label(window: pd.PeriodIndex) -> str:
    return f"{window[0].strftime('%b %Y')} – {window[-1].strftime('%b %Y')}"


# =============================================================================
# Year view
# =============================================================================

def fiscal_year_of(period: pd.Period) -> int:
    return period.year + 1 if period.month >= 7 else period.year


def view_year(period: pd.Period, fiscal: bool) -> int:
    return fiscal_year_of(period) if fiscal else period.year


def view_year_periods(year: int, fiscal: bool) -> pd.PeriodIndex:
    start = f"{year - 1}-07" if fiscal else f"{year}-01"
    return pd.period_range(start, periods=SEASON, freq="M")


def view_label(year: int, fiscal: bool) -> str:
    return f"FY{year}" if fiscal else str(year)


def label_year_number(label: str) -> int:
    return int(re.sub(r"\D", "", str(label)))


def fiscal_profile(profile: pd.DataFrame, first_fiscal_year: int) -> pd.DataFrame:
    long = profile.melt(id_vars="Month", var_name="Year", value_name="Value").dropna(subset=["Value"])
    if long.empty:
        return pd.DataFrame({"Month": FY_MONTHS})
    month_number = long["Month"].map(MONTH_NAME_TO_NUMBER)
    calendar_year = long["Year"].astype(int)
    long = long.assign(FY=np.where(month_number >= 7, calendar_year + 1, calendar_year))
    long = long.loc[long["FY"] >= first_fiscal_year]
    pivot = long.pivot(index="Month", columns="FY", values="Value").reindex(FY_MONTHS)
    pivot.columns.name = None
    return pivot.reset_index()


# =============================================================================
# Inventory analysis
# =============================================================================

STATUS_ORDER = ["Dead stock", "Slow moving", "Stock-out risk", "Healthy", "Allocated", "Out of stock", "Inactive"]

SKU_TYPE_RULES = [
    ("Legacy showroom display (*SHOW*)", lambda code: "SHOW" in code),
    ("Clearance product (LEX*)", lambda code: code.startswith("LEX")),
    ("Consignment (C*)", lambda code: code.startswith("C")),
    ("Standard (1*)", lambda code: code.startswith("1")),
    ("Part (4*)", lambda code: code.startswith("4")),
    ("Project (5*)", lambda code: code.startswith("5")),
    ("Bundled (6*)", lambda code: code.startswith("6")),
]
SKU_TYPES = [label for label, _ in SKU_TYPE_RULES] + ["Other"]


def classify_sku(code) -> str:
    text = "" if pd.isna(code) else str(code).strip().upper()
    if not text:
        return "Other"
    for label, matches in SKU_TYPE_RULES:
        if matches(text):
            return label
    return "Other"


def sku_type_rules_table() -> pd.DataFrame:
    patterns = ["contains SHOW", "starts with LEX", "starts with C", "starts with 1", "starts with 4", "starts with 5", "starts with 6", "anything else"]
    return pd.DataFrame({"Order": range(1, len(SKU_TYPES) + 1), "SKU type": SKU_TYPES, "Rule": patterns})


SITE_CATEGORIES = ["Showroom", "Warehouse", "Clearance", "Loan", "Ignored", "Unmapped"]
ASSIGNABLE_CATEGORIES = ["Showroom", "Warehouse", "Clearance", "Loan", "Ignored"]
SELLABLE_CATEGORIES = {"Warehouse", "Clearance"}


def category_for(code: str, site_mapping: dict) -> str:
    if code in site_mapping:
        return site_mapping[code]
    if "LOAN" in code:
        return "Loan"
    return "Unmapped"
CATEGORY_ORDER = {category: index for index, category in enumerate(SITE_CATEGORIES)}
DEFAULT_SITE_MAPPING = {
    "RSYD": "Showroom", "RMEL": "Showroom", "RPER": "Showroom", "RBRI": "Showroom", "ROXF": "Showroom",
    "W1SY": "Warehouse", "W3ME": "Warehouse", "W4PE": "Warehouse", "W5BR": "Warehouse",
    "XITS": "Clearance", "XITM": "Clearance", "XITP": "Clearance", "XITB": "Clearance",
}
COST_SOURCES = {
    "auto": "Automatic (inventory export → options data → sales)",
    "inventory": "Inventory export cost column",
    "options": "Options data cost column",
    "sales": "Derived from sales (COGS ÷ units sold in the window)",
}
MAPPING_FILE = base_dir / "backend_mapping.json"


def normalise_site_code(value) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def load_backend_mapping_file() -> dict:
    try:
        return json.loads(MAPPING_FILE.read_text(encoding="utf-8")) if MAPPING_FILE.exists() else {}
    except (OSError, ValueError):
        return {}


def save_backend_mapping_file(payload: dict) -> None:
    MAPPING_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def init_backend_state() -> None:
    if "site_mapping" in st.session_state:
        return
    saved = load_backend_mapping_file()
    site_mapping = saved.get("site_mapping") or DEFAULT_SITE_MAPPING
    st.session_state["site_mapping"] = {
        normalise_site_code(code): (category if category in SITE_CATEGORIES else "Unmapped")
        for code, category in site_mapping.items()
        if normalise_site_code(code)
    }
    st.session_state["inventory_column_mapping"] = saved.get("column_mapping", {})
    st.session_state["loan_column_mapping"] = saved.get("loan_column_mapping", {})
    st.session_state["cost_source"] = saved.get("cost_source", "auto") if saved.get("cost_source", "auto") in COST_SOURCES else "auto"
    st.session_state["site_editor_version"] = 0


def current_site_mapping() -> dict[str, str]:
    init_backend_state()
    return dict(st.session_state["site_mapping"])


def resolve_inventory_mapping(inventory: pd.DataFrame) -> dict[str, str | None]:
    init_backend_state()
    mapping = detect_inventory_columns(inventory)
    for role, choice in st.session_state["inventory_column_mapping"].items():
        if choice == "(none)":
            mapping[role] = None
        elif choice in inventory.columns:
            mapping[role] = choice
    return mapping


def site_mapping_frame(site_mapping: dict[str, str], site_units: pd.Series) -> pd.DataFrame:
    codes = sorted(
        set(site_mapping) | {code for code in site_units.index if code},
        key=lambda code: (CATEGORY_ORDER[category_for(code, site_mapping)], code),
    )
    return pd.DataFrame(
        [
            {
                "Site code": code,
                "Category": category_for(code, site_mapping),
                "Available to sell": category_for(code, site_mapping) in SELLABLE_CATEGORIES,
                "In inventory export": code in site_units.index,
                "Units on hand": float(site_units.get(code, 0.0)),
            }
            for code in codes
        ]
    )


def apply_site_editor_changes() -> None:
    key = f"backend_site_editor_{st.session_state['site_editor_version']}"
    state = st.session_state.get(key, {})
    frame = st.session_state["site_editor_frame"].copy()
    for row, changes in state.get("edited_rows", {}).items():
        for column, value in changes.items():
            frame.loc[int(row), column] = value
    if state.get("added_rows"):
        frame = pd.concat([frame, pd.DataFrame(state["added_rows"])], ignore_index=True)
    if state.get("deleted_rows"):
        frame = frame.drop(index=[int(row) for row in state["deleted_rows"]])
    mapping = {}
    for code, category in zip(frame.get("Site code", []), frame.get("Category", [])):
        code = normalise_site_code(code)
        if code:
            mapping[code] = category if category in SITE_CATEGORIES else "Unmapped"
    st.session_state["site_mapping"] = mapping
    st.session_state["site_editor_version"] += 1


def units_by_site(inventory: pd.DataFrame, mapping: dict) -> pd.Series:
    if not mapping.get("site") or not mapping.get("on_hand"):
        return pd.Series(dtype=float)
    codes = map_unique(inventory[mapping["site"]], normalise_site_code)
    units = to_number(inventory[mapping["on_hand"]]).fillna(0.0)
    return units.groupby(codes).sum()


def company_wide_totals(inventory: pd.DataFrame, mapping: dict, site_mapping: dict, cost_by_sku: pd.Series) -> dict:
    frame = pd.DataFrame({"SKU": map_unique(inventory[mapping["sku"]], sku_text)})
    frame["On hand"] = to_number(inventory[mapping["on_hand"]]).fillna(0.0)
    frame["Free"] = to_number(inventory[mapping["free"]]).fillna(0.0) if mapping.get("free") else frame["On hand"]
    if mapping.get("site"):
        sites = map_unique(inventory[mapping["site"]], normalise_site_code)
        frame["Category"] = map_unique(sites, lambda code: category_for(code, site_mapping))
    else:
        frame["Category"] = "Warehouse"
    row_cost = to_number(inventory[mapping["cost"]]) if mapping.get("cost") else pd.Series(np.nan, index=frame.index)
    frame["Unit cost"] = row_cost.where(row_cost.notna() & (row_cost > 0), frame["SKU"].map(cost_by_sku))
    frame = frame.loc[frame["SKU"].ne("")]
    frame["Allocated"] = (frame["On hand"] - frame["Free"]).clip(lower=0.0)
    frame["Value"] = frame["On hand"] * frame["Unit cost"]
    frame["Free value"] = frame["Free"] * frame["Unit cost"]
    frame["Allocated value"] = frame["Allocated"] * frame["Unit cost"]
    by_category = (
        frame.groupby("Category")[["On hand", "Free", "Allocated", "Value", "Free value", "Allocated value"]]
        .sum()
        .reindex(SITE_CATEGORIES)
        .fillna(0.0)
    )
    stocked = frame["On hand"] > 0
    return {
        "units": float(frame["On hand"].sum()),
        "free_units": float(frame["Free"].sum()),
        "allocated_units": float(frame["Allocated"].sum()),
        "value": float(frame["Value"].sum(skipna=True)),
        "free_value": float(frame["Free value"].sum(skipna=True)),
        "allocated_value": float(frame["Allocated value"].sum(skipna=True)),
        "showroom_value": float(by_category.loc["Showroom", "Value"]),
        "loan_value": float(by_category.loc["Loan", "Value"]),
        "loan_units": float(by_category.loc["Loan", "On hand"]),
        "by_category": by_category,
        "costed_share": float(frame.loc[stocked, "Unit cost"].notna().mean()) if stocked.any() else 0.0,
    }


LOAN_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "sku": SKU_HEADERS,
    "customer": ["customer name", "customer", "client name", "client", "account name", "account", "debtor", "borrower", "name"],
    "qty": ["qty", "quantity", "units", "qty out", "quantity out", "loan qty"],
    "from_site": ["from location", "from site", "from warehouse", "source location", "origin", "from"],
    "to_site": ["to location", "to site", "to warehouse", "destination", "to"],
    "date": ["date", "loan date", "transfer date", "document date", "issued"],
    "due": ["due date", "return date", "expected return", "return by", "due"],
    "document": ["document", "document no", "doc no", "reference", "ref", "transfer no", "loan no", "number"],
    "status": ["status", "state", "stage"],
    "returned_qty": ["returned qty", "qty returned", "returned", "qty in", "received qty"],
    "description": INVENTORY_COLUMN_CANDIDATES["description"],
}
LOAN_ROLE_LABELS = {
    "sku": "SKU column",
    "customer": "Customer name column",
    "qty": "Quantity column",
    "from_site": "From location column",
    "to_site": "To location column",
    "date": "Date column",
    "due": "Due / return date column",
    "document": "Document number column",
    "status": "Status column",
    "returned_qty": "Returned quantity column",
    "description": "Description column",
}
CLOSED_STATUS_WORDS = ("return", "closed", "complete", "cancel", "received")


def detect_loan_columns(loans: pd.DataFrame) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    used: list[str] = []
    for role, candidates in LOAN_COLUMN_CANDIDATES.items():
        col = detect_column(loans.columns, candidates, exclude=tuple(used))
        found[role] = col
        if col:
            used.append(col)
    return found


def resolve_loan_mapping(loans: pd.DataFrame) -> dict[str, str | None]:
    init_backend_state()
    mapping = detect_loan_columns(loans)
    for role, choice in st.session_state.get("loan_column_mapping", {}).items():
        if choice == "(none)":
            mapping[role] = None
        elif choice in loans.columns:
            mapping[role] = choice
    return mapping


def classify_loan_lines(
    loans: pd.DataFrame, mapping: dict, site_mapping: dict, cost_by_sku: pd.Series, today: pd.Timestamp
) -> pd.DataFrame:
    columns = ["Document", "Date", "Due", "Customer", "SKU", "Description", "Qty", "From site", "To site",
               "From category", "Type", "Days out", "Avg option cost", "Value"]
    if loans is None or loans.empty or not mapping.get("sku") or not mapping.get("qty"):
        return pd.DataFrame(columns=columns)

    def text(role: str) -> pd.Series:
        col = mapping.get(role)
        if not col:
            return pd.Series("", index=loans.index, dtype=object)
        values = loans[col].astype(str).str.strip()
        return values.where(loans[col].notna() & ~values.str.lower().isin(["", "nan", "none", "null"]), "")

    frame = pd.DataFrame({"SKU": map_unique(loans[mapping["sku"]], sku_text)})
    frame["Customer"] = text("customer")
    frame["Description"] = text("description")
    frame["Document"] = text("document")
    frame["From site"] = map_unique(text("from_site"), normalise_site_code)
    frame["To site"] = map_unique(text("to_site"), normalise_site_code)
    frame["Qty"] = to_number(loans[mapping["qty"]]).fillna(0.0)
    if mapping.get("returned_qty"):
        frame["Qty"] = (frame["Qty"] - to_number(loans[mapping["returned_qty"]]).fillna(0.0)).clip(lower=0.0)
    frame["Date"] = parse_dates(loans[mapping["date"]]) if mapping.get("date") else pd.NaT
    frame["Due"] = parse_dates(loans[mapping["due"]]) if mapping.get("due") else pd.NaT
    status = text("status").str.lower()
    closed = status.map(lambda value: any(word in value for word in CLOSED_STATUS_WORDS))
    frame = frame.loc[frame["SKU"].ne("") & (frame["Qty"] > 0) & ~closed].copy()

    frame["From category"] = map_unique(frame["From site"], lambda code: category_for(code, site_mapping) if code else "")
    is_loan = frame["Customer"].ne("") | frame["From category"].eq("Loan")
    frame["Type"] = np.where(is_loan, "Loan", "Transfer")
    frame["Days out"] = (today - pd.to_datetime(frame["Date"], errors="coerce")).dt.days
    frame["Avg option cost"] = frame["SKU"].map(cost_by_sku)
    frame["Value"] = frame["Qty"] * frame["Avg option cost"]
    return frame[columns].sort_values(["Type", "Value"], ascending=[True, False]).reset_index(drop=True)


GAUGE_START, GAUGE_END = -2 * np.pi / 3, 2 * np.pi / 3


def gauge_band_colour(value: float, target: float, tolerance: float) -> tuple[str, str]:
    if target <= 0:
        return "No target", "#9aa0a6"
    deviation = abs(value / target - 1)
    if deviation <= tolerance:
        return "On target", "#009e73"
    if deviation <= 3 * tolerance:
        return "Near target", "#e69f00"
    return "Off target", "#d55e00"


def gauge_chart(value: float, target: float, tolerance: float = 0.05, scale_max: float | None = None) -> alt.LayerChart:
    scale_max = scale_max or max(target * 1.5, value * 1.1, 1.0)
    span = GAUGE_END - GAUGE_START

    def angle(amount: float) -> float:
        return GAUGE_START + span * min(max(amount / scale_max, 0.0), 1.0)

    bands = []
    cuts = [0.0, target * (1 - 3 * tolerance), target * (1 - tolerance), target * (1 + tolerance), target * (1 + 3 * tolerance), scale_max]
    colours = ["#dcdcdc", "#f5d58a", "#009e73", "#f5d58a", "#dcdcdc"]
    for lower, upper, colour in zip(cuts[:-1], cuts[1:], colours):
        lower, upper = max(0.0, min(lower, scale_max)), max(0.0, min(upper, scale_max))
        if upper > lower:
            bands.append({"start": angle(lower), "end": angle(upper), "colour": colour})
    band_frame = pd.DataFrame(bands)
    label, colour = gauge_band_colour(value, target, tolerance)
    value_frame = pd.DataFrame({"start": [GAUGE_START], "end": [angle(value)], "colour": [colour]})
    tick = pd.DataFrame({"start": [angle(target) - 0.008], "end": [angle(target) + 0.008]})
    centre = pd.DataFrame(
        {
            "value": [f"${value / 1e6:,.2f}M"],
            "detail": [f"{value / target - 1:+.1%} vs ${target / 1e6:,.1f}M target" if target > 0 else "no target set"],
            "label": [label],
        }
    )

    ring = alt.Chart(band_frame).mark_arc(innerRadius=72, outerRadius=84).encode(
        theta=alt.Theta("start:Q", scale=None), theta2="end:Q", color=alt.Color("colour:N", scale=None),
    )
    track = alt.Chart(pd.DataFrame({"start": [GAUGE_START], "end": [GAUGE_END]})).mark_arc(
        innerRadius=56, outerRadius=66, color="#000000", opacity=0.07
    ).encode(theta=alt.Theta("start:Q", scale=None), theta2="end:Q")
    fill = alt.Chart(value_frame).mark_arc(innerRadius=56, outerRadius=66, cornerRadius=4).encode(
        theta=alt.Theta("start:Q", scale=None), theta2="end:Q", color=alt.Color("colour:N", scale=None),
    )
    marker = alt.Chart(tick).mark_arc(innerRadius=52, outerRadius=88, color="#31333F").encode(
        theta=alt.Theta("start:Q", scale=None), theta2="end:Q"
    )
    headline = alt.Chart(centre).mark_text(fontSize=21, fontWeight="bold", dy=-8).encode(text="value:N")
    subline = alt.Chart(centre).mark_text(fontSize=10.5, dy=11, opacity=0.75).encode(text="detail:N")
    status = alt.Chart(centre).mark_text(fontSize=11, dy=27, fontWeight="bold", color=colour).encode(text="label:N")
    return alt.layer(ring, track, fill, marker, headline, subline, status).properties(height=196).configure_view(strokeWidth=0)


@st.cache_data(show_spinner=False, max_entries=16, **BY_TOKEN)
def inventory_by_sku(inventory: pd.DataFrame, mapping: dict, site_mapping: dict) -> pd.DataFrame:
    frame = pd.DataFrame({"SKU": map_unique(inventory[mapping["sku"]], sku_text)})
    frame["On hand"] = to_number(inventory[mapping["on_hand"]]).fillna(0.0)
    frame["Free stock"] = (
        to_number(inventory[mapping["free"]]).fillna(0.0) if mapping.get("free") else frame["On hand"]
    )
    if mapping.get("site"):
        frame["Site"] = map_unique(inventory[mapping["site"]], normalise_site_code)
        frame["Category"] = map_unique(frame["Site"], lambda code: category_for(code, site_mapping))
    else:
        frame["Site"] = ""
        frame["Category"] = "Warehouse"
    frame["Sellable"] = frame["Category"].isin(SELLABLE_CATEGORIES)
    if mapping.get("cost"):
        frame["Unit cost"] = to_number(inventory[mapping["cost"]])
    if mapping.get("description"):
        frame["Inventory description"] = inventory[mapping["description"]].astype(str).str.strip()
    frame = frame.loc[frame["SKU"].ne("")]

    skus = pd.Index(sorted(frame["SKU"].unique()), name="SKU")
    sellable = frame.loc[frame["Sellable"]]
    out = pd.DataFrame(index=skus)
    out["On hand"] = sellable.groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    out["Free stock"] = sellable.groupby("SKU")["Free stock"].sum().reindex(skus).fillna(0.0)
    out["On hand (all sites)"] = frame.groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    out["Showroom units"] = (
        frame.loc[frame["Category"].eq("Showroom")].groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    )
    out["Unmapped units"] = (
        frame.loc[frame["Category"].eq("Unmapped")].groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    )
    out["Ignored units"] = (
        frame.loc[frame["Category"].eq("Ignored")].groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    )
    out["Loan units"] = (
        frame.loc[frame["Category"].eq("Loan")].groupby("SKU")["On hand"].sum().reindex(skus).fillna(0.0)
    )
    out["Sites with stock"] = (
        sellable.loc[sellable["On hand"] > 0].groupby("SKU")["Site"].nunique().reindex(skus).fillna(0).astype(int)
    )
    if "Unit cost" in frame.columns:
        costed = frame.loc[frame["Unit cost"].notna() & (frame["Unit cost"] > 0)].copy()
        costed["Weight"] = costed["On hand"].clip(lower=0)
        costed["Weighted"] = costed["Unit cost"] * costed["Weight"]
        grouped = costed.groupby("SKU")
        weighted = grouped["Weighted"].sum() / grouped["Weight"].sum()
        out["Inventory unit cost"] = weighted.where(grouped["Weight"].sum() > 0, grouped["Unit cost"].mean()).reindex(skus)
    if "Inventory description" in frame.columns:
        out["Inventory description"] = frame.groupby("SKU")["Inventory description"].first().reindex(skus)
    return out.reset_index()


@st.cache_data(show_spinner=False, max_entries=16, **BY_TOKEN)
def sku_movement(sales: pd.DataFrame, window_start: str, window_end: str) -> pd.DataFrame:
    dated = sales.dropna(subset=["Date"]).copy()
    key_column = "SKU Key" if "SKU Key" in dated.columns else "SKU"
    dated["SKU"] = map_unique(dated[key_column], sku_text)
    dated = dated.loc[dated["SKU"].ne("")]
    if dated.empty:
        return pd.DataFrame(columns=["SKU", "Units sold", "Sales value", "COGS", "Months with sales", "Last sale", "First sale"])

    period = dated["Date"].dt.to_period("M")
    start, end = pd.Period(window_start, freq="M"), pd.Period(window_end, freq="M")
    in_window = dated.loc[(period >= start) & (period <= end)].copy()
    in_window["Period"] = in_window["Date"].dt.to_period("M")

    aggregations = {"Units sold": ("Qty", "sum"), "Months with sales": ("Period", "nunique")}
    if "Sale" in in_window.columns:
        aggregations["Sales value"] = ("Sale", "sum")
    if "Cost" in in_window.columns:
        aggregations["COGS"] = ("Cost", "sum")
    window = in_window.groupby("SKU").agg(**aggregations)
    history = dated.groupby("SKU").agg(**{"Last sale": ("Date", "max"), "First sale": ("Date", "min")})

    attribute_map = {
        "Description": "Description",
        "OptionsData.Supplier": "Supplier",
        "OptionsData.Class": "Class",
        "OptionsData.Fourth Product Group": "Product group",
    }
    present = {src: dst for src, dst in attribute_map.items() if src in dated.columns}
    attributes = (
        dated.sort_values("Date").groupby("SKU")[list(present)].last().rename(columns=present)
        if present
        else pd.DataFrame(index=history.index)
    )
    return window.join(history, how="outer").join(attributes, how="left").reset_index()


@st.cache_data(show_spinner=False, max_entries=4, **BY_TOKEN)
def option_attributes(options: pd.DataFrame | None) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["SKU", "Description", "Supplier", "Class", "Product group", "Option unit cost"])
    if options is None or options.empty:
        return empty
    sku_col = detect_column(options.columns, ["stock code", "sku", "sku key"])
    if sku_col is None:
        return empty
    frame = pd.DataFrame({"SKU": map_unique(options[sku_col], sku_text)})
    for role, candidates in {
        "Description": INVENTORY_COLUMN_CANDIDATES["description"],
        "Supplier": ["supplier", "supplier brand", "brand", "vendor"],
        "Class": ["class"],
        "Product group": ["fourth product group", "product group", "category"],
    }.items():
        col = detect_column(options.columns, candidates, exclude=(sku_col,))
        frame[role] = options[col].astype(str).str.strip() if col else pd.NA
    cost_col = detect_column(options.columns, INVENTORY_COLUMN_CANDIDATES["cost"], exclude=(sku_col,))
    frame["Option unit cost"] = to_number(options[cost_col]) if cost_col else np.nan
    return frame.loc[frame["SKU"].ne("")].drop_duplicates(subset=["SKU"])


@st.cache_data(show_spinner=False, max_entries=4, **BY_TOKEN)
def open_po_by_sku(purchase_orders: pd.DataFrame | None) -> pd.Series:
    if purchase_orders is None or purchase_orders.empty:
        return pd.Series(dtype=float)
    sku_col = detect_column(purchase_orders.columns, SKU_HEADERS)
    qty_col = detect_column(purchase_orders.columns, OPEN_PO_HEADERS)
    if sku_col is None or qty_col is None:
        return pd.Series(dtype=float)
    keyed = pd.DataFrame(
        {"SKU": map_unique(purchase_orders[sku_col], sku_text), "Open PO qty": to_number(purchase_orders[qty_col]).fillna(0.0)}
    )
    return keyed.loc[keyed["SKU"].ne("")].groupby("SKU")["Open PO qty"].sum()


def resolve_avg_option_cost(master: pd.DataFrame, derived: pd.Series, cost_source: str) -> tuple[pd.Series, pd.Series]:
    candidates = {
        "inventory": master["Inventory unit cost"] if "Inventory unit cost" in master.columns else pd.Series(np.nan, index=master.index),
        "options": master["Option unit cost"] if "Option unit cost" in master.columns else pd.Series(np.nan, index=master.index),
        "sales": derived,
    }
    order = ["inventory", "options", "sales"]
    if cost_source in candidates:
        order = [cost_source] + [name for name in order if name != cost_source]
    cost = pd.Series(np.nan, index=master.index, dtype=float)
    source = pd.Series("none", index=master.index, dtype=object)
    for name in order:
        values = pd.to_numeric(candidates[name], errors="coerce")
        usable = cost.isna() & values.notna() & (values > 0)
        cost = cost.where(~usable, values)
        source = source.where(~usable, name)
    return cost, source


def build_inventory_table(
    inventory_sku: pd.DataFrame,
    movement: pd.DataFrame,
    attributes: pd.DataFrame,
    open_po: pd.Series,
    window_months: int,
    window_days: int,
    today: pd.Timestamp,
    dead_days: int,
    slow_dos: int,
    risk_dos: int,
    cost_source: str = "auto",
) -> pd.DataFrame:
    master = inventory_sku.merge(movement, on="SKU", how="outer")
    master = master.merge(attributes, on="SKU", how="left", suffixes=("", " (options)"))
    for column in ["Description", "Supplier", "Class", "Product group"]:
        option_column = f"{column} (options)"
        if option_column in master.columns:
            master[column] = master[column].where(master[column].notna(), master[option_column]) if column in master.columns else master[option_column]
            master = master.drop(columns=[option_column])
        elif column not in master.columns:
            master[column] = pd.NA
    if "Inventory description" in master.columns:
        master["Description"] = master["Description"].where(master["Description"].notna(), master["Inventory description"])
        master = master.drop(columns=["Inventory description"])
    master["Description"] = master["Description"].fillna("").astype(str)
    for column in ["Supplier", "Class", "Product group"]:
        master[column] = master[column].fillna("Unknown").astype(str).replace({"": "Unknown", "nan": "Unknown"})

    for column in ["On hand", "Free stock", "On hand (all sites)", "Showroom units", "Ignored units", "Unmapped units", "Loan units",
                   "Units sold", "Sales value", "COGS", "Months with sales"]:
        master[column] = pd.to_numeric(master[column], errors="coerce").fillna(0.0) if column in master.columns else 0.0
    if not open_po.empty:
        master["Open PO qty"] = master["SKU"].map(open_po).fillna(0.0)

    master["SKU type"] = map_unique(master["SKU"], classify_sku)
    master["Avg monthly units"] = master["Units sold"] / window_months
    daily_rate = master["Units sold"] / window_days
    with np.errstate(divide="ignore", invalid="ignore"):
        master["Days of supply"] = np.where(daily_rate > 0, master["On hand"] / daily_rate, np.inf)
        master["Days of supply (free)"] = np.where(daily_rate > 0, master["Free stock"] / daily_rate, np.inf)
        master["Turns (annualised)"] = np.where(
            master["On hand"] > 0, (master["Units sold"] * 12 / window_months) / master["On hand"], np.nan
        )
        derived_cost = pd.Series(
            np.where(master["Units sold"] > 0, master["COGS"] / master["Units sold"], np.nan), index=master.index
        )
        if "Open PO qty" in master.columns:
            master["Days of supply incl. POs"] = np.where(
                daily_rate > 0, (master["On hand"] + master["Open PO qty"]) / daily_rate, np.inf
            )

    master["Avg option cost"], master["Cost source"] = resolve_avg_option_cost(master, derived_cost, cost_source)
    master["Total stock value"] = master["On hand"] * master["Avg option cost"]
    master["Free stock value"] = master["Free stock"] * master["Avg option cost"]
    master = master.drop(columns=[c for c in ("Inventory unit cost", "Option unit cost") if c in master.columns])

    last_sale = pd.to_datetime(master["Last sale"], errors="coerce")
    master["Last sale"] = last_sale
    master["Days since last sale"] = (today - last_sale).dt.days

    on_hand = master["On hand"]
    free = master["Free stock"]
    dos = master["Days of supply"]
    free_dos = master["Days of supply (free)"]
    selling = master["Units sold"] > 0
    allocated = (on_hand > 0) & (free <= 0)
    never_or_stale = last_sale.isna() | (master["Days since last sale"] > dead_days)
    master["Status"] = np.select(
        [
            (on_hand <= 0) & selling,
            on_hand <= 0,
            (on_hand > 0) & selling & ~never_or_stale & (free_dos < risk_dos),
            allocated,
            (on_hand > 0) & never_or_stale,
            (on_hand > 0) & (dos > slow_dos),
        ],
        ["Out of stock", "Inactive", "Stock-out risk", "Allocated", "Dead stock", "Slow moving"],
        default="Healthy",
    )
    master["Active"] = master["Status"].ne("Inactive")
    master["Dead units"] = np.where(master["Status"] == "Dead stock", on_hand, 0.0)
    master["Slow units"] = np.where(master["Status"] == "Slow moving", on_hand, 0.0)
    master["Allocated units"] = np.where(master["Status"] == "Allocated", on_hand, 0.0)
    master["Days of supply"] = master["Days of supply"].replace(np.inf, np.nan)
    master["Days of supply (free)"] = master["Days of supply (free)"].replace(np.inf, np.nan)
    if "Days of supply incl. POs" in master.columns:
        master["Days of supply incl. POs"] = master["Days of supply incl. POs"].replace(np.inf, np.nan)
    return master.sort_values("On hand", ascending=False).reset_index(drop=True)


def summarise_inventory(master: pd.DataFrame, group_column: str, window_months: int, window_days: int) -> pd.DataFrame:
    grouped = master.groupby(group_column).agg(
        **{
            "SKUs": ("SKU", "nunique"),
            "Active SKUs": ("Active", "sum"),
            "On hand": ("On hand", "sum"),
            "Free stock": ("Free stock", "sum"),
            "Showroom units": ("Showroom units", "sum"),
            "Units sold": ("Units sold", "sum"),
            "Sales value": ("Sales value", "sum"),
            "Total stock value": ("Total stock value", "sum"),
            "Free stock value": ("Free stock value", "sum"),
            "Dead units": ("Dead units", "sum"),
            "Slow units": ("Slow units", "sum"),
            "Allocated units": ("Allocated units", "sum"),
            "Stock-out risk SKUs": ("Status", lambda s: int((s == "Stock-out risk").sum())),
        }
    )
    grouped["Active SKUs"] = grouped["Active SKUs"].astype(int)
    daily_rate = grouped["Units sold"] / window_days
    with np.errstate(divide="ignore", invalid="ignore"):
        grouped["Days of supply"] = np.where(daily_rate > 0, grouped["On hand"] / daily_rate, np.nan)
        grouped["Turns (annualised)"] = np.where(
            grouped["On hand"] > 0, (grouped["Units sold"] * 12 / window_months) / grouped["On hand"], np.nan
        )
        grouped["Dead stock %"] = np.where(grouped["On hand"] > 0, grouped["Dead units"] / grouped["On hand"] * 100, np.nan)
    return grouped.sort_values("On hand", ascending=False).reset_index()


# =============================================================================
# Layout helpers
# =============================================================================

def metric_card(title: str, tiles: list[dict]) -> None:
    with st.container(border=True):
        st.markdown(f"<p class='card-title'>{title}</p>", unsafe_allow_html=True)
        columns = st.columns(len(tiles), gap="small")
        for column, tile in zip(columns, tiles):
            column.metric(**tile)


def chart_formats(metric: str) -> tuple[str, str, str]:
    if metric in ["Sale", "Cost", "GP $"]:
        return "$~s", "$,.0f", f"{metric.replace(' $', '')} ($)"
    if metric == "GP%":
        return ".1f", ".1f", "GP %"
    return "~s", ",.0f", "Qty (units)"


def quiet_axis(**overrides) -> alt.Axis:
    settings = dict(grid=False, ticks=False, domain=False, labelPadding=6)
    settings.update(overrides)
    return alt.Axis(**settings)


def value_axis(fmt: str, **overrides) -> alt.Axis:
    settings = dict(
        format=fmt, grid=True, gridOpacity=0.35, gridDash=[2, 3], tickCount=5,
        ticks=False, domain=False, titleFontWeight="normal", titlePadding=10,
    )
    settings.update(overrides)
    return alt.Axis(**settings)


def filter_selectbox(label: str, frame: pd.DataFrame, column: str, default: str = "TOTAL") -> str:
    values = column_choices(frame, column)
    if len(values) > 1500:
        starting_value = "" if default == "TOTAL" else default
        typed_value = st.text_input(
            f"{label} contains",
            value=starting_value,
            key=f"forecast_{label}",
        ).strip()
        return typed_value or "TOTAL"
    choices = ["TOTAL", *values]
    index = choices.index(default) if default in choices else 0
    return st.selectbox(label, choices, index=index, key=f"forecast_{label}")


def table_height(rows: int, cap: int = 420) -> int:
    return min(cap, int(38 + 35 * rows) + 3)


@st.fragment
def stock_health_charts(filtered: pd.DataFrame, chart_group: str) -> None:
    control_left, _, control_right = st.columns([5, 1, 2], gap="small", vertical_alignment="center")
    with control_left:
        measure_label = st.radio(
            "Measure", ["Units on hand", "SKUs", "Total stock value", "Free stock value"], index=0,
            horizontal=True, key="inventory_chart_measure", label_visibility="collapsed",
        )
    with control_right:
        show_share = st.toggle("Show as % of group", value=False, key="inventory_chart_normalize")
    measure = {"Units on hand": "units", "SKUs": "skus", "Total stock value": "total_value", "Free stock value": "free_value"}[measure_label]

    chart_left, chart_right = st.columns([2, 3], gap="small")
    with chart_left:
        st.markdown("<p class='card-title'>Stock health</p>", unsafe_allow_html=True)
        st.altair_chart(stock_health_breakdown(filtered, measure=measure), **FULL_WIDTH)
    with chart_right:
        st.markdown(f"<p class='card-title'>Stock health by {chart_group.lower()} · top 15</p>", unsafe_allow_html=True)
        st.altair_chart(
            stock_health_by_group(filtered, chart_group, measure=measure, top_n=15, normalize=show_share),
            **FULL_WIDTH,
        )


@st.fragment
def loans_section(loan_lines: pd.DataFrame) -> None:
    loans = loan_lines.loc[loan_lines["Type"].eq("Loan")]
    show_transfers = st.toggle("Include inter-site transfers", value=False, key="inventory_show_transfers")
    shown = loan_lines if show_transfers else loans
    loan_columns = ["Type", "Customer", "SKU", "Description", "Qty", "Avg option cost", "Value",
                    "Date", "Days out", "Due", "From site", "From category", "To site", "Document"]
    st.dataframe(
        shown[loan_columns],
        hide_index=True,
        column_config={
            "Qty": st.column_config.NumberColumn(format="%d"),
            "Avg option cost": st.column_config.NumberColumn(format="$%.2f"),
            "Value": st.column_config.NumberColumn(format="$%.0f"),
            "Date": st.column_config.DatetimeColumn(format="DD MMM YYYY"),
            "Due": st.column_config.DatetimeColumn(format="DD MMM YYYY"),
            "Days out": st.column_config.NumberColumn(format="%d"),
        },
        height=table_height(len(shown)),
        **FULL_WIDTH,
    )
    st.download_button(
        "Download CSV", shown[loan_columns].to_csv(index=False).encode("utf-8"),
        file_name="stock_on_loan.csv", mime="text/csv", key="inventory_download_loans",
    )
    if not loans.empty and loans["Customer"].ne("").any():
        by_customer = (
            loans.loc[loans["Customer"].ne("")]
            .groupby("Customer").agg(Lines=("SKU", "size"), Units=("Qty", "sum"), Value=("Value", "sum"), **{"Oldest (days)": ("Days out", "max")})
            .sort_values("Value", ascending=False).reset_index()
        )
        st.caption("By customer")
        st.dataframe(
            by_customer, hide_index=True,
            column_config={"Units": st.column_config.NumberColumn(format="%d"), "Value": st.column_config.NumberColumn(format="$%.0f"), "Oldest (days)": st.column_config.NumberColumn(format="%d")},
            height=table_height(len(by_customer), cap=300), **FULL_WIDTH,
        )


# =============================================================================
# Page header, sidebar and navigation
# =============================================================================

header_left, header_right = st.columns([6, 1], gap="small", vertical_alignment="center")
with header_left:
    st.title("📦 Forecast and Inventory Program")
with header_right:
    if st.button("🔄 Refresh data", **FULL_WIDTH):
        load_csv.clear()
        load_raw_folder_data.clear()
        st.rerun()

st.sidebar.markdown(
    "<div style='text-align: center; margin-top: -10px;'>"
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
                selected_values = st.multiselect(column, unique_values, default=[], key=f"{key_base}_select")
                if selected_values:
                    filtered = filtered[filtered[column].astype("string").str.strip().isin(selected_values)]
            else:
                search_value = st.text_input(f"{column} contains", key=f"{key_base}_text").strip()
                if search_value:
                    filtered = filtered[
                        filtered[column].astype("string").str.contains(search_value, case=False, na=False, regex=False)
                    ]

    return filtered


selected_tab = st.radio(
    "View",
    ["Forecast", "Inventory Analysis", "Backend Mapping", "Options Data", "Formatted Data", "Document Status"],
    horizontal=True,
    label_visibility="collapsed",
)

prepared_data = None
source_names = []

# =============================================================================
# Forecast tab
# =============================================================================
if selected_tab == "Forecast":
    prepared_data, source_names = load_prepared_data()

    if prepared_data is None:
        st.info(
            "Add inventory files under Raw_data/inventory, PO files under Raw_data/PODash, "
            "options_data.csv, and historical/weekly sales workbooks under Raw_data/Sales Data."
        )
    else:
        try:
            sheets = prepared_data
            sales_data = sheets["Sales Data"]

            with st.sidebar:
                st.header("Forecast filters")
                metric_options = ["Qty", "Sale", "Cost", "GP%", "GP $"]
                metric = st.selectbox("View", metric_options, index=0)

                st.markdown("**Sales**")
                selected_location = filter_selectbox("Location", sales_data, "City")
                selected_channel = filter_selectbox("Sales channel", sales_data, "Sales Channel Net")

                st.markdown("**Product**")
                selected_classification = filter_selectbox("Classification", sales_data, "OptionsData.Fourth Product Group")
                selected_supplier = filter_selectbox("Supplier", sales_data, "Supplier Brand")
                selected_class = filter_selectbox("Class", sales_data, "OptionsData.Class")
                selected_model = filter_selectbox("Model", sales_data, "Model")
                selected_sku = filter_selectbox("SKU", sales_data, "SKU Key")

                with st.expander("Forecast settings", expanded=False):
                    history_from_year = st.number_input(
                        "History from Year", min_value=2000, max_value=2100, value=2024, step=1,
                    )
                    forecast_from = pd.Timestamp(f"{history_from_year}-01-01")
                    bias = st.number_input("Forecast bias", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
                    yoy_weight = st.slider("YoY trend weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

                with st.expander("Target settings", expanded=False):
                    sale_target = st.number_input(
                        "Sale Target ($)", min_value=0.0, value=70000000.0, step=1000000.0,
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
            engine_notes: list[str] = []
            selected_sales, note = engine_call("filter_sales", local_filter_sales, sales_data, filters)
            engine_notes.append(note)
            selected_sales = selected_sales.copy()
            if metric == "GP%" and not selected_sales.empty and "GP%" in selected_sales.columns:
                selected_sales["GP%"] = selected_sales["GP%"].mean()

            result = build_forecasts(selected_sales, metric, yoy_weight, bias, history_from_year)
            this_year = result["this_year"]
            forecast_this_year = result["forecast_this"]
            forecast_next_year = result["forecast_next"]
            rolling_window = result["rolling_window"]
            actual_series = result["actual"]
            last_complete = result["last_complete"]

            purchase_months, note = engine_call(
                "monthly_purchase_orders", local_monthly_purchase_orders, sheets["Purchase Orders"], filters
            )
            engine_notes.append(note)
            (on_hand, free_stock), note = engine_call(
                "inventory_totals", local_inventory_totals, sheets["Current Inventory"], filters
            )
            engine_notes.append(note)

            if result["actual"].empty:
                st.warning("No completed months of dated sales match these selections.")
            else:
                context_col, view_col = st.columns([5, 2], gap="small", vertical_alignment="center")
                with view_col:
                    year_view = st.selectbox(
                        "Year Type View",
                        ["Calendar Year (Jan-Dec)", "Financial Year (Jul-Jun)"],
                        index=0,
                        key="year_type_view_select",
                        label_visibility="collapsed",
                    )
                fiscal = "Financial Year" in year_view
                active_filters = [f"{name}: {value}" for name, value in filters.items() if value != "TOTAL"]
                with context_col:
                    st.caption(
                        f"Actuals complete through {last_complete.strftime('%b %Y')} · "
                        f"{len(selected_sales):,} sales rows in selection · "
                        + (" · ".join(active_filters) if active_filters else "all locations, channels and products")
                    )

                total_po = 0.0
                if not purchase_months.empty and {"Date", "Open PO Qty"} <= set(purchase_months.columns):
                    po_periods = parse_dates(purchase_months["Date"]).dt.to_period("M")
                    po_by_period = purchase_months.groupby(po_periods)["Open PO Qty"].sum()
                    total_po = float(po_by_period[po_by_period.index <= rolling_window[-1]].sum())

                rolling_total = result["rolling_total"]
                mape = result["mape"]
                forecast_label_prefix = "$" if metric in ["Sale", "Cost", "GP $"] else ""
                if rolling_total is None:
                    rolling_display = "n/a"
                elif metric == "GP%":
                    rolling_display = f"{rolling_total:.1f}%"
                else:
                    rolling_display = f"{forecast_label_prefix}{rolling_total:,.0f}"

                current_period = last_complete + 1
                view_this_year = view_year(current_period, fiscal)
                this_label = view_label(view_this_year, fiscal)
                this_year_periods = view_year_periods(view_this_year, fiscal)
                ytd_periods = [p for p in this_year_periods if p <= last_complete]
                ytg_periods = [p for p in this_year_periods if p > last_complete]

                forecast_frames_for_series = [
                    frame for frame in (forecast_this_year, forecast_next_year) if not frame.empty
                ]
                fc_series = (
                    pd.concat(
                        [
                            pd.Series(frame["Forecast"].values, index=pd.to_datetime(frame["Date"]).dt.to_period("M"))
                            for frame in forecast_frames_for_series
                        ]
                    )
                    if forecast_frames_for_series
                    else pd.Series(dtype=float)
                )

                if metric == "GP%":
                    ytd_act = float(actual_series.
