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
targets_folder_path = raw_data_path / "targets"
purchasing_folder_path = raw_data_path / "purchasing"

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
    "allocated": ["qty alloc", "allocated", "qty allocated", "allocated qty", "alloc qty", "alloc"],
    "damaged": ["qty d", "damaged", "qty damaged", "damaged qty"],
    "floor": ["qty f", "floor stock", "qty floor", "floor qty", "floor"],
    "loan": ["qty l", "qty loan", "loan qty", "on loan", "loan"],
}
INVENTORY_ROLE_LABELS = {
    "sku": "SKU column",
    "on_hand": "On-hand quantity column",
    "free": "Free stock column",
    "site": "Site / location column",
    "value": "Stock value column",
    "cost": "Average option cost column",
    "description": "Description column",
    "allocated": "Allocated quantity column (purchasing)",
    "damaged": "Damaged quantity column (purchasing)",
    "floor": "Floor stock quantity column (purchasing)",
    "loan": "Loan quantity column (purchasing)",
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


def _vendor_code_key(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_vendor_brand_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        workbook = pd.ExcelFile(path)
        frame = workbook.parse(header=None)
    except Exception:
        return {}
    if frame.empty or frame.shape[1] < 5:
        return {}
    table = frame.iloc[:, [2, 4]].copy()
    table.columns = ["Options_ID__c", "Brand__r.Name"]
    if table.head(1).astype(str).stack().str.contains("supplier|brand|code", case=False, na=False).any():
        table = table.iloc[1:].reset_index(drop=True)
    lookup: dict[str, str] = {}
    for code, brand in table.itertuples(index=False, name=None):
        key = _vendor_code_key(code)
        final = "" if pd.isna(brand) else str(brand).strip()
        if key and final:
            lookup[key] = final
    return lookup


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
        elif isinstance(frame, dict):
            for sub_name, sub_frame in frame.items():
                if isinstance(sub_frame, pd.DataFrame):
                    sub_frame.attrs["cache_token"] = f"{token}:{name}:{sub_name}"
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
    vendor_brand_map = load_vendor_brand_map(raw_data_path / "Vendor Code and Brand.xlsx")
    if vendor_brand_map:
        def map_vendor_brand(value):
            if pd.isna(value):
                return value
            return vendor_brand_map.get(_vendor_code_key(value), value)
        sales["Supplier Brand"] = sales["Supplier Brand"].map(map_vendor_brand) if "Supplier Brand" in sales.columns and "Supplier Brand" in sales.columns else sales.get("Supplier Brand")
        if options is not None and not options.empty and "Supplier" in options.columns:
            options = options.copy()
            options["Supplier"] = options["Supplier"].map(map_vendor_brand)
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

    vendor_brand_map = load_vendor_brand_map(raw_data_path / "Vendor Code and Brand.xlsx")
    if vendor_brand_map:
        def map_vendor_brand(value):
            if pd.isna(value):
                return value
            return vendor_brand_map.get(_vendor_code_key(value), value)
        for frame_name in ("Options Data", "Current Inventory", "Purchase Orders"):
            frame = sources.get(frame_name)
            if frame is None or not isinstance(frame, pd.DataFrame):
                continue
            if "Supplier" in frame.columns:
                frame["Supplier"] = frame["Supplier"].map(map_vendor_brand)
        if "Sales Raw" in sources and isinstance(sources["Sales Raw"], pd.DataFrame):
            sales_raw = sources["Sales Raw"]
            if "Supplier" in sales_raw.columns:
                sales_raw["Supplier"] = sales_raw["Supplier"].map(map_vendor_brand)

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

    # Monthly targets (optional): newest file under Raw_data/targets.
    targets_file = newest_file(targets_folder_path)
    if targets_file is not None:
        try:
            sheets["Targets"] = read_tabular(targets_file)
            source_names.append(str(targets_file.relative_to(raw_data_path)))
        except Exception as e:
            sheets["Targets"] = pd.DataFrame()
            notes.append(f"Could not read the targets file '{targets_file.name}': {e}")
    else:
        sheets["Targets"] = pd.DataFrame()

    # Purchasing parameters (optional): workbooks under Raw_data/purchasing.
    try:
        sheets["Purchasing Files"] = read_purchasing_folder(purchasing_folder_path)
    except Exception as e:
        sheets["Purchasing Files"] = {}
        notes.append(f"Could not read the purchasing parameter files: {e}")
    if purchasing_folder_path.exists():
        source_names.extend(
            str(p.relative_to(raw_data_path)) for p in sorted(purchasing_folder_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS | {".xlsm"}
        )
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
# Monthly targets
#
# Two sources, in order of preference:
#   1. A targets file under Raw_data/targets: one row per month (and optionally
#      per location / channel / supplier / classification / class / model / SKU)
#      with a target per metric. The active filters pick the matching rows.
#   2. The annual Sale Target in the sidebar, spread over the months by last
#      year's seasonality and scaled to the selection's share of last year.
# Either way, a filter the targets cannot express directly pro-rates the target
# by the selection's share of the prior year's actuals, so the line always
# follows the filters.
# =============================================================================

TARGET_DIMENSIONS: dict[str, list[str]] = {
    "Location": ["location", "city", "site", "store", "showroom", "state"],
    "Sales Channel": ["sales channel net", "sales channel", "channel"],
    "Classification": ["classification", "fourth product group", "product group", "category"],
    "Supplier": ["supplier brand", "supplier", "brand", "vendor"],
    "Class": ["class"],
    "Model": ["model"],
    "SKU": ["sku key", "sku", "stock code", "product code", "item code"],
}
TARGET_METRIC_HEADERS: dict[str, list[str]] = {
    "Sale": ["sale target", "sales target", "sale", "sales", "revenue", "target"],
    "Qty": ["qty target", "units target", "qty", "units", "quantity"],
    "Cost": ["cost target", "cost"],
    "GP $": ["gp $", "gp$", "gp target", "gross profit", "gp dollars"],
    "GP%": ["gp%", "gp %", "margin %", "margin"],
}
TARGET_PERIOD_HEADERS = ["period", "date", "month start", "month"]
TARGET_YEAR_HEADERS = ["year", "calendar year", "cy"]
TARGET_FY_HEADERS = ["financial year", "fiscal year", "fy"]
TARGET_MONTH_HEADERS = ["month", "month name", "month no", "mth"]
TARGET_WILDCARDS = {"", "all", "total", "*", "any", "nan", "none"}


def _month_number(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and 1 <= int(value) <= 12:
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit() and 1 <= int(text) <= 12:
        return int(text)
    for number, name in MONTH_NUMBER_TO_NAME.items():
        if text == name.lower() or text == name.lower()[:3]:
            return number
    return None


def _fy_end_year(value) -> float:
    """FY27, FY2027, 2027, 2026/27, 2026-2027 → 2027 (the year the financial year ends)."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    span = re.search(r"(\d{2,4})\s*[/\-–]\s*(\d{2,4})", text)
    if span:
        end = int(span.group(2))
        return float(end if end > 100 else 2000 + end)
    digits = re.search(r"(\d{2,4})", text)
    if not digits:
        return np.nan
    year = int(digits.group(1))
    return float(year if year > 100 else 2000 + year)


def parse_targets(targets: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Normalise a targets file to long form: Period (month), Metric, <dimensions>, Target.

    Accepts a date/period column, or Year (or FY) + Month columns; targets as
    one numeric column per metric (wide) or a Metric + Target pair (long).
    Dimension cells that are blank / ALL / TOTAL mean "every value".
    Returns the long frame and the list of dimension columns the file carries.
    """
    empty = pd.DataFrame(columns=["Period", "Metric", "Target"])
    if targets is None or targets.empty:
        return empty, []
    frame = targets.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    used: list[str] = []

    # Period
    period_col = detect_column(frame.columns, TARGET_PERIOD_HEADERS)
    period = None
    if period_col is not None and frame[period_col].map(_month_number).notna().mean() <= 0.5:
        # A real date / period column (not a bare month number or name, which needs a year column).
        parsed = parse_dates(frame[period_col])
        if parsed.notna().mean() > 0.5:
            period = parsed.dt.to_period("M")
            used.append(period_col)
    if period is None:
        fy_col = detect_column(frame.columns, TARGET_FY_HEADERS)
        year_col = detect_column(frame.columns, TARGET_YEAR_HEADERS, exclude=(fy_col or "",))
        month_col = detect_column(frame.columns, TARGET_MONTH_HEADERS, exclude=tuple(c for c in (year_col, fy_col) if c))
        if month_col is None or (year_col is None and fy_col is None):
            return empty, []
        months = frame[month_col].map(_month_number)
        if year_col is not None:
            years = pd.to_numeric(frame[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
        else:
            fy = frame[fy_col].map(_fy_end_year)
            years = np.where(months <= 6, fy, fy - 1)
        valid = months.notna() & pd.notna(years)
        period = pd.Series(pd.NaT, index=frame.index, dtype="period[M]")
        period.loc[valid] = pd.PeriodIndex(
            [f"{int(y)}-{int(m):02d}" for y, m in zip(pd.Series(years)[valid], months[valid])], freq="M"
        )
        used += [c for c in (year_col, fy_col, month_col) if c]
    frame["Period"] = period

    # Dimensions
    dimensions: list[str] = []
    for name, candidates in TARGET_DIMENSIONS.items():
        col = detect_column(frame.columns, candidates, exclude=tuple(used))
        if col is not None:
            frame[name] = frame[col].astype(str).str.strip().where(frame[col].notna(), "")
            frame[name] = frame[name].where(~frame[name].str.lower().isin(TARGET_WILDCARDS), "")
            dimensions.append(name)
            used.append(col)

    # Metrics: long (Metric + Target) or wide (one column per metric)
    metric_col = detect_column(frame.columns, ["metric", "measure", "kpi"], exclude=tuple(used))
    if metric_col is not None:
        value_col = detect_column(frame.columns, ["target", "value", "amount", "goal", "budget"], exclude=tuple(used) + (metric_col,))
        if value_col is None:
            return empty, dimensions
        aliases = {alias: metric for metric, heads in TARGET_METRIC_HEADERS.items() for alias in heads}
        frame["Metric"] = frame[metric_col].astype(str).str.strip().str.lower().map(lambda v: aliases.get(v, v.title()))
        frame["Target"] = to_number(frame[value_col])
        long = frame[["Period", "Metric", *dimensions, "Target"]]
    else:
        pieces = []
        for metric, heads in TARGET_METRIC_HEADERS.items():
            col = detect_column(frame.columns, heads, exclude=tuple(used))
            if col is None:
                continue
            used.append(col)
            piece = frame[["Period", *dimensions]].copy()
            piece["Metric"] = metric
            piece["Target"] = to_number(frame[col])
            pieces.append(piece)
        if not pieces:
            return empty, dimensions
        long = pd.concat(pieces, ignore_index=True)[["Period", "Metric", *dimensions, "Target"]]
    long = long.loc[long["Period"].notna() & long["Target"].notna()].reset_index(drop=True)
    return long, dimensions


def targets_from_file(
    long: pd.DataFrame, dimensions: list[str], metric: str, filters: dict, periods: pd.PeriodIndex
) -> tuple[pd.Series | None, list[str]]:
    """Target per period for the metric under the filters the file can express.

    For a dimension the file has: an active filter keeps rows equal to its
    value; no filter keeps the file's company-level rows (blank / ALL) when it
    has any, otherwise the sum of all rows. Returns (series, dimensions the
    file could not express for the active filters).
    """
    rows = long.loc[long["Metric"].eq(metric)]
    if rows.empty:
        return None, []
    for dim in dimensions:
        value = filters.get(dim, "TOTAL")
        column = rows[dim].str.lower()
        if value != "TOTAL":
            wanted = str(value).strip().lower()
            exact = column.eq(wanted)
            # Same semantics as the sales filter: exact value, else "contains" (typed partial values).
            rows = rows.loc[exact] if exact.any() else rows.loc[column.str.contains(wanted, regex=False) & column.ne("")]
        elif column.eq("").any():
            rows = rows.loc[column.eq("")]
    if rows.empty:
        return None, []
    grouped = rows.groupby("Period")["Target"]
    series = (grouped.mean() if metric == "GP%" else grouped.sum()).reindex(periods)
    unexpressed = [key for key, value in filters.items() if value != "TOTAL" and key not in dimensions]
    return series, unexpressed


def prior_year_total(sales: pd.DataFrame, metric: str, prior_periods: pd.PeriodIndex) -> float:
    series = monthly_series(sales, metric)
    if series.empty:
        return 0.0
    return float(series.reindex(prior_periods).fillna(0.0).sum())


def derived_targets(
    annual_target: float,
    company_prior: pd.Series,
    selection_share: float,
    periods: pd.PeriodIndex,
    prior_periods: pd.PeriodIndex,
) -> pd.Series:
    """Spread an annual target over the year's months by last year's seasonality, times the selection's share."""
    prior = company_prior.reindex(prior_periods).fillna(0.0)
    if prior.sum() > 0:
        shares = (prior / prior.sum()).to_numpy()
    else:
        shares = np.full(len(periods), 1.0 / len(periods))
    return pd.Series(annual_target * shares * selection_share, index=periods, dtype=float)


@st.cache_data(show_spinner=False, max_entries=16, **BY_TOKEN)
def company_monthly_series(sales: pd.DataFrame, metric: str, before: pd.Timestamp | None) -> pd.Series:
    return monthly_series(sales, metric, before=before)


def period_total(series: pd.Series, periods: pd.PeriodIndex) -> float:
    return float(series.reindex(periods).fillna(0.0).sum()) if not series.empty else 0.0


def resolve_monthly_targets(
    targets_long: pd.DataFrame,
    target_dimensions: list[str],
    sales_data: pd.DataFrame,
    company_series: pd.Series,
    selection_series: pd.Series,
    filters: dict,
    metric: str,
    periods: pd.PeriodIndex,
    prior_periods: pd.PeriodIndex,
    share_periods: pd.PeriodIndex,
    annual_target: float,
) -> tuple[pd.Series | None, str]:
    """Monthly target for the selection over `periods`, plus a one-line description of how it was built.

    Targets file first (rows matching the filters it can express, pro-rated by the
    selection's share for the filters it cannot); otherwise the sidebar Sale Target
    spread by the selection's prior-year months as a share of the company's prior year.
    `company_series` / `selection_series` are the monthly series of all sales and of
    the filtered selection for `metric` (complete months).
    """
    active = {key: value for key, value in filters.items() if value != "TOTAL"}
    if not targets_long.empty:
        series, unexpressed = targets_from_file(targets_long, target_dimensions, metric, filters, periods)
        if series is not None and series.notna().any():
            expressed = [key for key in active if key in target_dimensions]
            basis = "Targets file"
            if expressed:
                basis += " · rows for " + ", ".join(f"{key} {active[key]}" for key in expressed)
            if unexpressed:
                if metric == "GP%":
                    share = 1.0
                elif expressed:
                    base_filters = {key: (value if key in target_dimensions else "TOTAL") for key, value in filters.items()}
                    base_sales, _ = engine_call("filter_sales", local_filter_sales, sales_data, base_filters)
                    base_total = period_total(monthly_series(base_sales, metric), share_periods)
                    share = min(max(period_total(selection_series, share_periods) / base_total, 0.0), 1.0) if base_total > 0 else None
                else:
                    base_total = period_total(company_series, share_periods)
                    share = min(max(period_total(selection_series, share_periods) / base_total, 0.0), 1.0) if base_total > 0 else None
                if share is None:
                    return None, (
                        "Targets file has no rows for " + ", ".join(unexpressed)
                        + " and the selection has no sales in the last 12 months to pro-rate by."
                    )
                series = series * share
                basis += f" · pro-rated to {share:.1%} for " + ", ".join(f"{key} {active[key]}" for key in unexpressed)
            return series, basis
        if series is None and active and any(key in target_dimensions for key in active):
            return None, "Targets file has no rows matching " + ", ".join(f"{key} {active[key]}" for key in active if key in target_dimensions)
        if metric != "Sale":
            return None, f"Targets file carries no {metric} target."
    if metric != "Sale":
        return None, f"No target for {metric}: add a targets file under Raw_data/targets with a {metric} column."
    if annual_target <= 0:
        return None, "Sale Target is zero."
    denominator = period_total(company_series, prior_periods)
    if denominator <= 0:
        return None, f"No sales in {prior_periods[0].strftime('%b %Y')}–{prior_periods[-1].strftime('%b %Y')} to spread the Sale Target over."
    shares = (
        selection_series.reindex(prior_periods).fillna(0.0).to_numpy() if not selection_series.empty else np.zeros(len(prior_periods))
    ) / denominator
    if shares.sum() <= 0:
        return None, "Selection had no sales in the prior year to spread the Sale Target over."
    series = pd.Series(annual_target * shares, index=periods, dtype=float)
    basis = f"Sale Target ${annual_target:,.0f} spread by last year's months"
    if active:
        basis += f" · selection was {shares.sum():.1%} of last year"
    return series, basis


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
    "W1SY": "Warehouse", "W3ME": "Warehouse", "W4PE": "Warehouse", "W5BR": "Warehouse", "W6AD": "Warehouse",
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
# Purchasing — Forward Order Schedule (F.O.S.)
#
# Port of the Workings tab of Forward_Order_Schedule.xlsm. Per SKU:
#   • sales history by city and channel for the last N complete months, each
#     sales location allocated to a physical city (or split by %) and to
#     Retail / Commercial via the Sales Locations parameters;
#   • free stock = on hand − allocated − damaged/floor/loan at Warehouse sites;
#   • open purchase orders by arrival month and city, linked (to a sales order)
#     or free; a PO without an ETA is estimated as order date + lead time;
#   • a rolling monthly projection of national and per-city free stock
#     (opening + free arrivals + confirmed new order − average monthly sales,
#     the current month pro-rated for the days left);
#   • planned order qty = MOH required × average monthly sales − projected free
#     stock in the month a new order would arrive (as-at date + lead time).
# =============================================================================

FOS_CITIES = ["SYD", "BNE", "MEL", "ADL", "PER"]
FOS_CITY_NAMES = {"SYD": "Sydney", "BNE": "Brisbane", "MEL": "Melbourne", "ADL": "Adelaide", "PER": "Perth"}
FOS_CITY_BY_NAME = {name.lower(): code for code, name in FOS_CITY_NAMES.items()} | {"oxford st": "SYD", "oxford street": "SYD", "paddington": "SYD"}
FOS_CHANNELS = ["Retail", "Commercial"]
FOS_LOCATION_COLUMNS = ["Code", "Location", "Channel", "City", "SYD", "BNE", "MEL", "ADL", "PER"]
DEFAULT_SALES_LOCATIONS: list[dict] = [
    {"Code": "1WAR", "Location": "Warranty Sales Orders", "Channel": "Retail", "City": "Split", "SYD": 0.35, "BNE": 0.20, "MEL": 0.25, "ADL": 0.0, "PER": 0.20},
    {"Code": "ABRI", "Location": "A&D Brisbane", "Channel": "Retail", "City": "BNE"},
    {"Code": "AMEL", "Location": "A&D Melbourne", "Channel": "Retail", "City": "MEL"},
    {"Code": "APER", "Location": "A&D Perth", "Channel": "Retail", "City": "PER"},
    {"Code": "ASYD", "Location": "A&D Sydney", "Channel": "Retail", "City": "SYD"},
    {"Code": "CBRI", "Location": "Contract Brisbane", "Channel": "Commercial", "City": "BNE"},
    {"Code": "CMEL", "Location": "Contract Melbourne", "Channel": "Commercial", "City": "MEL"},
    {"Code": "CPER", "Location": "Contract Perth", "Channel": "Commercial", "City": "PER"},
    {"Code": "CSYD", "Location": "Contract Sydney", "Channel": "Commercial", "City": "SYD"},
    {"Code": "DEAL", "Location": "Sub Dealer Sales", "Channel": "Commercial", "City": "Split", "SYD": 0.60, "BNE": 0.02, "MEL": 0.33, "ADL": 0.0, "PER": 0.05},
    {"Code": "EDIT", "Location": "Edition", "Channel": "Retail", "City": "SYD"},
    {"Code": "ESTR", "Location": "ECOMMERCE SALES", "Channel": "Retail", "City": "Split", "SYD": 0.40, "BNE": 0.23, "MEL": 0.30, "ADL": 0.0, "PER": 0.07},
    {"Code": "LEDS", "Location": "DESIGN STUDIO", "Channel": "Retail", "City": "Split", "SYD": 0.70, "BNE": 0.0, "MEL": 0.30, "ADL": 0.0, "PER": 0.0},
    {"Code": "RBRI", "Location": "Retail Brisbane", "Channel": "Retail", "City": "BNE"},
    {"Code": "RMEL", "Location": "Retail Melbourne", "Channel": "Retail", "City": "MEL"},
    {"Code": "RPER", "Location": "Retail Perth", "Channel": "Retail", "City": "PER"},
    {"Code": "RSYD", "Location": "Retail Sydney", "Channel": "Retail", "City": "SYD"},
    {"Code": "ROXF", "Location": "Retail Oxford St", "Channel": "Retail", "City": "SYD"},
    {"Code": "W1SY", "Location": "Living Edge Warehouse - Sydney", "Channel": "Retail", "City": "SYD"},
    {"Code": "W3ME", "Location": "Pedemonts Melbourne", "Channel": "Retail", "City": "MEL"},
    {"Code": "WARB", "Location": "Warranty Brisbane", "Channel": "Retail", "City": "BNE"},
    {"Code": "WARM", "Location": "Warranty Melbourne", "Channel": "Retail", "City": "MEL"},
    {"Code": "WARP", "Location": "Warranty Perth", "Channel": "Retail", "City": "PER"},
    {"Code": "WARS", "Location": "Warranty Sydney", "Channel": "Retail", "City": "SYD"},
    {"Code": "XITB", "Location": "Brisbane Clearance Sale", "Channel": "Retail", "City": "BNE"},
    {"Code": "XITM", "Location": "Exit Product - Melbourne", "Channel": "Retail", "City": "MEL"},
    {"Code": "XITP", "Location": "Perth Clearance Product", "Channel": "Retail", "City": "PER"},
    {"Code": "XITS", "Location": "Exit Product - Sydney", "Channel": "Retail", "City": "SYD"},
]
DEFAULT_WAREHOUSE_CITIES = {"W1SY": "SYD", "W5BR": "BNE", "W3ME": "MEL", "W6AD": "ADL", "W4PE": "PER"}
DEFAULT_PURCHASING_SETTINGS = {"moh_required": 2.0, "history_months": 5, "horizon_months": 11}
FOS_PLANNING_HORIZON_MAX = 18

OPTION_FIELD_CANDIDATES: dict[str, list[str]] = {
    "lead_time": ["days supply", "lead time days", "lead time", "leadtime", "lead days"],
    "rrp": ["retail ex gst sell", "rrp", "retail price", "sell price", "retail sell", "retail"],
    "purchase_cost": ["purchase ex gst buy", "purchase cost", "purchase price", "buy price", "purchase"],
    "store_cost": ["store ex gst buy", "store cost", "landed cost"],
    "supplier_qty": ["supplier qty available", "supplier qty", "supplier stock", "supplier quantity"],
    "supplier_qty_date": ["supplier qty updated date", "supplier qty updated", "supplier qty date"],
    "discontinued": ["discontinued flag", "discontinued"],
    "supplier_code": ["supplier code", "supp stock code"],
}
PO_FIELD_CANDIDATES: dict[str, list[str]] = {
    "ordered": ["qty ordered", "ordered qty", "order qty", "quantity ordered", "ordered"] + OPEN_PO_HEADERS,
    "received": ["received", "qty received", "received qty", "quantity received"],
    "sales_order": ["sales order", "so number", "so no", "so #", "linked order", "customer order", "sales order number"],
    "eta": ["eta date", "eta", "expected date", "expected arrival", "arrival date", "due date", "expected"],
    "order_date": ["po order date", "order date", "po date", "ordered date", "date"],
    "location": ["po location", "location", "site", "warehouse", "deliver to", "destination"],
    "po_number": ["po order", "po number", "po no", "po #", "purchase order", "po"],
    "supplier": ["supplier", "vendor", "brand"],
    "unit_cost": ["unit $", "unit cost", "unit price", "cost"],
}


def normalise_text(value) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "").replace(" ", " ").strip().lower())


def sales_locations_frame(rows: list[dict] | None) -> pd.DataFrame:
    frame = pd.DataFrame(rows if rows else DEFAULT_SALES_LOCATIONS)
    for column in FOS_LOCATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0 if column in FOS_CITIES else ""
    frame = frame[FOS_LOCATION_COLUMNS].copy()
    for column in ("Code", "Location", "Channel", "City"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    for city in FOS_CITIES:
        frame[city] = pd.to_numeric(frame[city], errors="coerce").fillna(0.0)
    return frame.reset_index(drop=True)


def location_rules(frame: pd.DataFrame) -> dict[str, tuple[str, dict[str, float] | None]]:
    """Normalised location name / code → (channel, city weights)."""
    rules: dict[str, tuple[str, dict[str, float] | None]] = {}
    for row in frame.itertuples(index=False):
        channel = "Commercial" if str(row.Channel).strip().lower().startswith("com") else "Retail"
        city = str(row.City).strip().upper()
        if city in FOS_CITIES:
            weights: dict[str, float] | None = {city: 1.0}
        else:
            raw = {c: float(getattr(row, c)) for c in FOS_CITIES}
            total = sum(v for v in raw.values() if v > 0)
            weights = {c: v / total for c, v in raw.items() if v > 0} if total > 0 else None
        for key in (row.Code, row.Location):
            if normalise_text(key):
                rules[normalise_text(key)] = (channel, weights)
    return rules


def classify_sales_location(location: str, city_hint: str, rules: dict) -> tuple[str, dict[str, float] | None]:
    key = normalise_text(location)
    if key in rules:
        return rules[key]
    channel = "Commercial" if any(word in key for word in ("contract", "dealer", "commercial")) else "Retail"
    hint = FOS_CITY_BY_NAME.get(normalise_text(city_hint))
    if hint is None:
        for name, code in FOS_CITY_BY_NAME.items():
            if name in key:
                hint = code
                break
    return channel, ({hint: 1.0} if hint else None)


def warehouse_city(code: str, warehouse_cities: dict[str, str]) -> str:
    code = normalise_site_code(code)
    if code in warehouse_cities:
        return warehouse_cities[code]
    for suffix, city in (("SYD", "SYD"), ("SY", "SYD"), ("MEL", "MEL"), ("ME", "MEL"), ("PER", "PER"), ("PE", "PER"),
                         ("BRI", "BNE"), ("BR", "BNE"), ("BNE", "BNE"), ("ADL", "ADL"), ("AD", "ADL")):
        if code.endswith(suffix):
            return city
    return "OTH"


def rules_signature(frame: pd.DataFrame) -> tuple:
    return tuple(tuple(row) for row in frame.itertuples(index=False))


@st.cache_data(show_spinner=False, max_entries=8, **BY_TOKEN)
def fos_sales_history(sales: pd.DataFrame, start: str, end: str, rules_rows: tuple) -> pd.DataFrame:
    """Unit sales per SKU × month × city × channel over the window, locations allocated per the rules."""
    columns = ["SKU", "Period", "City", "Channel", "Qty"]
    if sales.empty or "Date" not in sales.columns or "Qty" not in sales.columns:
        return pd.DataFrame(columns=columns)
    rules = location_rules(sales_locations_frame([dict(zip(FOS_LOCATION_COLUMNS, row)) for row in rules_rows]))
    dated = sales.dropna(subset=["Date"])
    period = dated["Date"].dt.to_period("M")
    first, last = pd.Period(start, freq="M"), pd.Period(end, freq="M")
    dated = dated.loc[(period >= first) & (period <= last)]
    if dated.empty:
        return pd.DataFrame(columns=columns)
    key_column = "SKU Key" if "SKU Key" in dated.columns else "SKU"
    frame = pd.DataFrame(
        {
            "SKU": map_unique(dated[key_column], sku_text),
            "Period": dated["Date"].dt.to_period("M").astype(str),
            "Location": dated["Location"].astype(object).fillna("").astype(str).to_numpy() if "Location" in dated.columns else "",
            "CityHint": dated["City"].astype(object).fillna("").astype(str).to_numpy() if "City" in dated.columns else "",
            "Qty": to_number(dated["Qty"]).fillna(0.0),
        }
    )
    frame = frame.loc[frame["SKU"].ne("") & frame["Qty"].ne(0)]
    grouped = frame.groupby(["SKU", "Period", "Location", "CityHint"], sort=False, observed=True)["Qty"].sum().reset_index()
    pairs = grouped[["Location", "CityHint"]].drop_duplicates()
    decisions = {
        (loc, hint): classify_sales_location(loc, hint, rules) for loc, hint in zip(pairs["Location"], pairs["CityHint"])
    }
    known_mix: dict[str, float] = {}
    for (loc, hint), (_channel, weights) in decisions.items():
        if weights:
            qty = float(grouped.loc[grouped["Location"].eq(loc) & grouped["CityHint"].eq(hint), "Qty"].sum())
            for city, weight in weights.items():
                known_mix[city] = known_mix.get(city, 0.0) + qty * weight
    mix_total = sum(v for v in known_mix.values() if v > 0)
    fallback = (
        {city: v / mix_total for city, v in known_mix.items() if v > 0}
        if mix_total > 0
        else {city: 0.25 for city in ("SYD", "BNE", "MEL", "PER")}
    )
    pieces = []
    for (loc, hint), (channel, weights) in decisions.items():
        subset = grouped.loc[grouped["Location"].eq(loc) & grouped["CityHint"].eq(hint), ["SKU", "Period", "Qty"]]
        for city, weight in (weights or fallback).items():
            piece = subset.copy()
            piece["City"] = city
            piece["Channel"] = channel
            piece["Qty"] = piece["Qty"] * weight
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=columns)
    out = pd.concat(pieces, ignore_index=True)
    return out.groupby(["SKU", "Period", "City", "Channel"], sort=False)["Qty"].sum().reset_index()[columns]


def detect_option_fields(options: pd.DataFrame | None) -> dict[str, str | None]:
    if options is None or options.empty:
        return {role: None for role in OPTION_FIELD_CANDIDATES}
    sku_col = detect_column(options.columns, ["stock code", "sku", "sku key"])
    used = [sku_col] if sku_col else []
    found = {}
    for role, candidates in OPTION_FIELD_CANDIDATES.items():
        col = detect_column(options.columns, candidates, exclude=tuple(c for c in used if c))
        found[role] = col
        if col:
            used.append(col)
    return found


@st.cache_data(show_spinner=False, max_entries=8, **BY_TOKEN)
def fos_option_fields(options: pd.DataFrame | None) -> pd.DataFrame:
    """SKU-level purchasing attributes from the options export (lead time, RRP, purchase cost, supplier stock)."""
    columns = ["SKU", "Lead time (days)", "RRP", "Purchase cost", "Store cost", "Supplier qty", "Supplier qty updated", "Discontinued", "Supplier code"]
    if options is None or options.empty:
        return pd.DataFrame(columns=columns)
    sku_col = detect_column(options.columns, ["stock code", "sku", "sku key"])
    if sku_col is None:
        return pd.DataFrame(columns=columns)
    fields = detect_option_fields(options)
    frame = pd.DataFrame({"SKU": map_unique(options[sku_col], sku_text)})
    numeric = {"Lead time (days)": "lead_time", "RRP": "rrp", "Purchase cost": "purchase_cost", "Store cost": "store_cost", "Supplier qty": "supplier_qty"}
    for label, role in numeric.items():
        frame[label] = to_number(options[fields[role]]) if fields[role] else np.nan
    frame["Supplier qty updated"] = parse_dates(options[fields["supplier_qty_date"]]) if fields["supplier_qty_date"] else pd.NaT
    frame["Discontinued"] = (
        options[fields["discontinued"]].astype(str).str.strip().str.upper().isin({"Y", "YES", "TRUE", "1"})
        if fields["discontinued"] else False
    )
    frame["Supplier code"] = options[fields["supplier_code"]].astype(str).str.strip() if fields["supplier_code"] else ""
    return frame.loc[frame["SKU"].ne("")].drop_duplicates(subset=["SKU"])[columns]


def fos_stock_by_city(inventory: pd.DataFrame, mapping: dict, site_mapping: dict, warehouse_cities: dict[str, str]) -> pd.DataFrame:
    """Warehouse stock per SKU × city: total, allocated, damaged/floor/loan, free (= total − allocated − D/F/L)."""
    columns = ["SKU", "City", "Total", "Allocated", "DFL", "Free"]
    if inventory is None or inventory.empty or not mapping.get("sku") or not mapping.get("on_hand"):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({"SKU": map_unique(inventory[mapping["sku"]], sku_text)})
    frame["Total"] = to_number(inventory[mapping["on_hand"]]).fillna(0.0)
    frame["Allocated"] = to_number(inventory[mapping["allocated"]]).fillna(0.0) if mapping.get("allocated") else np.nan
    dfl = pd.Series(0.0, index=frame.index)
    has_dfl = False
    for role in ("damaged", "floor", "loan"):
        if mapping.get(role):
            dfl = dfl + to_number(inventory[mapping[role]]).fillna(0.0)
            has_dfl = True
    frame["DFL"] = dfl if has_dfl else np.nan
    if mapping.get("allocated"):
        frame["Free"] = frame["Total"] - frame["Allocated"] - frame["DFL"].fillna(0.0)
    elif mapping.get("free"):
        frame["Free"] = to_number(inventory[mapping["free"]]).fillna(0.0)
        frame["Allocated"] = frame["Total"] - frame["Free"]
    else:
        frame["Free"] = frame["Total"]
        frame["Allocated"] = 0.0
    frame["Allocated"] = frame["Allocated"].fillna(0.0)
    frame["DFL"] = frame["DFL"].fillna(0.0)
    if mapping.get("site"):
        sites = map_unique(inventory[mapping["site"]], normalise_site_code)
        category = map_unique(sites, lambda code: category_for(code, site_mapping))
        frame["City"] = map_unique(sites, lambda code: warehouse_city(code, warehouse_cities))
        frame = frame.loc[category.eq("Warehouse")]
    else:
        frame["City"] = "SYD"
    frame = frame.loc[frame["SKU"].ne("")]
    return frame.groupby(["SKU", "City"], sort=False)[["Total", "Allocated", "DFL", "Free"]].sum().reset_index()[columns]


def detect_po_fields(purchase_orders: pd.DataFrame | None) -> dict[str, str | None]:
    if purchase_orders is None or purchase_orders.empty:
        return {role: None for role in PO_FIELD_CANDIDATES} | {"sku": None}
    sku_col = detect_column(purchase_orders.columns, ["stock code", "product code", "sku", "item code"])
    used = [sku_col] if sku_col else []
    found: dict[str, str | None] = {"sku": sku_col}
    for role, candidates in PO_FIELD_CANDIDATES.items():
        col = detect_column(purchase_orders.columns, candidates, exclude=tuple(c for c in used if c))
        found[role] = col
        if col:
            used.append(col)
    return found


def fos_purchase_orders(
    purchase_orders: pd.DataFrame | None,
    lead_times: pd.Series,
    asof: pd.Timestamp,
    site_mapping: dict,
    warehouse_cities: dict[str, str],
    default_lead_days: float,
) -> pd.DataFrame:
    """Open PO lines with an arrival month and city. ETA blank → order date + lead time (flagged Estimated)."""
    columns = ["SKU", "PO", "Supplier", "Open qty", "Linked", "Sales order", "Order date", "ETA", "Estimated", "Overdue",
               "Site", "City", "Warehouse", "Period", "Unit cost"]
    if purchase_orders is None or purchase_orders.empty:
        return pd.DataFrame(columns=columns)
    fields = detect_po_fields(purchase_orders)
    if not fields["sku"] or not fields["ordered"]:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({"SKU": map_unique(purchase_orders[fields["sku"]], sku_text)})
    ordered = to_number(purchase_orders[fields["ordered"]]).fillna(0.0)
    received = to_number(purchase_orders[fields["received"]]).fillna(0.0) if fields["received"] else 0.0
    frame["Open qty"] = (ordered - received).clip(lower=0.0)
    frame["PO"] = purchase_orders[fields["po_number"]].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip().to_numpy() if fields["po_number"] else ""
    frame["Supplier"] = purchase_orders[fields["supplier"]].fillna("").astype(str).str.strip().to_numpy() if fields["supplier"] else ""
    frame["Unit cost"] = to_number(purchase_orders[fields["unit_cost"]]) if fields["unit_cost"] else np.nan
    so = purchase_orders[fields["sales_order"]] if fields["sales_order"] else pd.Series("", index=purchase_orders.index)
    so_text = so.fillna("").astype(str).str.strip()
    blank_so = so_text.str.lower().isin({"", "nan", "none", "0", "0.0", "/  /"}).fillna(True)
    frame["Sales order"] = so_text.where(~blank_so, "").to_numpy()
    frame["Linked"] = frame["Sales order"].ne("")
    order_date = parse_dates(purchase_orders[fields["order_date"]]) if fields["order_date"] else pd.Series(pd.NaT, index=purchase_orders.index)
    eta = parse_dates(purchase_orders[fields["eta"]]) if fields["eta"] else pd.Series(pd.NaT, index=purchase_orders.index)
    frame["Order date"] = order_date.to_numpy()
    lead = frame["SKU"].map(lead_times).astype(float)
    lead = lead.where(lead.notna() & (lead > 0), default_lead_days)
    estimated_eta = pd.to_datetime(order_date.to_numpy()) + pd.to_timedelta(lead.to_numpy(), unit="D")
    frame["Estimated"] = eta.isna().to_numpy()
    frame["ETA"] = np.where(frame["Estimated"], estimated_eta, eta.to_numpy())
    frame["ETA"] = pd.to_datetime(frame["ETA"])
    frame["ETA"] = frame["ETA"].fillna(asof)
    frame["Overdue"] = frame["ETA"] < asof
    frame["Period"] = frame["ETA"].clip(lower=asof).dt.to_period("M").astype(str)
    if fields["location"]:
        sites = map_unique(purchase_orders[fields["location"]], normalise_site_code)
        frame["Site"] = sites.to_numpy()
        frame["Warehouse"] = map_unique(sites, lambda code: category_for(code, site_mapping) == "Warehouse").to_numpy()
        frame["City"] = map_unique(sites, lambda code: warehouse_city(code, warehouse_cities)).to_numpy()
    else:
        frame["Site"] = ""
        frame["Warehouse"] = True
        frame["City"] = "SYD"
    frame = frame.loc[frame["SKU"].ne("") & (frame["Open qty"] > 0)]
    return frame[columns].reset_index(drop=True)


def fos_schedule(
    skus: list[str],
    history: pd.DataFrame,
    history_periods: list[str],
    stock: pd.DataFrame,
    po_lines: pd.DataFrame,
    lead_times: pd.Series,
    confirmed: dict[str, float],
    asof: pd.Timestamp,
    moh_required: float,
    horizon_months: int,
) -> dict:
    """Project free stock month by month and derive the planned order per SKU.

    Returns {"summary": DataFrame, "national": DataFrame (SKU × month), "cities": {city: DataFrame},
             "periods": [str], "arrivals": DataFrame (SKU × month free arrivals), "linked": DataFrame}.
    """
    index = pd.Index(skus, name="SKU")
    cities = list(FOS_CITIES) + (["OTH"] if ("OTH" in set(stock["City"]) | set(po_lines["City"])) else [])
    current = asof.to_period("M")
    periods = [str(current + k) for k in range(horizon_months)]
    n, m = len(index), horizon_months

    # --- sales: monthly national totals (zeros for months without sales), averages
    hist_pivot = (
        history.pivot_table(index="SKU", columns="Period", values="Qty", aggfunc="sum", fill_value=0.0)
        if not history.empty else pd.DataFrame(index=index)
    ).reindex(index=index, columns=history_periods, fill_value=0.0).astype(float)
    avg = hist_pivot.mean(axis=1) if history_periods else pd.Series(0.0, index=index)
    by_channel = (
        history.pivot_table(index="SKU", columns="Channel", values="Qty", aggfunc="sum", fill_value=0.0)
        if not history.empty else pd.DataFrame(index=index)
    ).reindex(index=index, columns=FOS_CHANNELS, fill_value=0.0).astype(float) / max(len(history_periods), 1)
    city_avg = (
        history.pivot_table(index="SKU", columns="City", values="Qty", aggfunc="sum", fill_value=0.0)
        if not history.empty else pd.DataFrame(index=index)
    ).reindex(index=index, columns=cities, fill_value=0.0).astype(float) / max(len(history_periods), 1)

    # --- stock now
    def stock_pivot(column: str) -> pd.DataFrame:
        if stock.empty:
            return pd.DataFrame(0.0, index=index, columns=cities)
        return stock.pivot_table(index="SKU", columns="City", values=column, aggfunc="sum", fill_value=0.0).reindex(
            index=index, columns=cities, fill_value=0.0
        ).astype(float)

    free_now = stock_pivot("Free")
    total_now = stock_pivot("Total")
    allocated_now = stock_pivot("Allocated")
    dfl_now = stock_pivot("DFL")

    # --- arrivals (warehouse POs only), by SKU × period (× city)
    wh = po_lines.loc[po_lines["Warehouse"]] if not po_lines.empty else po_lines
    beyond = wh.loc[~wh["Period"].isin(periods)] if not wh.empty else wh
    wh_in = wh.loc[wh["Period"].isin(periods)] if not wh.empty else wh

    def arrivals(subset: pd.DataFrame, by_city: bool) -> pd.DataFrame | dict:
        if by_city:
            out = {}
            for city in cities:
                part = subset.loc[subset["City"].eq(city)]
                out[city] = (
                    part.pivot_table(index="SKU", columns="Period", values="Open qty", aggfunc="sum", fill_value=0.0)
                    if not part.empty else pd.DataFrame(index=index)
                ).reindex(index=index, columns=periods, fill_value=0.0).astype(float)
            return out
        return (
            subset.pivot_table(index="SKU", columns="Period", values="Open qty", aggfunc="sum", fill_value=0.0)
            if not subset.empty else pd.DataFrame(index=index)
        ).reindex(index=index, columns=periods, fill_value=0.0).astype(float)

    free_arrivals = arrivals(wh_in.loc[~wh_in["Linked"]] if not wh_in.empty else wh_in, by_city=False)
    linked_arrivals = arrivals(wh_in.loc[wh_in["Linked"]] if not wh_in.empty else wh_in, by_city=False)
    free_arrivals_city = arrivals(wh_in.loc[~wh_in["Linked"]] if not wh_in.empty else wh_in, by_city=True)

    # --- new (confirmed) orders land in the month of as-at + lead time
    lead = pd.Series(index.map(lambda s: lead_times.get(s, np.nan)), index=index, dtype=float).fillna(0.0).clip(lower=0.0)
    arrival_date = pd.Series([asof + pd.Timedelta(days=int(round(d))) for d in lead], index=index)
    arrival_index = pd.Series([min((date.to_period("M") - current).n, m - 1) for date in arrival_date], index=index).clip(lower=0)
    confirmed_qty = pd.Series(index.map(lambda s: float(confirmed.get(s, 0.0) or 0.0)), index=index, dtype=float)
    new_orders = pd.DataFrame(0.0, index=index, columns=periods)
    for sku, k, qty in zip(index, arrival_index, confirmed_qty):
        if qty:
            new_orders.loc[sku, periods[int(k)]] += qty

    # --- projection
    prorata = (asof.days_in_month - asof.day + 1) / asof.days_in_month
    nat = np.zeros((n, m))
    nat_no_new = np.zeros((n, m))
    avg_values = avg.to_numpy()
    opening_nat = free_now.sum(axis=1).to_numpy()
    for k in range(m):
        usage = avg_values * (prorata if k == 0 else 1.0)
        prev = opening_nat if k == 0 else nat[:, k - 1]
        prev_no_new = opening_nat if k == 0 else nat_no_new[:, k - 1]
        nat[:, k] = prev + free_arrivals.iloc[:, k].to_numpy() + new_orders.iloc[:, k].to_numpy() - usage
        nat_no_new[:, k] = prev_no_new + free_arrivals.iloc[:, k].to_numpy() - usage
    national = pd.DataFrame(nat, index=index, columns=periods)
    city_frames: dict[str, pd.DataFrame] = {}
    for city in cities:
        c = np.zeros((n, m))
        usage_city = city_avg[city].to_numpy()
        for k in range(m):
            prev = free_now[city].to_numpy() if k == 0 else c[:, k - 1]
            c[:, k] = prev + free_arrivals_city[city].iloc[:, k].to_numpy() - usage_city * (prorata if k == 0 else 1.0)
        city_frames[city] = pd.DataFrame(c, index=index, columns=periods)

    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(avg_values[:, None] > 0, national.to_numpy() / avg_values[:, None], np.nan)
    months_of_cover = pd.DataFrame(cover, index=index, columns=periods)

    # --- planned order: bring free stock in the arrival month up to MOH × average monthly sales
    at_arrival = np.array([nat[i, int(k)] for i, k in enumerate(arrival_index)])
    planned_raw = moh_required * avg_values - at_arrival
    planned = np.where((avg_values > 0) & (planned_raw > 0), np.ceil(planned_raw - 1e-9), 0.0)
    first_negative = []
    for i in range(n):
        negatives = np.where(nat_no_new[i] < 0)[0]
        first_negative.append(periods[int(negatives[0])] if len(negatives) else "")

    summary = pd.DataFrame(index=index)
    summary["Avg monthly sales"] = avg
    summary["Retail avg"] = by_channel["Retail"]
    summary["Commercial avg"] = by_channel["Commercial"]
    for period in history_periods:
        summary[f"Sales {pd.Period(period).strftime('%b %y')}"] = hist_pivot[period]
    summary["Total SOH"] = total_now.sum(axis=1)
    summary["Allocated SOH"] = allocated_now.sum(axis=1)
    summary["Damaged/floor/loan"] = dfl_now.sum(axis=1)
    summary["Free SOH"] = free_now.sum(axis=1)
    for city in cities:
        summary[f"Free {city}"] = free_now[city]
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["Months of cover now"] = np.where(avg_values > 0, summary["Free SOH"].to_numpy() / avg_values, np.nan)
    summary["Open PO free"] = free_arrivals.sum(axis=1) + (
        beyond.loc[~beyond["Linked"]].groupby("SKU")["Open qty"].sum().reindex(index).fillna(0.0) if not beyond.empty else 0.0
    )
    summary["Open PO linked"] = linked_arrivals.sum(axis=1) + (
        beyond.loc[beyond["Linked"]].groupby("SKU")["Open qty"].sum().reindex(index).fillna(0.0) if not beyond.empty else 0.0
    )
    summary["Open PO beyond horizon"] = beyond.groupby("SKU")["Open qty"].sum().reindex(index).fillna(0.0) if not beyond.empty else 0.0
    summary["Lead time (days)"] = lead
    summary["Arrival date"] = arrival_date.dt.normalize()
    summary["Arrival month"] = [periods[int(k)] for k in arrival_index]
    summary["Free SOH at arrival"] = at_arrival
    summary["Planned order qty"] = planned
    summary["Confirmed order qty"] = confirmed_qty
    summary["Stock-out month"] = first_negative
    return {
        "summary": summary.reset_index(),
        "national": national,
        "cover": months_of_cover,
        "cities": city_frames,
        "periods": periods,
        "free_arrivals": free_arrivals,
        "linked_arrivals": linked_arrivals,
        "new_orders": new_orders,
        "prorata": prorata,
    }


def read_purchasing_folder(folder: Path) -> dict[str, pd.DataFrame]:
    """Optional parameter workbooks under Raw_data/purchasing (sheet names decide the role)."""
    roles = {
        "Sales Locations": ["sales locations", "sales location", "locations", "parameters"],
        "Warehouses": ["warehouses", "warehouse cities", "sites"],
        "Product Groups": ["product groups", "product group"],
        "WEB Category": ["web category", "web categories", "categories"],
        "Confirmed Orders": ["confirmed orders", "confirmed", "orders"],
    }
    found: dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return found
    files = sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS | {".xlsm"}),
        key=lambda p: p.stat().st_mtime,
    )
    for path in files:
        if path.suffix.lower() == ".csv":
            stem = normalise_text(path.stem)
            for role, names in roles.items():
                if any(name in stem for name in names):
                    try:
                        found[role] = read_tabular(path)
                    except Exception:
                        pass
            continue
        try:
            book = pd.ExcelFile(path)
        except Exception:
            continue
        for sheet in book.sheet_names:
            key = normalise_text(sheet)
            for role, names in roles.items():
                if any(name == key or name in key for name in names):
                    try:
                        frame = book.parse(sheet, header=None)
                    except Exception:
                        continue
                    header_row = detect_header_row(frame.head(20))
                    if key == "parameters":
                        # the workbook's Parameters tab: the header is the row holding "Sales Location"
                        hits = [i for i in range(min(len(frame), 20)) if any(normalise_text(v) == "sales location" for v in frame.iloc[i])]
                        header_row = hits[0] if hits else header_row
                    frame.columns = [str(c).strip() for c in frame.iloc[header_row]]
                    frame = frame.iloc[header_row + 1 :].dropna(how="all").reset_index(drop=True)
                    if role == "Sales Locations" and key == "parameters":
                        frame = parameters_sheet_to_locations(frame)
                    found[role] = frame
    return found


def parameters_sheet_to_locations(frame: pd.DataFrame) -> pd.DataFrame:
    """The workbook's Parameters tab (Sales Location / Physical location / SYD..PER columns) → Sales Locations table."""
    headers = [normalise_text(c) for c in frame.columns]
    if "sales location" not in headers:
        return pd.DataFrame(columns=FOS_LOCATION_COLUMNS)
    code_pos = headers.index("sales location")
    category_pos = code_pos - 1 if code_pos >= 1 else None
    name_pos = code_pos + 1 if code_pos + 1 < len(headers) else None
    physical_pos = headers.index("physical location") if "physical location" in headers else None
    share_pos = {city: headers.index(city.lower()) for city in FOS_CITIES if city.lower() in headers}

    def cell(row, pos):
        if pos is None:
            return None
        value = row.iloc[pos]
        return None if pd.isna(value) or str(value).strip() in ("", "None", "nan") else value

    rows = []
    for _, row in frame.iterrows():
        code = cell(row, code_pos)
        if code is None:
            continue
        category = str(cell(row, category_pos) or "Retail")
        physical = str(cell(row, physical_pos) or "").strip().upper()
        record = {
            "Code": str(code).strip(),
            "Location": str(cell(row, name_pos) or code).strip(),
            "Channel": "Commercial" if category.strip().lower().startswith("com") else "Retail",
            "City": physical if physical in FOS_CITIES else "Split",
        }
        for city in FOS_CITIES:
            value = cell(row, share_pos.get(city))
            try:
                record[city] = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                record[city] = 0.0
        rows.append(record)
    return pd.DataFrame(rows, columns=FOS_LOCATION_COLUMNS)


def warehouses_from_frame(frame: pd.DataFrame | None) -> dict[str, str]:
    if frame is None or frame.empty:
        return {}
    site_col = detect_column(frame.columns, ["site", "location", "warehouse", "code"])
    city_col = detect_column(frame.columns, ["city", "physical location", "state"], exclude=(site_col or "",))
    if not site_col or not city_col:
        return {}
    out = {}
    for site, city in zip(frame[site_col], frame[city_col]):
        code = normalise_site_code(site)
        value = str(city).strip() if pd.notna(city) else ""
        resolved = value.upper() if value.upper() in FOS_CITIES else FOS_CITY_BY_NAME.get(value.lower(), "")
        if code and resolved:
            out[code] = resolved
    return out


def lookup_frame(frame: pd.DataFrame | None, key_candidates: list[str], value_candidates: list[str]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=object)
    key_col = detect_column(frame.columns, key_candidates)
    value_col = detect_column(frame.columns, value_candidates, exclude=(key_col or "",))
    if not key_col or not value_col:
        return pd.Series(dtype=object)
    keys = frame[key_col].map(sku_text)
    values = frame[value_col].astype(str).str.strip()
    series = pd.Series(values.to_numpy(), index=keys.to_numpy())
    series = series.loc[series.index.astype(str) != ""]
    return series[~series.index.duplicated(keep="first")]


def confirmed_orders_from_frame(frame: pd.DataFrame | None) -> dict[str, float]:
    if frame is None or frame.empty:
        return {}
    sku_col = detect_column(frame.columns, SKU_HEADERS)
    qty_col = detect_column(frame.columns, ["confirmed qty", "confirmed order qty", "confirmed", "qty", "quantity"], exclude=(sku_col or "",))
    if not sku_col or not qty_col:
        return {}
    out = {}
    for sku, qty in zip(frame[sku_col].map(sku_text), to_number(frame[qty_col])):
        if sku and pd.notna(qty) and qty > 0:
            out[sku] = float(qty)
    return out


def purchasing_state() -> dict:
    """Session-held purchasing parameters (sales locations, warehouse cities, settings), seeded from the mapping file."""
    init_backend_state()
    if "purchasing_locations" not in st.session_state:
        saved = load_backend_mapping_file().get("purchasing", {})
        st.session_state["purchasing_locations"] = saved.get("sales_locations") or [dict(r) for r in DEFAULT_SALES_LOCATIONS]
        st.session_state["purchasing_warehouses"] = {
            normalise_site_code(k): v for k, v in (saved.get("warehouse_cities") or DEFAULT_WAREHOUSE_CITIES).items() if v in FOS_CITIES
        }
        settings = dict(DEFAULT_PURCHASING_SETTINGS)
        settings.update({k: v for k, v in (saved.get("settings") or {}).items() if k in settings})
        st.session_state["purchasing_settings"] = settings
        st.session_state["purchasing_confirmed"] = {}
        st.session_state["purchasing_editor_version"] = 0
        st.session_state["purchasing_file_defaults_applied"] = False
    return {
        "locations": st.session_state["purchasing_locations"],
        "warehouses": st.session_state["purchasing_warehouses"],
        "settings": st.session_state["purchasing_settings"],
    }


def apply_purchasing_files(files: dict[str, pd.DataFrame]) -> None:
    """Parameter workbooks under Raw_data/purchasing seed the session once (the file wins over built-in defaults)."""
    if st.session_state.get("purchasing_file_defaults_applied"):
        return
    st.session_state["purchasing_file_defaults_applied"] = True
    saved = load_backend_mapping_file().get("purchasing", {})
    if "Sales Locations" in files and not saved.get("sales_locations"):
        frame = files["Sales Locations"]
        if set(FOS_LOCATION_COLUMNS[:4]).issubset(set(frame.columns)):
            st.session_state["purchasing_locations"] = sales_locations_frame(frame.to_dict("records")).to_dict("records")
    if "Warehouses" in files and not saved.get("warehouse_cities"):
        cities = warehouses_from_frame(files["Warehouses"])
        if cities:
            st.session_state["purchasing_warehouses"] = cities
    if "Confirmed Orders" in files and not st.session_state.get("purchasing_confirmed"):
        st.session_state["purchasing_confirmed"] = confirmed_orders_from_frame(files["Confirmed Orders"])


def purchasing_payload() -> dict:
    return {
        "sales_locations": st.session_state.get("purchasing_locations", []),
        "warehouse_cities": st.session_state.get("purchasing_warehouses", {}),
        "settings": st.session_state.get("purchasing_settings", {}),
    }


@st.cache_data(show_spinner=False, max_entries=4, **BY_TOKEN)
def sku_descriptions(sales: pd.DataFrame) -> pd.Series:
    """Latest description seen in the sales lines, per SKU (the cleaned sales are sorted newest first)."""
    if sales.empty or "Description" not in sales.columns:
        return pd.Series(dtype=object)
    key_column = "SKU Key" if "SKU Key" in sales.columns else "SKU"
    keys = map_unique(sales[key_column], sku_text)
    described = sales["Description"].astype(str).str.strip()
    frame = pd.DataFrame({"SKU": keys, "Description": described}).loc[lambda f: f["SKU"].ne("") & f["Description"].ne("")]
    return frame.groupby("SKU", sort=False)["Description"].first()


def pinned_text_column():
    try:
        return st.column_config.TextColumn(pinned=True)
    except TypeError:
        return st.column_config.TextColumn()


def fos_history_table(history: pd.DataFrame, sku: str, periods: list[str]) -> pd.DataFrame:
    """The workbook's sales block for one SKU: month rows × city/channel columns with Nat and Total."""
    part = history.loc[history["SKU"].eq(sku)]
    columns = []
    for city in FOS_CITIES:
        for channel in FOS_CHANNELS:
            columns.append(f"{city} {channel}")
    table = pd.DataFrame(0.0, index=periods, columns=columns)
    for row in part.itertuples(index=False):
        column = f"{row.City} {row.Channel}"
        if column in table.columns and row.Period in table.index:
            table.loc[row.Period, column] += row.Qty
    table["Nat Retail"] = table[[f"{c} Retail" for c in FOS_CITIES]].sum(axis=1)
    table["Nat Commercial"] = table[[f"{c} Commercial" for c in FOS_CITIES]].sum(axis=1)
    table["TOTAL"] = table["Nat Retail"] + table["Nat Commercial"]
    table.index = [pd.Period(p).strftime("%b %Y") for p in table.index]
    table.index.name = "Month"
    return table


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
    ["Forecast", "Inventory Analysis", "Purchasing", "Backend Mapping", "Options Data", "Formatted Data", "Document Status"],
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
                        help="Company-wide annual Sale target for the year in view. Used when no targets file is loaded.",
                    )
                    show_target_line = st.toggle("Show target line", value=True, key="show_target_line")
                    st.caption(
                        "Monthly targets come from the newest file in Raw_data/targets when there is one "
                        "(month plus optional location / channel / supplier / classification / class / model / SKU "
                        "columns); otherwise the Sale Target is spread by last year's months. "
                        "Either way the line follows the filters above."
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
                    ytd_act = float(actual_series.reindex(ytd_periods).mean()) if ytd_periods else np.nan
                    ytd_fc = float(fc_series.reindex(ytd_periods).mean()) if ytd_periods else np.nan
                    ytg_fc = float(fc_series.reindex(ytg_periods).mean()) if ytg_periods else np.nan
                    fy_act_fc = float(pd.concat([actual_series.reindex(ytd_periods), fc_series.reindex(ytg_periods)]).mean())
                    fy_fc = float(fc_series.reindex(this_year_periods).mean())
                else:
                    ytd_act = float(actual_series.reindex(ytd_periods).sum()) if ytd_periods else 0.0
                    ytd_fc = float(fc_series.reindex(ytd_periods).sum()) if ytd_periods else 0.0
                    ytg_fc = float(fc_series.reindex(ytg_periods).sum()) if ytg_periods else 0.0
                    fy_act_fc = ytd_act + ytg_fc
                    fy_fc = float(fc_series.reindex(this_year_periods).sum())

                ytd_diff = ytd_act - ytd_fc if pd.notna(ytd_act) and pd.notna(ytd_fc) else np.nan
                ytd_pct = (ytd_diff / ytd_fc * 100) if pd.notna(ytd_diff) and ytd_fc != 0 else np.nan

                fy_diff = fy_act_fc - fy_fc if pd.notna(fy_act_fc) and pd.notna(fy_fc) else np.nan
                fy_pct = (fy_diff / fy_fc * 100) if pd.notna(fy_diff) and fy_fc != 0 else np.nan

                fmt_str = lambda x: f"{x:.1f}%" if metric == "GP%" else f"{forecast_label_prefix}{x:,.0f}"

                prior_label = view_label(view_this_year - 1, fiscal)
                prior_year_periods = view_year_periods(view_this_year - 1, fiscal)
                if metric == "GP%":
                    prior_fy_act = float(actual_series.reindex(prior_year_periods).mean())
                else:
                    prior_fy_act = float(actual_series.reindex(prior_year_periods).sum())

                # Monthly targets for the year in view (filter-aware)
                targets_long, target_dimensions = parse_targets(sheets.get("Targets"))
                share_periods = pd.period_range(end=last_complete, periods=SEASON, freq="M")
                company_series = company_monthly_series(sales_data, metric, current_period.to_timestamp())
                target_series, target_basis = resolve_monthly_targets(
                    targets_long, target_dimensions, sales_data, company_series, actual_series, filters, metric,
                    this_year_periods, prior_year_periods, share_periods, float(sale_target),
                )
                next_target_series = None
                if not targets_long.empty:
                    next_target_series, _ = resolve_monthly_targets(
                        targets_long, target_dimensions, sales_data, company_series, actual_series, filters, metric,
                        view_year_periods(view_this_year + 1, fiscal), this_year_periods, share_periods, 0.0,
                    )
                has_target = target_series is not None and target_series.notna().any()
                if has_target:
                    if metric == "GP%":
                        ytd_target = float(target_series.reindex(ytd_periods).mean()) if ytd_periods else np.nan
                        fy_target = float(target_series.mean())
                    else:
                        ytd_target = float(target_series.reindex(ytd_periods).fillna(0.0).sum()) if ytd_periods else 0.0
                        fy_target = float(target_series.fillna(0.0).sum())
                    ytd_target_diff = ytd_act - ytd_target if pd.notna(ytd_act) and pd.notna(ytd_target) else np.nan
                    ytd_target_pct = (ytd_target_diff / ytd_target * 100) if pd.notna(ytd_target_diff) and ytd_target != 0 else np.nan
                    fy_target_diff = fy_act_fc - fy_target if pd.notna(fy_act_fc) and pd.notna(fy_target) else np.nan
                    fy_target_pct = (fy_target_diff / fy_target * 100) if pd.notna(fy_target_diff) and fy_target != 0 else np.nan
                else:
                    ytd_target = fy_target = ytd_target_diff = ytd_target_pct = fy_target_diff = fy_target_pct = np.nan

                year_start_label = this_year_periods[0].strftime("%b %Y")
                ytd_end_label = ytd_periods[-1].strftime("%b %Y") if ytd_periods else "n/a"
                baseline_help = (
                    f"Blended forecast for all 12 months of {this_label}"
                    + (
                        f" (Jul–Dec from the {this_year} forecast, Jan–Jun from the {this_year + 1} forecast)."
                        if fiscal
                        else "."
                    )
                )

                # Headline cards
                card_left, card_right = st.columns([2, 3], gap="small")
                with card_left:
                    metric_card(
                        "Forecast",
                        [
                            dict(
                                label="Rolling 12-Month Forecast",
                                value=rolling_display,
                                help=f"{window_label(rolling_window)} · "
                                + ("average GP%" if metric == "GP%" else f"total {metric}"),
                            ),
                            dict(
                                label="Historical wMAPE",
                                value=f"{mape:.1%}" if mape is not None else "n/a",
                                help=f"Rolling previous 12 months (Weighted) · {window_label(result['mape_window'])}",
                            ),
                        ],
                    )
                with card_right:
                    metric_card(
                        "Stock & supply",
                        [
                            dict(label="On hand · all sites", value=f"{on_hand:,.0f}"),
                            dict(label="Free stock · all sites", value=f"{free_stock:,.0f}"),
                            dict(
                                label="Open PO quantity",
                                value=f"{total_po:,.0f}",
                                help=f"Open POs due by {rolling_window[-1].strftime('%b %Y')}, including overdue.",
                            ),
                        ],
                    )

                metric_card(
                    f"{this_label} forecast vs actual",
                    [
                        dict(
                            label="YTD Actual vs Forecast",
                            value=fmt_str(ytd_act),
                            delta=f"{ytd_diff:+.0f} ({ytd_pct:+.1f}%)" if pd.notna(ytd_diff) else "n/a",
                            help=f"Actuals from {year_start_label} to {ytd_end_label} vs blended forecast.",
                        ),
                        dict(
                            label="YTG Forecast Remaining",
                            value=fmt_str(ytg_fc),
                            help=f"Blended forecast for remaining months of {this_label}.",
                        ),
                        dict(
                            label="Full Year Outlook (Actual + Forecast)",
                            value=fmt_str(fy_act_fc),
                            delta=f"{fy_diff:+.0f} ({fy_pct:+.1f}%) vs Baseline" if pd.notna(fy_diff) else "n/a",
                            help="YTD Actuals + YTG Forecast vs full-year forecast baseline.",
                        ),
                        dict(label="Full Year Baseline Forecast", value=fmt_str(fy_fc), help=baseline_help),
                    ]
                    + (
                        [
                            dict(
                                label="YTD Actual vs Target",
                                value=fmt_str(ytd_target),
                                delta=f"{ytd_target_diff:+,.0f} ({ytd_target_pct:+.1f}%)" if pd.notna(ytd_target_diff) else "n/a",
                                help=f"Target for {year_start_label} to {ytd_end_label} for this selection · {target_basis}",
                            )
                        ]
                        if has_target
                        else []
                    ),
                )

                profile_from = pd.Timestamp(f"{history_from_year - 1}-01-01") if fiscal else forecast_from
                annual_profile, note = engine_call(
                    "yearly_monthly_profile",
                    local_yearly_monthly_profile,
                    selected_sales,
                    pd.DataFrame(columns=["Date", "Forecast"]),
                    metric,
                    profile_from,
                )
                engine_notes.append(note)
                if fiscal:
                    annual_profile = fiscal_profile(annual_profile, int(history_from_year))
                if metric == "GP%" and not annual_profile.empty:
                    for col in [c for c in annual_profile.columns if c != "Month"]:
                        annual_profile[col] = annual_profile[col].mean()

                year_columns = sorted(
                    (column for column in annual_profile.columns if column != "Month"),
                    key=lambda value: int(value),
                )
                year_labels = [view_label(int(column), fiscal) for column in year_columns]
                month_sort_order = FY_MONTHS if fiscal else MONTHS

                st.subheader(
                    "Monthly comparison by " + ("financial year (Jul–Jun)" if fiscal else "calendar year (Jan–Dec)"),
                )

                actual_chart_data = annual_profile.melt(
                    id_vars="Month", var_name="Year", value_name="Value"
                ).dropna(subset=["Value"])
                actual_chart_data["Series"] = actual_chart_data["Year"].map(lambda y: view_label(int(y), fiscal))
                actual_chart_data["Line Type"] = "Actual"

                forecast_frames = [
                    frame.dropna(subset=["Forecast"])
                    for frame in (forecast_this_year, forecast_next_year)
                    if not frame.empty
                ]
                forecast_frames = [frame for frame in forecast_frames if not frame.empty]
                if forecast_frames:
                    forecast_chart_data = pd.concat(forecast_frames, ignore_index=True)
                    forecast_periods = forecast_chart_data["Date"].dt.to_period("M")
                    forecast_chart_data["YearNumber"] = forecast_periods.map(lambda p: view_year(p, fiscal))
                    forecast_chart_data = forecast_chart_data.loc[
                        forecast_chart_data["YearNumber"] >= view_this_year
                    ].copy()
                    forecast_chart_data["Year"] = forecast_chart_data["YearNumber"].map(lambda y: view_label(int(y), fiscal))
                    forecast_chart_data["Month"] = forecast_chart_data["Date"].dt.month.map(MONTH_NUMBER_TO_NAME)
                else:
                    forecast_chart_data = pd.DataFrame(columns=["Date", "Forecast", "YearNumber", "Year", "Month"])
                forecast_plot_data = pd.DataFrame(
                    {
                        "Month": forecast_chart_data["Month"],
                        "Series": "Forecast " + forecast_chart_data["Year"].astype(str),
                        "Line Type": "Forecast",
                        "Value": forecast_chart_data["Forecast"],
                    }
                )
                target_frames = []
                if show_target_line:
                    for year_number, series in ((view_this_year, target_series), (view_this_year + 1, next_target_series)):
                        if series is None or not series.notna().any():
                            continue
                        clean = series.dropna()
                        target_frames.append(
                            pd.DataFrame(
                                {
                                    "Month": [MONTH_NUMBER_TO_NAME[p.month] for p in clean.index],
                                    "Series": f"Target {view_label(year_number, fiscal)}",
                                    "Line Type": "Target",
                                    "Value": clean.to_numpy(),
                                }
                            )
                        )
                target_plot_data = (
                    pd.concat(target_frames, ignore_index=True)
                    if target_frames
                    else pd.DataFrame(columns=["Month", "Series", "Line Type", "Value"])
                )
                chart_data = pd.concat(
                    [
                        frame
                        for frame in (actual_chart_data[["Month", "Series", "Line Type", "Value"]], forecast_plot_data, target_plot_data)
                        if not frame.empty
                    ],
                    ignore_index=True,
                )
                forecast_series = sorted(forecast_plot_data["Series"].unique(), key=label_year_number)
                target_series_labels = sorted(target_plot_data["Series"].unique(), key=label_year_number)
                series_order = list(dict.fromkeys(year_labels + forecast_series))
                palette = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#f0e442", "#000000"]
                series_colors = [palette[index % len(palette)] for index in range(len(series_order))]
                target_palette = ["#1f2937", "#6b7280"]
                series_order += target_series_labels
                series_colors += [target_palette[index % len(target_palette)] for index in range(len(target_series_labels))]
                axis_format, value_format, y_title = chart_formats(metric)

                legend_selection = alt.selection_point(
                    name="series_legend",
                    fields=["Series"],
                    bind="legend",
                    toggle="true",
                    clear=False,
                )

                month_x_axis = alt.X(
                    "Month:N",
                    sort=month_sort_order,
                    title=None,
                    axis=quiet_axis(labelAngle=0),
                )

                value_y_axis = alt.Y(
                    "Value:Q",
                    title=y_title,
                    scale=alt.Scale(zero=False),
                    axis=value_axis(axis_format),
                )

                yearly_chart = (
                    alt.Chart(chart_data)
                    .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2)
                    .encode(
                        x=month_x_axis,
                        y=value_y_axis,
                        color=alt.Color(
                            "Series:N",
                            sort=series_order,
                            scale=alt.Scale(domain=series_order, range=series_colors),
                            legend=alt.Legend(
                                title=None, orient="top", direction="horizontal", symbolType="stroke",
                                symbolStrokeWidth=3, labelFontSize=11, padding=0, offset=6,
                            ),
                        ),
                        strokeDash=alt.StrokeDash(
                            "Line Type:N",
                            scale=alt.Scale(domain=["Actual", "Forecast", "Target"], range=[[1, 0], [7, 4], [2, 3]]),
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

                series_labels = [view_label(int(year), fiscal) for year in year_columns]
                forecast_labels = [f"Forecast {year}" for year in sorted(forecast_chart_data["Year"].unique(), key=label_year_number)]
                forecast_lookup = {
                    (row["Year"], row["Month"]): row["Forecast"]
                    for _, row in forecast_chart_data.iterrows()
                    if pd.notna(row["Forecast"])
                }
                target_lookup = {
                    (row["Series"], row["Month"]): row["Value"] for _, row in target_plot_data.iterrows() if pd.notna(row["Value"])
                }
                profile_by_month = annual_profile.set_index("Month") if not annual_profile.empty else pd.DataFrame()
                prefix, python_spec = ("$", value_format[1:]) if value_format.startswith("$") else ("", value_format)
                hover_rows = []
                for month in month_sort_order:
                    row = {"Month": month}
                    for year, label in zip(year_columns, series_labels):
                        value = profile_by_month.at[month, year] if month in profile_by_month.index and year in profile_by_month.columns else np.nan
                        row[label] = f"{prefix}{format(float(value), python_spec)}" if pd.notna(value) else ""
                    for label in forecast_labels:
                        value = forecast_lookup.get((label.removeprefix("Forecast "), month))
                        row[label] = f"{prefix}{format(float(value), python_spec)}" if value is not None else ""
                    for label in target_series_labels:
                        value = target_lookup.get((label, month))
                        row[label] = f"{prefix}{format(float(value), python_spec)}" if value is not None else ""
                    hover_rows.append(row)
                hover_frame = pd.DataFrame(hover_rows)
                hover_fields = [
                    label for label in series_labels + forecast_labels + target_series_labels if hover_frame[label].ne("").any()
                ]
                month_hover_selection = alt.selection_point(
                    name="hover_month", fields=["Month"], on="pointerover", clear="pointerout",
                )
                month_hover_charts = []
                if hover_fields:
                    month_hover_charts.append(
                        alt.Chart(hover_frame)
                        .mark_rule(color="#667085", strokeWidth=12)
                        .encode(
                            x=month_x_axis,
                            opacity=alt.condition(month_hover_selection, alt.value(0.18), alt.value(0), empty=False),
                            tooltip=[alt.Tooltip("Month:N")] + [alt.Tooltip(f"{label}:N", title=label) for label in hover_fields],
                        )
                        .add_params(month_hover_selection)
                    )

                st.altair_chart(
                    alt.layer(yearly_chart, *month_hover_charts).properties(height=360).configure_view(strokeWidth=0),
                    **FULL_WIDTH,
                )
                st.caption(
                    f"{'FY Jul–Jun' if fiscal else 'Calendar Jan–Dec'} · Solid lines: actual · Dashed lines: forecast"
                    + (" · Dotted line: target" if target_series_labels else "")
                    + " · Click a legend entry to show or hide a series · Hover a month to compare every year."
                )
                if show_target_line:
                    if has_target:
                        st.caption(f"🎯 Target {this_label} for this selection · {target_basis}")
                    else:
                        st.caption(f"🎯 No target line · {target_basis}")

                with st.expander("📊 Monthly Data Table", expanded=True):
                    pivot_data = chart_data.pivot(index="Month", columns="Series", values="Value").reindex(month_sort_order)
                    pivot_data = pivot_data.reindex(columns=[col for col in series_order if col in pivot_data.columns])
                    pivot_data = pivot_data.T

                    column_config_table = {}
                    for col in pivot_data.columns:
                        if metric in ["Sale", "Cost", "GP $"]:
                            column_config_table[col] = st.column_config.NumberColumn(col, format="$%,.0f")
                        elif metric == "GP%":
                            column_config_table[col] = st.column_config.NumberColumn(col, format="%.1f%%")
                        else:
                            column_config_table[col] = st.column_config.NumberColumn(col, format=",.0f")

                    st.dataframe(
                        pivot_data.reset_index().rename(columns={"index": "Series"}),
                        hide_index=True,
                        column_config=column_config_table,
                        height=table_height(len(pivot_data)),
                        **FULL_WIDTH,
                    )

                    st.download_button(
                        "Download forecast table CSV",
                        pivot_data.reset_index().rename(columns={"index": "Series"}).to_csv(index=False).encode("utf-8"),
                        file_name="monthly_forecast_comparison.csv",
                        mime="text/csv",
                        key="download_monthly_forecast_table",
                    )

                if has_target and metric != "GP%":
                    target_80 = fy_target * 0.8
                    target_80_diff = fy_act_fc - target_80
                    target_80_pct = (target_80_diff / target_80) * 100 if target_80 > 0 else np.nan

                    with st.expander(f"🎯 Target & performance comparison · {this_label}", expanded=False):
                        st.caption(f"Targets for this selection · {target_basis}")
                        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5, gap="small")
                        t_col1.metric(
                            "Full Year Outlook", fmt_str(fy_act_fc),
                            delta=f"{fy_target_diff:+,.0f} ({fy_target_pct:+.1f}% vs Target)" if pd.notna(fy_target_diff) else "n/a",
                        )
                        t_col2.metric(
                            f"100% {metric} Target", fmt_str(fy_target),
                            delta=f"{fy_target_pct:+.1f}% to Target" if pd.notna(fy_target_pct) else "n/a",
                        )
                        t_col3.metric(
                            f"80% {metric} Target", fmt_str(target_80),
                            delta=f"{target_80_pct:+.1f}% to 80% Target" if pd.notna(target_80_pct) else "n/a",
                        )
                        t_col4.metric(
                            "YTD Actual vs Target", fmt_str(ytd_act),
                            delta=f"{ytd_target_diff:+,.0f} ({ytd_target_pct:+.1f}%)" if pd.notna(ytd_target_diff) else "n/a",
                            help=f"YTD target {fmt_str(ytd_target)} · {year_start_label} to {ytd_end_label}",
                        )
                        t_col5.metric(f"{prior_label} Full Year Actuals", fmt_str(prior_fy_act))

            for note in [n for n in engine_notes if n]:
                st.warning(note)

        except Exception as e:
            st.error(f"Something went wrong while rendering the forecast: {e}")

# =============================================================================
# Inventory Analysis tab
# =============================================================================
elif selected_tab == "Inventory Analysis":
    prepared_data, source_names = load_prepared_data()

    if prepared_data is None:
        st.info("Add inventory files under Raw_data/inventory and sales workbooks to analyse inventory.")
    else:
        try:
            sheets = prepared_data
            sales_data = sheets["Sales Data"]
            inventory_raw = sheets["Current Inventory"]
            options_raw = sheets.get("OptionsData")
            purchase_orders_raw = sheets.get("Purchase Orders")

            today = pd.Timestamp.today().normalize()
            current_period = today.to_period("M")
            last_complete = current_period - 1

            loans_raw = sheets.get("Loans", pd.DataFrame())
            loan_mapping = resolve_loan_mapping(loans_raw) if not loans_raw.empty else {}
            loan_file_name = next((Path(name).name for name in source_names if "loan" in str(name).lower()), "the loan file")

            attributes = option_attributes(options_raw)
            mapping = resolve_inventory_mapping(inventory_raw)
            site_mapping = current_site_mapping()
            cost_source = st.session_state.get("cost_source", "auto")

            with st.sidebar:
                st.header("Inventory filters")
                st.markdown("**Product**")
                filter_sources = {
                    "Supplier": ("Supplier", "OptionsData.Supplier"),
                    "Class": ("Class", "OptionsData.Class"),
                    "Product group": ("Product group", "OptionsData.Fourth Product Group"),
                }
                product_filters: dict[str, list[str]] = {}
                product_filters["SKU type"] = st.multiselect("SKU type", SKU_TYPES, default=[], key="inventory_filter_SKU type",
                                                             help="From the stock-code pattern; rules on the Backend Mapping tab.")
                for label, (attribute_column, sales_column) in filter_sources.items():
                    values = sorted(
                        set(column_choices(attributes, attribute_column)) | set(column_choices(sales_data, sales_column)),
                        key=str.casefold,
                    )
                    product_filters[label] = st.multiselect(label, values, default=[], key=f"inventory_filter_{label}")
                description_search = st.text_input(
                    "SKU or description contains", key="inventory_filter_text",
                    help="End with * to match a prefix, e.g. 123*",
                ).strip()

                with st.expander("Inventory settings", expanded=False):
                    window_months = st.selectbox(
                        "Sales window (complete months)", [3, 6, 12], index=2, key="inventory_window_months",
                    )
                    dead_days = st.number_input(
                        "Dead stock: no sale for (days)", min_value=30, max_value=1095, value=180, step=30,
                        key="inventory_dead_days",
                    )
                    slow_dos = st.number_input(
                        "Slow moving: days of supply above", min_value=30, max_value=3650, value=365, step=30,
                        key="inventory_slow_dos",
                    )
                    risk_dos = st.number_input(
                        "Stock-out risk: days of supply below", min_value=7, max_value=365, value=60, step=7,
                        key="inventory_risk_dos",
                    )
                    showroom_target = float(st.number_input(
                        "Showroom value target ($)", min_value=0.0, value=4_000_000.0, step=100_000.0,
                        key="inventory_showroom_target",
                    ))
                    showroom_tolerance = float(st.number_input(
                        "Showroom target tolerance (%)", min_value=1.0, max_value=50.0, value=5.0, step=1.0,
                        key="inventory_showroom_tolerance", help="Within this band either side of the target shows green.",
                    )) / 100
                st.caption("Site categories and export columns are set on the **Backend Mapping** tab.")

            if not mapping["sku"] or not mapping["on_hand"]:
                st.warning(
                    "The SKU and on-hand quantity columns of the inventory export could not be identified. "
                    "Choose them on the **Backend Mapping** tab."
                )
            else:
                window_end = last_complete
                window_start = last_complete - (window_months - 1)
                window_periods = pd.period_range(window_start, window_end, freq="M")
                window_days = int(sum(period.days_in_month for period in window_periods))

                inventory_sku = inventory_by_sku(inventory_raw, mapping, site_mapping)
                movement = sku_movement(sales_data, str(window_start), str(window_end))
                open_po = open_po_by_sku(purchase_orders_raw)
                master = build_inventory_table(
                    inventory_sku, movement, attributes, open_po,
                    window_months, window_days, today, int(dead_days), int(slow_dos), int(risk_dos),
                    cost_source=cost_source,
                )
                cost_by_sku = master.set_index("SKU")["Avg option cost"]
                loan_lines = classify_loan_lines(loans_raw, loan_mapping, site_mapping, cost_by_sku, today)
                if loan_lines.empty:
                    master["On loan units"] = 0.0
                else:
                    on_loan = loan_lines.loc[loan_lines["Type"].eq("Loan")].groupby("SKU")["Qty"].sum()
                    master["On loan units"] = master["SKU"].map(on_loan).fillna(0.0)

                filtered = master
                for label, chosen in product_filters.items():
                    if chosen:
                        filtered = filtered.loc[filtered[label].isin(chosen)]
                if description_search:
                    if description_search.endswith("*"):
                        pattern, is_regex = f"^{re.escape(description_search[:-1])}", True
                    else:
                        pattern, is_regex = description_search, False
                    text_match = (
                        filtered["SKU"].str.contains(pattern, case=False, regex=is_regex, na=False)
                        | filtered["Description"].str.contains(pattern, case=False, regex=is_regex, na=False)
                    )
                    filtered = filtered.loc[text_match]
                filtered = filtered.reset_index(drop=True)

                site_units = units_by_site(inventory_raw, mapping)
                showroom_units = float(sum(units for code, units in site_units.items() if category_for(code, site_mapping) == "Showroom"))
                ignored_units = float(sum(units for code, units in site_units.items() if category_for(code, site_mapping) == "Ignored"))
                loan_units = float(sum(units for code, units in site_units.items() if category_for(code, site_mapping) == "Loan"))
                unmapped_codes = sorted(code for code in site_units.index if code and category_for(code, site_mapping) == "Unmapped")

                context_col, group_col = st.columns([5, 2], gap="small", vertical_alignment="center")
                with group_col:
                    group_by = st.selectbox(
                        "Group by", ["SKU", "Supplier", "Class", "Product group", "SKU type"], index=0,
                        key="inventory_group_by", label_visibility="collapsed",
                    )
                with context_col:
                    st.caption(
                        f"Warehouse & clearance stock only · showroom excluded ({showroom_units:,.0f} units)"
                        + (f" · loan stock excluded ({loan_units:,.0f} units)" if loan_units else "")
                        + (f" · ignored sites skipped ({ignored_units:,.0f} units)" if ignored_units else "")
                        + f" · sales window {window_label(window_periods)} ({window_days} days) · {len(filtered):,} SKUs"
                    )
                if not mapping.get("site"):
                    st.warning(
                        "The inventory export has no site column, so every row is treated as available to sell. "
                        "Pick the site column on the **Backend Mapping** tab to exclude showroom stock."
                    )
                elif unmapped_codes:
                    st.warning(
                        f"{len(unmapped_codes)} site code(s) in the inventory export are not categorised and are "
                        f"excluded from stock health: {', '.join(unmapped_codes)}. Categorise them on the **Backend Mapping** tab "
                        f"(choose *Ignored* for sites that should not be tracked)."
                    )

                if filtered.empty:
                    st.warning("No SKUs match these filters.")
                else:
                    total_on_hand = float(filtered["On hand"].sum())
                    total_free = float(filtered["Free stock"].sum())
                    total_units = float(filtered["Units sold"].sum())
                    total_value = float(filtered["Total stock value"].sum(skipna=True))
                    free_value = float(filtered["Free stock value"].sum(skipna=True))
                    dead_units = float(filtered["Dead units"].sum())
                    slow_units = float(filtered["Slow units"].sum())
                    dead_value = float(filtered.loc[filtered["Status"] == "Dead stock", "Total stock value"].sum(skipna=True))
                    slow_value = float(filtered.loc[filtered["Status"] == "Slow moving", "Total stock value"].sum(skipna=True))
                    allocated_units = float(filtered["Allocated units"].sum())
                    allocated_value = float(filtered.loc[filtered["Status"] == "Allocated", "Total stock value"].sum(skipna=True))
                    active_skus = int(filtered["Active"].sum())
                    status_counts = filtered["Status"].value_counts()
                    costed_share = float(filtered.loc[filtered["On hand"] > 0, "Avg option cost"].notna().mean()) if (filtered["On hand"] > 0).any() else 0.0
                    overall_daily_rate = total_units / window_days if window_days else 0.0
                    overall_dos = total_on_hand / overall_daily_rate if overall_daily_rate > 0 else np.nan
                    overall_turns = (total_units * 12 / window_months) / total_on_hand if total_on_hand > 0 else np.nan

                    def pct_of_on_hand(units: float) -> str:
                        return f"{units / total_on_hand * 100:.0f}% of on hand" if total_on_hand > 0 else "n/a"

                    totals = company_wide_totals(inventory_raw, mapping, site_mapping, cost_by_sku)
                    by_category = totals["by_category"]
                    units_breakdown = " · ".join(
                        f"{category} {by_category.loc[category, 'On hand']:,.0f}"
                        for category in SITE_CATEGORIES if by_category.loc[category, "On hand"] > 0
                    )
                    sellable_value = float(by_category.loc[["Warehouse", "Clearance"], "Value"].sum())
                    clearance_value = float(by_category.loc["Clearance", "Value"])
                    clearance_units = float(by_category.loc["Clearance", "On hand"])
                    untracked_value = float(by_category.loc[["Ignored", "Unmapped"], "Value"].sum())
                    untracked_units = float(by_category.loc[["Ignored", "Unmapped"], "On hand"].sum())
                    at_risk = master.loc[master["Status"].isin(["Dead stock", "Slow moving"])]
                    at_risk_value = float(at_risk["Total stock value"].sum(skipna=True))
                    on_order_units = float(open_po.sum()) if not open_po.empty else 0.0
                    on_order_value = float((open_po * open_po.index.map(cost_by_sku)).sum(skipna=True)) if not open_po.empty else 0.0

                    loans = loan_lines.loc[loan_lines["Type"].eq("Loan")]
                    transfers = loan_lines.loc[loan_lines["Type"].eq("Transfer")]
                    if not loan_lines.empty:
                        loan_value, loan_units = float(loans["Value"].sum(skipna=True)), float(loans["Qty"].sum())
                        loan_help = (
                            f"{loan_units:,.0f} units on {len(loans):,} open loan lines to {loans['Customer'].replace('', np.nan).nunique():,} customers, "
                            f"from {loan_file_name} · inventory Loan sites hold {totals['loan_units']:,.0f} units (${totals['loan_value']:,.0f})."
                        )
                    else:
                        loan_value, loan_units = totals["loan_value"], totals["loan_units"]
                        loan_help = (
                            f"{loan_units:,.0f} units at sites categorised as Loan. Add a loan file under Raw_data/loan for "
                            "customer-level detail; codes containing LOAN are picked up automatically."
                        )
                    transfer_value, transfer_units = float(transfers["Value"].sum(skipna=True)), float(transfers["Qty"].sum())

                    wide_col, gauge_col = st.columns([3, 1.25], gap="small")
                    with wide_col:
                        metric_card(
                            "Company-wide stock · all sites, all SKUs, before filters",
                            [
                                dict(label="Total SOH (units)", value=f"{totals['units']:,.0f}", help=units_breakdown),
                                dict(label="Total SOH value", value=f"${totals['value']:,.0f}", help="Every unit at every site × average option cost."),
                                dict(
                                    label="Sellable value", value=f"${sellable_value:,.0f}",
                                    help=f"Warehouse + clearance sites: {by_category.loc[['Warehouse', 'Clearance'], 'On hand'].sum():,.0f} units.",
                                ),
                                dict(
                                    label="Free stock value", value=f"${totals['free_value']:,.0f}",
                                    help=f"{totals['free_units']:,.0f} free (unallocated) units × average option cost, across all sites.",
                                ),
                                dict(
                                    label="Allocated value", value=f"${totals['allocated_value']:,.0f}",
                                    help=f"{totals['allocated_units']:,.0f} units on hand but not free: committed to sales orders.",
                                ),
                            ],
                        )
                        metric_card(
                            "Holdings & exposure",
                            [
                                dict(label="Loan value", value=f"${loan_value:,.0f}", help=loan_help),
                                dict(
                                    label="Clearance value", value=f"${clearance_value:,.0f}",
                                    help=f"{clearance_units:,.0f} units at clearance sites (XIT*): sellable, but already marked down.",
                                ),
                                dict(
                                    label="On order (open POs)", value=f"${on_order_value:,.0f}",
                                    help=f"{on_order_units:,.0f} units on open purchase orders × average option cost: inbound commitment not yet in stock.",
                                ),
                                dict(
                                    label="Dead & slow value", value=f"${at_risk_value:,.0f}",
                                    help=f"{len(at_risk):,} SKUs classed Dead stock or Slow moving within sellable stock: the value most at risk of write-down.",
                                ),
                                dict(
                                    label="Untracked value", value=f"${untracked_value:,.0f}",
                                    help=f"{untracked_units:,.0f} units at Ignored or Unmapped sites (quarantine, transit, undecided)."
                                    + (f" Plus {transfer_units:,.0f} units (${transfer_value:,.0f}) on open inter-site transfers." if transfer_units else ""),
                                ),
                            ],
                        )
                        if totals["costed_share"] < 0.999:
                            st.caption(f"Values cover {totals['costed_share']:.0%} of stocked units; the rest have no cost from any source.")
                    with gauge_col:
                        with st.container(border=True):
                            st.markdown("<p class='card-title'>Showroom stock value</p>", unsafe_allow_html=True)
                            st.altair_chart(gauge_chart(totals["showroom_value"], showroom_target, showroom_tolerance), **FULL_WIDTH)
                            st.caption(
                                f"{by_category.loc['Showroom', 'On hand']:,.0f} showroom units · green within ±{showroom_tolerance:.0%} of "
                                f"${showroom_target / 1e6:,.1f}M, amber within ±{3 * showroom_tolerance:.0%}."
                            )

                    metric_card(
                        "Available-to-sell stock · warehouse & clearance",
                        [
                            dict(label="Units on hand", value=f"{total_on_hand:,.0f}", help=f"Free stock {total_free:,.0f} units."),
                            dict(label="Active SKUs", value=f"{active_skus:,}", help="SKUs with available stock or sales in the window (every status except Inactive)."),
                            dict(
                                label="Total stock value", value=f"${total_value:,.0f}",
                                help=f"Units on hand × average option cost · {costed_share:.0%} of stocked SKUs have a cost ({COST_SOURCES[cost_source]}).",
                            ),
                            dict(label="Free stock value", value=f"${free_value:,.0f}", help="Free (unallocated) units × average option cost."),
                        ],
                    )

                    health_col, movement_col = st.columns([4, 2], gap="small")
                    with health_col:
                        metric_card(
                            "Stock health",
                            [
                                dict(label="Dead stock", value=f"{dead_units:,.0f}", delta=pct_of_on_hand(dead_units), delta_color="off",
                                     help=f"{int(status_counts.get('Dead stock', 0))} SKUs · ${dead_value:,.0f} · stock with no sale in {int(dead_days)} days."),
                                dict(label="Slow moving", value=f"{slow_units:,.0f}", delta=pct_of_on_hand(slow_units), delta_color="off",
                                     help=f"{int(status_counts.get('Slow moving', 0))} SKUs · ${slow_value:,.0f} · more than {int(slow_dos)} days of supply."),
                                dict(label="Stock-out risk", value=f"{int(status_counts.get('Stock-out risk', 0)):,} SKUs",
                                     help=f"Free stock covers fewer than {int(risk_dos)} days · plus {int(status_counts.get('Out of stock', 0))} SKUs already out of sellable stock with recent sales."),
                                dict(label="Allocated", value=f"{allocated_units:,.0f}", delta=pct_of_on_hand(allocated_units), delta_color="off",
                                     help=f"{int(status_counts.get('Allocated', 0))} SKUs · ${allocated_value:,.0f} · on hand with no free stock: committed to sales orders, never counted as dead or slow."),
                            ],
                        )
                    with movement_col:
                        metric_card(
                            f"Movement · last {window_months} months",
                            [
                                dict(label="Units sold", value=f"{total_units:,.0f}"),
                                dict(label="Days of supply", value=f"{overall_dos:,.0f}" if pd.notna(overall_dos) else "n/a",
                                     help="Available units ÷ average daily units sold in the window."),
                                dict(label="Turns (annualised)", value=f"{overall_turns:.1f}×" if pd.notna(overall_turns) else "n/a"),
                            ],
                        )

                    chart_group = group_by if group_by != "SKU" else "Supplier"
                    stock_health_charts(filtered, chart_group)

                    sku_columns = [
                        "SKU", "Description", "SKU type", "Supplier", "Class", "Product group", "Status",
                        "On hand", "Free stock", "Showroom units", "Avg option cost", "Total stock value", "Free stock value",
                        "Units sold", "Avg monthly units", "Days of supply", "Days of supply (free)", "Turns (annualised)",
                        "Last sale", "Days since last sale", "Sales value", "Cost source",
                    ]
                    if "Open PO qty" in filtered.columns:
                        sku_columns += ["Open PO qty", "Days of supply incl. POs"]
                    if (filtered["Unmapped units"] > 0).any():
                        sku_columns.append("Unmapped units")
                    if (filtered["Ignored units"] > 0).any():
                        sku_columns.append("Ignored units")
                    if (filtered["On loan units"] > 0).any():
                        sku_columns.append("On loan units")
                    sku_columns = [c for c in sku_columns if c in filtered.columns]

                    sku_column_config = {
                        "On hand": st.column_config.NumberColumn("On hand (sellable)", format="%d"),
                        "Free stock": st.column_config.NumberColumn(format="%d"),
                        "Showroom units": st.column_config.NumberColumn(format="%d", help="Display stock; excluded from stock health."),
                        "Unmapped units": st.column_config.NumberColumn(format="%d", help="Stock at site codes that are not categorised."),
                        "Ignored units": st.column_config.NumberColumn(format="%d", help="Stock at sites categorised as Ignored."),
                        "On loan units": st.column_config.NumberColumn(format="%d", help="Open loan lines to customers from the loan file."),
                        "Days of supply (free)": st.column_config.NumberColumn(format="%d", help="Free stock ÷ daily rate; drives the stock-out risk flag."),
                        "Avg option cost": st.column_config.NumberColumn(format="$%.2f"),
                        "Total stock value": st.column_config.NumberColumn(format="$%.0f"),
                        "Free stock value": st.column_config.NumberColumn(format="$%.0f"),
                        "Units sold": st.column_config.NumberColumn(f"Units sold ({window_months}m)", format="%d"),
                        "Avg monthly units": st.column_config.NumberColumn(format="%.1f"),
                        "Days of supply": st.column_config.NumberColumn(format="%d"),
                        "Days of supply incl. POs": st.column_config.NumberColumn(format="%d"),
                        "Turns (annualised)": st.column_config.NumberColumn(format="%.1f"),
                        "Last sale": st.column_config.DatetimeColumn(format="DD MMM YYYY"),
                        "Days since last sale": st.column_config.NumberColumn(format="%d"),
                        "Sales value": st.column_config.NumberColumn(f"Sales ({window_months}m)", format="$%.0f"),
                        "Open PO qty": st.column_config.NumberColumn(format="%d"),
                    }

                    def show_table(frame: pd.DataFrame, file_name: str, key: str) -> None:
                        st.dataframe(
                            frame[sku_columns], hide_index=True, column_config=sku_column_config,
                            height=table_height(len(frame)), **FULL_WIDTH,
                        )
                        st.download_button(
                            "Download CSV", frame[sku_columns].to_csv(index=False).encode("utf-8"),
                            file_name=file_name, mime="text/csv", key=key,
                        )

                    if group_by != "SKU":
                        summary = summarise_inventory(filtered, group_by, window_months, window_days)
                        with st.expander(f"📊 Summary by {group_by.lower()}", expanded=True):
                            st.dataframe(
                                summary,
                                hide_index=True,
                                column_config={
                                    "On hand": st.column_config.NumberColumn("On hand (sellable)", format="%d"),
                                    "Free stock": st.column_config.NumberColumn(format="%d"),
                                    "Showroom units": st.column_config.NumberColumn(format="%d"),
                                    "Units sold": st.column_config.NumberColumn(f"Units sold ({window_months}m)", format="%d"),
                                    "Sales value": st.column_config.NumberColumn(f"Sales ({window_months}m)", format="$%.0f"),
                                    "Total stock value": st.column_config.NumberColumn(format="$%.0f"),
                                    "Free stock value": st.column_config.NumberColumn(format="$%.0f"),
                                    "Dead units": st.column_config.NumberColumn(format="%d"),
                                    "Slow units": st.column_config.NumberColumn(format="%d"),
                                    "Allocated units": st.column_config.NumberColumn(format="%d"),
                                    "Dead stock %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Days of supply": st.column_config.NumberColumn(format="%d"),
                                    "Turns (annualised)": st.column_config.NumberColumn(format="%.1f"),
                                },
                                height=table_height(len(summary)),
                                **FULL_WIDTH,
                            )

                    if not loans_raw.empty:
                        if loan_lines.empty:
                            st.warning(
                                f"{loan_file_name} was read but no loan or transfer lines could be built from it. "
                                "Check the loan file columns on the **Backend Mapping** tab."
                            )
                        else:
                            with st.expander(
                                f"🤝 Stock on loan · {len(loans):,} lines · {loan_units:,.0f} units · ${loan_value:,.0f}"
                                + (f" · plus {len(transfers):,} transfer lines" if len(transfers) else ""),
                                expanded=False,
                            ):
                                loans_section(loan_lines)

                    with st.expander(f"📋 All SKUs · {len(filtered):,}", expanded=False):
                        show_table(filtered, "inventory_analysis.csv", "inventory_download_all")

        except Exception as e:
            st.error(f"Something went wrong while rendering inventory analysis: {e}")

# =============================================================================
# Purchasing tab — Forward Order Schedule
# =============================================================================
elif selected_tab == "Purchasing":
    prepared_data, source_names = load_prepared_data()

    if prepared_data is None:
        st.info("Add the inventory, PO dashboard and sales exports under Raw_data to build the forward order schedule.")
    else:
        try:
            sheets = prepared_data
            sales_data = sheets["Sales Data"]
            inventory_raw = sheets["Current Inventory"]
            options_raw = sheets.get("OptionsData")
            purchase_orders_raw = sheets.get("Purchase Orders")
            purchasing_files = sheets.get("Purchasing Files", {}) or {}

            state = purchasing_state()
            apply_purchasing_files(purchasing_files)
            settings = st.session_state["purchasing_settings"]
            locations_frame = sales_locations_frame(st.session_state["purchasing_locations"])
            warehouse_cities = dict(st.session_state["purchasing_warehouses"])

            attributes = option_attributes(options_raw)
            option_fields = fos_option_fields(options_raw)
            mapping = resolve_inventory_mapping(inventory_raw)
            site_mapping = current_site_mapping()

            today = pd.Timestamp.today().normalize()
            report_date_col = detect_column(inventory_raw.columns, ["report date", "as at date", "as at", "snapshot date", "extract date"]) if not inventory_raw.empty else None
            default_asof = today
            if report_date_col:
                parsed = parse_dates(inventory_raw[report_date_col]).dropna()
                if not parsed.empty and parsed.max() <= today + pd.Timedelta(days=1):
                    default_asof = parsed.max().normalize()

            with st.sidebar:
                st.header("Purchasing settings")
                asof_date = st.date_input(
                    "As-at date (reports refreshed)", value=default_asof.date(), key="purchasing_asof",
                    help="Date the inventory and PO exports were refreshed. Sales in the current month are pro-rated from this day; "
                         "new orders arrive at this date plus the SKU's lead time.",
                )
                moh_required = float(st.number_input(
                    "MOH required (months of holding)", min_value=0.0, max_value=24.0, value=float(settings["moh_required"]), step=0.5,
                    key="purchasing_moh", help="Target months of average sales to hold when a new order arrives (the workbook used 2).",
                ))
                history_months = int(st.number_input(
                    "Months of sales history", min_value=1, max_value=24, value=int(settings["history_months"]), step=1,
                    key="purchasing_history", help="Complete months before the as-at month averaged for monthly sales (the workbook used 5).",
                ))
                horizon_months = int(st.number_input(
                    "Months to project", min_value=3, max_value=FOS_PLANNING_HORIZON_MAX, value=int(settings["horizon_months"]), step=1,
                    key="purchasing_horizon", help="Rolling calendar length, starting with the as-at month (the workbook used 11).",
                ))
                st.session_state["purchasing_settings"] = {"moh_required": moh_required, "history_months": history_months, "horizon_months": horizon_months}

                st.markdown("**Scope**")
                program_values = column_choices(attributes, "Product group")
                default_program = ["CORE"] if "CORE" in program_values else []
                scope_filters: dict[str, list[str]] = {}
                scope_filters["Product group"] = st.multiselect(
                    "Stock program / product group", program_values, default=default_program, key="purchasing_filter_program",
                    help="Fourth Product Group from the options data. CORE is the stocked range the workbook planned.",
                )
                scope_filters["Supplier"] = st.multiselect("Supplier", column_choices(attributes, "Supplier"), default=[], key="purchasing_filter_supplier")
                scope_filters["Class"] = st.multiselect("Group (class code)", column_choices(attributes, "Class"), default=[], key="purchasing_filter_class")
                scope_filters["SKU type"] = st.multiselect("SKU type", SKU_TYPES, default=[], key="purchasing_filter_skutype")
                purchasing_search = st.text_input("SKU or description contains", key="purchasing_filter_text").strip()
                only_orders = st.toggle("Only SKUs needing an order", value=False, key="purchasing_only_orders")
                include_discontinued = st.toggle("Include discontinued", value=False, key="purchasing_include_discontinued")
                st.caption("Sales-location and warehouse mappings are on the **Backend Mapping** tab.")

            asof = pd.Timestamp(asof_date)
            current_period = asof.to_period("M")
            history_periods = [str(current_period - k) for k in range(history_months, 0, -1)]

            if not mapping["sku"] or not mapping["on_hand"]:
                st.warning("The SKU and on-hand quantity columns of the inventory export could not be identified. Choose them on the **Backend Mapping** tab.")
            else:
                history = fos_sales_history(sales_data, history_periods[0], history_periods[-1], rules_signature(locations_frame))
                stock = fos_stock_by_city(inventory_raw, mapping, site_mapping, warehouse_cities)
                lead_times = option_fields.set_index("SKU")["Lead time (days)"] if not option_fields.empty else pd.Series(dtype=float)
                positive_leads = lead_times[lead_times > 0]
                default_lead = float(positive_leads.median()) if not positive_leads.empty else 60.0
                po_lines = fos_purchase_orders(purchase_orders_raw, lead_times, asof, site_mapping, warehouse_cities, default_lead)

                universe = sorted(
                    set(stock["SKU"]) | set(po_lines.loc[po_lines["Warehouse"], "SKU"]) | set(history["SKU"])
                )
                confirmed = dict(st.session_state["purchasing_confirmed"])
                result = fos_schedule(universe, history, history_periods, stock, po_lines, lead_times, confirmed, asof, moh_required, horizon_months)
                summary = result["summary"]

                # --- enrich with catalogue attributes and costs
                summary = summary.merge(attributes, on="SKU", how="left").merge(option_fields, on="SKU", how="left", suffixes=("", "_opt"))
                inventory_sku = inventory_by_sku(inventory_raw, mapping, site_mapping)
                inventory_cost = inventory_sku.set_index("SKU")["Inventory unit cost"] if "Inventory unit cost" in inventory_sku.columns else pd.Series(dtype=float)
                inventory_description = inventory_sku.set_index("SKU")["Inventory description"] if "Inventory description" in inventory_sku.columns else pd.Series(dtype=object)
                summary["Avg option cost"] = summary["SKU"].map(inventory_cost)
                summary["Avg option cost"] = summary["Avg option cost"].where(summary["Avg option cost"].gt(0), summary["Option unit cost"])
                summary["Description"] = summary["Description"].where(summary["Description"].notna() & summary["Description"].astype(str).str.strip().ne(""), summary["SKU"].map(inventory_description))
                missing_description = summary["Description"].isna() | summary["Description"].astype(str).str.strip().eq("")
                if missing_description.any() and "Description" in sales_data.columns:
                    summary["Description"] = summary["Description"].where(~missing_description, summary["SKU"].map(sku_descriptions(sales_data)))
                summary["Description"] = summary["Description"].fillna("").astype(str).str.replace(r"_x000D_|\r", "", regex=True).str.split("\n").str[0].str.strip()
                summary["SKU type"] = summary["SKU"].map(classify_sku)
                group_class = lookup_frame(purchasing_files.get("Product Groups"), ["product group code", "group code", "code", "class"], ["product group description", "description", "group class", "name"])
                summary["Group class"] = summary["Class"].map(lambda c: group_class.get(sku_text(c), "") if pd.notna(c) else "") if not group_class.empty else ""
                web_category = lookup_frame(purchasing_files.get("WEB Category"), ["options id", "stock code", "sku"], ["category"])
                web_subcategory = lookup_frame(purchasing_files.get("WEB Category"), ["options id", "stock code", "sku"], ["sub category", "subcategory"])
                summary["Category"] = summary["SKU"].map(web_category).fillna("") if not web_category.empty else ""
                summary["Sub category"] = summary["SKU"].map(web_subcategory).fillna("") if not web_subcategory.empty else ""
                summary["Discontinued"] = summary["Discontinued"].fillna(False).astype(bool)

                cost = pd.to_numeric(summary["Avg option cost"], errors="coerce")
                summary["Free SOH value"] = summary["Free SOH"] * cost.fillna(0.0)
                summary["Planned order value"] = summary["Planned order qty"] * cost.fillna(0.0)
                summary["Confirmed order value"] = summary["Confirmed order qty"] * cost.fillna(0.0)
                purchase_cost = pd.to_numeric(summary["Purchase cost"], errors="coerce")
                summary["Total purchase AUD"] = summary["Confirmed order qty"] * purchase_cost.where(purchase_cost.gt(0), cost).fillna(0.0)
                rrp = pd.to_numeric(summary["RRP"], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    summary["ROI"] = np.where(cost.gt(0) & rrp.notna(), (rrp - cost) / cost, np.nan)
                summary["Lead time source"] = np.where(summary["Lead time (days)"].gt(0), "options data", f"default {default_lead:.0f} d")

                # --- filters
                view = summary
                if not include_discontinued:
                    view = view.loc[~view["Discontinued"]]
                for label, chosen in scope_filters.items():
                    if chosen and label in view.columns:
                        view = view.loc[view[label].astype(str).isin(chosen)]
                if purchasing_search:
                    if purchasing_search.endswith("*"):
                        pattern, is_regex = f"^{re.escape(purchasing_search[:-1])}", True
                    else:
                        pattern, is_regex = purchasing_search, False
                    text_match = (
                        view["SKU"].str.contains(pattern, case=False, regex=is_regex, na=False)
                        | view["Description"].str.contains(pattern, case=False, regex=is_regex, na=False)
                    )
                    view = view.loc[text_match]
                if only_orders:
                    view = view.loc[view["Planned order qty"].gt(0)]
                view = view.sort_values(["Planned order value", "Planned order qty", "SKU"], ascending=[False, False, True]).reset_index(drop=True)

                periods = result["periods"]
                period_labels = {p: pd.Period(p).strftime("%b %y") for p in periods}
                scope_index = pd.Index(view["SKU"])

                # --- headline
                window_text = f"{pd.Period(history_periods[0]).strftime('%b %Y')} – {pd.Period(history_periods[-1]).strftime('%b %Y')}"
                st.subheader("Forward order schedule")
                st.caption(
                    f"As at **{asof.strftime('%d %b %Y')}** · sales averaged over {history_months} complete months ({window_text}) · "
                    f"MOH required {moh_required:g} · projecting {horizon_months} months from {pd.Period(periods[0]).strftime('%b %Y')} to {pd.Period(periods[-1]).strftime('%b %Y')} · "
                    f"{len(view):,} of {len(summary):,} SKUs in scope"
                )
                stockouts = int(view["Stock-out month"].ne("").sum())
                metric_card(
                    "Schedule at a glance",
                    [
                        dict(label="SKUs needing an order", value=f"{int(view['Planned order qty'].gt(0).sum()):,}", help="Planned order qty above zero at the SKU's arrival month."),
                        dict(label="Planned order value", value=f"${view['Planned order value'].sum():,.0f}", help="Planned qty × average option cost."),
                        dict(label="Confirmed order value", value=f"${view['Confirmed order value'].sum():,.0f}", help="Confirmed qty × average option cost. Total purchase AUD uses the purchase price."),
                        dict(label="Free SOH value", value=f"${view['Free SOH value'].sum():,.0f}", help="Warehouse free stock (on hand − allocated − damaged/floor/loan) × average option cost."),
                        dict(label="Open PO units (free / linked)", value=f"{view['Open PO free'].sum():,.0f} / {view['Open PO linked'].sum():,.0f}", help="Open purchase orders to warehouses. Linked = tied to a sales order."),
                        dict(label="Stock-outs in horizon", value=f"{stockouts:,}", help="SKUs whose projected national free stock goes negative before confirmed orders."),
                    ],
                )

                # --- schedule table (confirmed order qty is editable)
                st.markdown("<p class='card-title'>Schedule</p>", unsafe_allow_html=True)
                st.caption(
                    "Type a **Confirmed order qty** to see it land in the projection (arrival = as-at date + lead time) and reduce the planned quantity. "
                    "Planned order qty = MOH × average monthly sales − projected free stock in the arrival month, rounded up; blank when stock already covers it."
                )
                decision_columns = [
                    "SKU", "Description", "Supplier", "Lead time (days)", "Avg monthly sales", "Free SOH", "Months of cover now",
                    "Open PO free", "Open PO linked", "Arrival date", "Free SOH at arrival", "Planned order qty", "Confirmed order qty",
                    "Avg option cost", "Planned order value", "Confirmed order value", "Total purchase AUD", "RRP", "ROI", "Stock-out month",
                ]
                detail_columns = [
                    "Retail avg", "Commercial avg", *[f"Sales {pd.Period(p).strftime('%b %y')}" for p in history_periods],
                    "Total SOH", "Allocated SOH", "Damaged/floor/loan", *[f"Free {c}" for c in FOS_CITIES if f"Free {c}" in view.columns],
                    "Supplier qty", "Supplier qty updated", "Class", "Group class", "Product group", "Category", "Sub category", "SKU type", "Lead time source",
                ]
                show_all_columns = st.toggle(
                    "Show all columns (sales by month, stock by city, catalogue attributes)", value=False, key="purchasing_show_all_columns",
                )
                display_columns = decision_columns + (detail_columns if show_all_columns else [])
                display_columns = [c for c in display_columns if c in view.columns]
                editor_frame = view[display_columns].copy()
                editor_frame["Stock-out month"] = editor_frame["Stock-out month"].map(lambda p: pd.Period(p).strftime("%b %Y") if p else "")
                st.session_state["purchasing_editor_frame"] = editor_frame[["SKU", "Confirmed order qty"]].copy()
                editor_key = f"purchasing_editor_{st.session_state['purchasing_editor_version']}"

                def apply_confirmed_changes() -> None:
                    changes = st.session_state.get(editor_key, {}).get("edited_rows", {})
                    base = st.session_state["purchasing_editor_frame"]
                    confirmed_now = dict(st.session_state["purchasing_confirmed"])
                    for row, edits in changes.items():
                        if "Confirmed order qty" in edits:
                            sku = base.iloc[int(row)]["SKU"]
                            value = edits["Confirmed order qty"]
                            if value is None or pd.isna(value) or float(value) <= 0:
                                confirmed_now.pop(sku, None)
                            else:
                                confirmed_now[sku] = float(value)
                    st.session_state["purchasing_confirmed"] = confirmed_now
                    st.session_state["purchasing_editor_version"] += 1

                money = st.column_config.NumberColumn(format="$%,.0f")
                units = st.column_config.NumberColumn(format="%,.0f")
                decimal1 = st.column_config.NumberColumn(format="%.1f")
                column_config = {
                    "SKU": pinned_text_column(),
                    "Lead time (days)": units, "Avg monthly sales": decimal1, "Retail avg": decimal1, "Commercial avg": decimal1,
                    "Total SOH": units, "Allocated SOH": units, "Damaged/floor/loan": units, "Free SOH": units,
                    "Months of cover now": decimal1, "Open PO free": units, "Open PO linked": units,
                    "Arrival date": st.column_config.DateColumn(format="DD MMM YYYY"), "Free SOH at arrival": decimal1,
                    "Planned order qty": st.column_config.NumberColumn(format="%,.0f", help="Rounded up; blank/0 = no order needed."),
                    "Confirmed order qty": st.column_config.NumberColumn(format="%,.0f", min_value=0, step=1, help="Editable — the buyer's confirmed quantity."),
                    "Avg option cost": money, "Planned order value": money, "Confirmed order value": money, "Total purchase AUD": money, "RRP": money,
                    "ROI": st.column_config.NumberColumn(format="%.0f%%", help="(RRP − average option cost) ÷ cost"),
                    "Supplier qty": units, "Supplier qty updated": st.column_config.DateColumn(format="DD MMM YYYY"),
                }
                for p in history_periods:
                    column_config[f"Sales {pd.Period(p).strftime('%b %y')}"] = decimal1
                for c in FOS_CITIES:
                    column_config[f"Free {c}"] = units
                editor_frame["ROI"] = editor_frame["ROI"] * 100 if "ROI" in editor_frame.columns else np.nan
                st.data_editor(
                    editor_frame,
                    key=editor_key,
                    on_change=apply_confirmed_changes,
                    hide_index=True,
                    disabled=[c for c in editor_frame.columns if c != "Confirmed order qty"],
                    column_config=column_config,
                    height=table_height(len(editor_frame), cap=560),
                    **FULL_WIDTH,
                )
                download_a, download_b, download_c, _ = st.columns([1, 1, 1, 3], gap="small")
                download_a.download_button(
                    "Download schedule (CSV)", editor_frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"forward_order_schedule_{asof.strftime('%Y%m%d')}.csv", mime="text/csv", key="purchasing_download_schedule", **FULL_WIDTH,
                )
                confirmed_frame = pd.DataFrame(
                    [{"Stock Code": sku, "Confirmed qty": qty} for sku, qty in sorted(st.session_state["purchasing_confirmed"].items()) if qty > 0]
                )
                download_b.download_button(
                    "Download confirmed orders (CSV)", confirmed_frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name="confirmed_orders.csv", mime="text/csv", key="purchasing_download_confirmed", **FULL_WIDTH,
                    help="Save under Raw_data/purchasing (or into the Confirmed Orders sheet of purchasing_parameters.xlsx) to reload next time.",
                )
                if download_c.button("Clear confirmed quantities", key="purchasing_clear_confirmed", **FULL_WIDTH):
                    st.session_state["purchasing_confirmed"] = {}
                    st.session_state["purchasing_editor_version"] += 1
                    st.rerun()

                # --- monthly projection
                with st.expander("📅 Monthly projection — national free stock", expanded=True):
                    projection_measure = st.radio(
                        "Show", ["Free SOH (units)", "Months of cover", "Free arrivals (units)"], horizontal=True,
                        key="purchasing_projection_measure", label_visibility="collapsed",
                    )
                    source = {"Free SOH (units)": result["national"], "Months of cover": result["cover"], "Free arrivals (units)": result["free_arrivals"]}[projection_measure]
                    projection = source.reindex(scope_index).copy()
                    projection.columns = [period_labels[p] for p in projection.columns]
                    projection.insert(0, "Description", view.set_index("SKU")["Description"].reindex(scope_index).to_numpy())
                    projection.insert(1, "Avg monthly sales", view.set_index("SKU")["Avg monthly sales"].reindex(scope_index).to_numpy())
                    projection = projection.reset_index()
                    number_columns = [period_labels[p] for p in periods]
                    fmt = "{:.1f}" if projection_measure == "Months of cover" else "{:,.0f}"

                    def shade(value):
                        if pd.isna(value):
                            return ""
                        if projection_measure == "Months of cover":
                            if value < 0: return "background-color: #f8d7da; color: #7a1f1f"
                            if value < moh_required: return "background-color: #fff3cd"
                            return ""
                        if value < 0: return "background-color: #f8d7da; color: #7a1f1f"
                        return ""

                    STYLE_LIMIT = 300
                    if len(projection) <= STYLE_LIMIT:
                        styled = projection.style.format({c: fmt for c in number_columns} | {"Avg monthly sales": "{:.1f}"}, na_rep="").map(shade, subset=number_columns)
                        st.dataframe(styled, hide_index=True, height=table_height(len(projection), cap=520), **FULL_WIDTH)
                    else:
                        number_format = "%.1f" if projection_measure == "Months of cover" else "%,.0f"
                        st.dataframe(
                            projection, hide_index=True, height=table_height(len(projection), cap=520),
                            column_config={c: st.column_config.NumberColumn(format=number_format) for c in number_columns} | {"Avg monthly sales": decimal1},
                            **FULL_WIDTH,
                        )
                        st.caption(f"Cell shading is switched off above {STYLE_LIMIT} SKUs — narrow the scope in the sidebar to see stock-out months highlighted.")
                    st.caption(
                        f"Month 1 ({pd.Period(periods[0]).strftime('%b %Y')}) deducts {result['prorata']:.0%} of a month's average sales — the days left from the as-at date. "
                        "Each later month: opening free stock + free PO arrivals + confirmed new order − average monthly sales. "
                        "Red = projected stock-out; amber (months of cover) = below the MOH target."
                    )
                    st.download_button(
                        "Download projection (CSV)", projection.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"fos_projection_{asof.strftime('%Y%m%d')}.csv", mime="text/csv", key="purchasing_download_projection",
                    )

                # --- SKU detail
                with st.expander("🔎 SKU detail — projection by city, sales history and open POs", expanded=False):
                    if view.empty:
                        st.caption("No SKUs in scope.")
                    else:
                        sku_options = view["SKU"].tolist()
                        sku_labels = {row.SKU: f"{row.SKU} · {row.Description}" for row in view.itertuples()}
                        chosen_sku = st.selectbox("SKU", sku_options, format_func=lambda s: sku_labels.get(s, s), key="purchasing_detail_sku")
                        row = view.loc[view["SKU"].eq(chosen_sku)].iloc[0]
                        detail_tiles = [
                            dict(label="Avg monthly sales", value=f"{row['Avg monthly sales']:.1f}", help=f"{window_text}"),
                            dict(label="Free SOH now", value=f"{row['Free SOH']:,.0f}", help=f"Total {row['Total SOH']:,.0f} · allocated {row['Allocated SOH']:,.0f} · D/F/L {row['Damaged/floor/loan']:,.0f}"),
                            dict(label="Lead time", value=f"{row['Lead time (days)']:.0f} d", help=f"Arrives {row['Arrival date'].strftime('%d %b %Y')} ({row['Lead time source']})"),
                            dict(label="Free SOH at arrival", value=f"{row['Free SOH at arrival']:,.1f}", help=f"Projected national free stock at the end of {pd.Period(row['Arrival month']).strftime('%b %Y')}"),
                            dict(label="Planned order qty", value=f"{row['Planned order qty']:,.0f}", help=f"MOH {moh_required:g} × {row['Avg monthly sales']:.1f} − {row['Free SOH at arrival']:,.1f}"),
                            dict(label="Confirmed order qty", value=f"{row['Confirmed order qty']:,.0f}", help=f"Value ${row['Confirmed order value']:,.0f} at average option cost"),
                        ]
                        metric_card(sku_labels[chosen_sku], detail_tiles)

                        chart_rows = []
                        for p in periods:
                            chart_rows.append({"Month": period_labels[p], "Series": "Free SOH (end of month)", "Value": float(result["national"].loc[chosen_sku, p])})
                            chart_rows.append({"Month": period_labels[p], "Series": "Free PO arrivals", "Value": float(result["free_arrivals"].loc[chosen_sku, p])})
                            chart_rows.append({"Month": period_labels[p], "Series": "Linked PO arrivals", "Value": float(result["linked_arrivals"].loc[chosen_sku, p])})
                            chart_rows.append({"Month": period_labels[p], "Series": "Confirmed new order", "Value": float(result["new_orders"].loc[chosen_sku, p])})
                        chart_frame = pd.DataFrame(chart_rows)
                        month_order = [period_labels[p] for p in periods]
                        target_value = float(moh_required * row["Avg monthly sales"])
                        soh_layer = (
                            alt.Chart(chart_frame.loc[chart_frame["Series"].eq("Free SOH (end of month)")])
                            .mark_bar(size=22)
                            .encode(
                                x=alt.X("Month:N", sort=month_order, axis=quiet_axis(labelAngle=0, title=None)),
                                y=alt.Y("Value:Q", title="Units", axis=value_axis(",.0f")),
                                color=alt.condition(alt.datum.Value < 0, alt.value("#9F1D1D"), alt.value("#1C5CAB")),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Value:Q", title="Free SOH", format=",.1f")],
                            )
                        )
                        arrivals_layer = (
                            alt.Chart(chart_frame.loc[chart_frame["Series"].isin(["Free PO arrivals", "Linked PO arrivals", "Confirmed new order"]) & chart_frame["Value"].gt(0)])
                            .mark_point(filled=True, size=90)
                            .encode(
                                x=alt.X("Month:N", sort=month_order),
                                y=alt.Y("Value:Q"),
                                shape=alt.Shape("Series:N", scale=alt.Scale(domain=["Free PO arrivals", "Linked PO arrivals", "Confirmed new order"], range=["circle", "diamond", "triangle-up"]), legend=alt.Legend(title=None, orient="top")),
                                color=alt.Color("Series:N", scale=alt.Scale(domain=["Free PO arrivals", "Linked PO arrivals", "Confirmed new order"], range=["#199E70", "#898781", "#D95926"]), legend=None),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=",.0f")],
                            )
                        )
                        target_layer = alt.Chart(pd.DataFrame({"y": [target_value]})).mark_rule(strokeDash=[6, 4], color="#0B0B0B").encode(y="y:Q")
                        target_text = alt.Chart(pd.DataFrame({"y": [target_value], "label": [f"MOH target {target_value:,.0f}"]})).mark_text(align="left", dx=4, dy=-6, color="#0B0B0B").encode(y="y:Q", text="label:N", x=alt.value(0))
                        zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#C3C2B7").encode(y="y:Q")
                        st.altair_chart(
                            alt.layer(zero_rule, soh_layer, arrivals_layer, target_layer, target_text).properties(height=260).configure_view(strokeWidth=0),
                            **FULL_WIDTH,
                        )
                        st.caption("Bars: projected national free stock at month end (red = stock-out). Markers: PO arrivals (● free, ◆ linked to a sales order) and the confirmed new order (▲). Dashed line: MOH target.")

                        city_table = pd.DataFrame(
                            {period_labels[p]: [float(result["cities"][c].loc[chosen_sku, p]) for c in result["cities"]] for p in periods},
                            index=[f"{c} free SOH" for c in result["cities"]],
                        )
                        city_table.insert(0, "Now", [float(view.set_index("SKU").loc[chosen_sku, f"Free {c}"]) if f"Free {c}" in view.columns else np.nan for c in result["cities"]])
                        city_table.loc["NAT free SOH"] = [float(row["Free SOH"])] + [float(result["national"].loc[chosen_sku, p]) for p in periods]
                        city_table.loc["Months of cover"] = [float(row["Months of cover now"]) if pd.notna(row["Months of cover now"]) else np.nan] + [float(result["cover"].loc[chosen_sku, p]) for p in periods]
                        st.markdown("**Closing free stock by city**")
                        st.dataframe(
                            city_table.style.format("{:,.1f}", na_rep="").map(lambda v: "background-color: #f8d7da; color: #7a1f1f" if pd.notna(v) and v < 0 else ""),
                            height=table_height(len(city_table)), **FULL_WIDTH,
                        )
                        st.caption("City rows deduct that city's own average sales and add its free PO arrivals; the confirmed new order is only known nationally, so it appears in the NAT row.")

                        st.markdown("**Sales history (units)**")
                        st.dataframe(
                            fos_history_table(history, chosen_sku, history_periods).style.format("{:,.1f}"),
                            height=table_height(len(history_periods) + 1), **FULL_WIDTH,
                        )
                        sku_pos = po_lines.loc[po_lines["SKU"].eq(chosen_sku)].copy()
                        st.markdown("**Open purchase orders**")
                        if sku_pos.empty:
                            st.caption("No open purchase orders for this SKU.")
                        else:
                            sku_pos["Counted"] = np.where(sku_pos["Warehouse"], "Yes", "No — not a Warehouse site")
                            sku_pos["ETA basis"] = np.where(sku_pos["Estimated"], "Estimated (order date + lead time)", "From export")
                            show = sku_pos[["PO", "Supplier", "Site", "City", "Open qty", "Linked", "Sales order", "Order date", "ETA", "ETA basis", "Overdue", "Counted"]]
                            st.dataframe(
                                show, hide_index=True, height=table_height(len(show)),
                                column_config={"Order date": st.column_config.DateColumn(format="DD MMM YYYY"), "ETA": st.column_config.DateColumn(format="DD MMM YYYY"), "Open qty": units},
                                **FULL_WIDTH,
                            )

                # --- data checks
                with st.expander("🧪 Data checks and method", expanded=False):
                    rules = location_rules(locations_frame)
                    seen_locations = (
                        sales_data.dropna(subset=["Date"]).loc[lambda d: d["Date"].dt.to_period("M").astype(str).isin(history_periods), ["Location", "City"]]
                        .astype(str).drop_duplicates()
                        if "Location" in sales_data.columns else pd.DataFrame(columns=["Location", "City"])
                    )
                    unmapped_locations = sorted({loc for loc, hint in zip(seen_locations["Location"], seen_locations["City"]) if normalise_text(loc) not in rules})
                    po_sites = po_lines.loc[~po_lines["Warehouse"], "Site"].value_counts() if not po_lines.empty else pd.Series(dtype=int)
                    no_lead = view.loc[view["Lead time (days)"].le(0) | view["Lead time (days)"].isna(), "SKU"].tolist()
                    checks = [
                        f"Sales locations in the window without a rule (allocated by their city name, or by the observed city mix when none): {', '.join(unmapped_locations) if unmapped_locations else 'none'}.",
                        f"Open PO lines to non-Warehouse sites (not counted as incoming stock): {int(po_sites.sum()) if not po_sites.empty else 0}"
                        + (f" — {', '.join(f'{s} ({n})' for s, n in po_sites.head(8).items())}" if not po_sites.empty else "") + ".",
                        f"PO lines with an estimated ETA (blank in the export): {int(po_lines['Estimated'].sum()) if not po_lines.empty else 0}; overdue ETAs counted in the as-at month: {int(po_lines['Overdue'].sum()) if not po_lines.empty else 0}.",
                        f"SKUs in scope without a lead time in the options data (default {default_lead:.0f} days used): {len(no_lead)}" + (f" — {', '.join(no_lead[:12])}{'…' if len(no_lead) > 12 else ''}" if no_lead else "") + ".",
                        f"Warehouse cities: {', '.join(f'{k} → {v}' for k, v in sorted(warehouse_cities.items()))}; other Warehouse-category codes are placed by their suffix (…SY → SYD, …ME → MEL, …PE → PER, …BR → BNE, …AD → ADL).",
                        "Stock counted: sites in the Warehouse category only (as the workbook: W1SY, W5BR, W3ME, W6AD, W4PE). Free = on hand − allocated − damaged/floor/loan (Qty Alloc, Qty D, Qty F, Qty L in the export).",
                        "Open PO quantity = ordered − received. A PO whose ETA has passed is counted in the as-at month rather than dropped.",
                    ]
                    for check in checks:
                        st.markdown(f"- {check}")
                    if purchasing_files:
                        st.caption("Parameter files loaded from Raw_data/purchasing: " + ", ".join(sorted(purchasing_files)) + ".")
                    else:
                        st.caption("No parameter workbook under Raw_data/purchasing — built-in defaults from the workbook's Parameters tab are in use (editable on the Backend Mapping tab).")
        except Exception as e:
            st.error(f"Error building the purchasing schedule: {e}")
            st.exception(e)

# =============================================================================
# Backend Mapping tab
# =============================================================================
elif selected_tab == "Backend Mapping":
    init_backend_state()
    prepared_data, source_names = load_prepared_data()
    inventory_raw = prepared_data["Current Inventory"] if prepared_data is not None else pd.DataFrame()
    options_raw = prepared_data.get("OptionsData") if prepared_data is not None else None

    st.subheader("Backend mapping")
    st.caption(
        "Configuration the inventory analysis runs on. Changes apply immediately for this session; "
        "save them to make them the default for this deployment."
    )

    detected_mapping = resolve_inventory_mapping(inventory_raw) if not inventory_raw.empty else {}
    site_units = units_by_site(inventory_raw, detected_mapping) if detected_mapping else pd.Series(dtype=float)
    site_mapping = current_site_mapping()

    with st.container(border=True):
        st.markdown("<p class='card-title'>Site categories</p>", unsafe_allow_html=True)
        st.caption(
            "**Warehouse** and **Clearance** stock is available to sell and counts towards stock health. "
            "**Showroom** stock is display stock and is excluded. **Loan** stock is out with clients: not for sale, but "
            "valued on the Inventory tab (codes containing LOAN land here automatically). **Ignored** sites (quarantine, "
            "returns, in transit, anything not worth tracking) are skipped silently. A code still **Unmapped** is skipped "
            "too, but flagged until it gets a category. Add a row for a new site code; codes are matched case-insensitively."
        )
        editor_frame = site_mapping_frame(site_mapping, site_units)
        st.session_state["site_editor_frame"] = editor_frame
        editor_key = f"backend_site_editor_{st.session_state['site_editor_version']}"
        st.data_editor(
            editor_frame,
            key=editor_key,
            on_change=apply_site_editor_changes,
            num_rows="dynamic",
            hide_index=True,
            disabled=["Available to sell", "In inventory export", "Units on hand"],
            column_config={
                "Site code": st.column_config.TextColumn(required=True, max_chars=12, validate=r"^\s*[A-Za-z0-9_-]+\s*$"),
                "Category": st.column_config.SelectboxColumn(
                    options=SITE_CATEGORIES, required=True,
                    help="Showroom, Warehouse, Clearance, Loan or Ignored. Unmapped means not yet decided.",
                ),
                "Available to sell": st.column_config.CheckboxColumn(help="Derived from the category."),
                "In inventory export": st.column_config.CheckboxColumn(help="The code appears in the current inventory export."),
                "Units on hand": st.column_config.NumberColumn(format="%d", help="All SKUs at this site in the current export."),
            },
            height=table_height(len(editor_frame), cap=520),
            **FULL_WIDTH,
        )

        unmapped = sorted(code for code in site_units.index if code and site_mapping.get(code, "Unmapped") == "Unmapped")
        categorised_units = {
            category: float(sum(units for code, units in site_units.items() if site_mapping.get(code) == category))
            for category in ASSIGNABLE_CATEGORIES
        }
        summary_bits = [f"{category}: {units:,.0f} units" for category, units in categorised_units.items()]
        st.caption(
            " · ".join(summary_bits)
            + (f" · **Unmapped (needs a decision): {', '.join(unmapped)}**" if unmapped else " · every site code in the export has a category")
        )

        action_reset, action_save, action_download, _ = st.columns([1, 1, 1, 3], gap="small")
        if action_reset.button("Reset to defaults", key="backend_reset", **FULL_WIDTH):
            st.session_state["site_mapping"] = dict(DEFAULT_SITE_MAPPING)
            st.session_state["site_editor_version"] += 1
            st.rerun()
        if action_save.button("Save as deployment default", key="backend_save", **FULL_WIDTH, help=f"Writes {MAPPING_FILE.name} next to app.py."):
            try:
                save_backend_mapping_file(
                    {
                        "site_mapping": st.session_state["site_mapping"],
                        "column_mapping": st.session_state["inventory_column_mapping"],
                        "loan_column_mapping": st.session_state["loan_column_mapping"],
                        "cost_source": st.session_state["cost_source"],
                        "purchasing": purchasing_payload(),
                    }
                )
                st.success(f"Saved to {MAPPING_FILE.name}. Commit it to the repository to keep it across redeploys.")
            except OSError as e:
                st.error(f"Could not write {MAPPING_FILE.name}: {e}")
        action_download.download_button(
            "Download JSON",
            json.dumps(
                {
                    "site_mapping": st.session_state["site_mapping"],
                    "column_mapping": st.session_state["inventory_column_mapping"],
                    "loan_column_mapping": st.session_state["loan_column_mapping"],
                    "cost_source": st.session_state["cost_source"],
                    "purchasing": purchasing_payload(),
                },
                indent=2,
            ).encode("utf-8"),
            file_name="backend_mapping.json",
            mime="application/json",
            key="backend_download",
            **FULL_WIDTH,
        )

    with st.container(border=True):
        st.markdown("<p class='card-title'>Purchasing parameters (forward order schedule)</p>", unsafe_allow_html=True)
        st.caption(
            "How the **Purchasing** tab reads the sales export and the stock sites. Each sales location is counted as Retail or "
            "Commercial and in one city — or **Split** across cities by the shares (0–1, adding to 1). Warehouse sites need a city so "
            "stock and PO arrivals land in the right column. Defaults come from the Parameters tab of Forward_Order_Schedule.xlsm; "
            "a workbook under Raw_data/purchasing (purchasing_parameters.xlsx) seeds them instead, and edits here apply to this session."
        )
        purchasing_state()
        if "purchasing_params_version" not in st.session_state:
            st.session_state["purchasing_params_version"] = 0
        locations_editor_frame = sales_locations_frame(st.session_state["purchasing_locations"])
        st.session_state["purchasing_locations_frame"] = locations_editor_frame
        locations_key = f"backend_purchasing_locations_{st.session_state['purchasing_params_version']}"

        def apply_locations_changes() -> None:
            state = st.session_state.get(locations_key, {})
            frame = st.session_state["purchasing_locations_frame"].copy()
            for row, changes in state.get("edited_rows", {}).items():
                for column, value in changes.items():
                    frame.loc[int(row), column] = value
            if state.get("added_rows"):
                frame = pd.concat([frame, pd.DataFrame(state["added_rows"])], ignore_index=True)
            if state.get("deleted_rows"):
                frame = frame.drop(index=[int(row) for row in state["deleted_rows"]])
            frame = sales_locations_frame(frame.to_dict("records"))
            frame = frame.loc[frame["Location"].ne("") | frame["Code"].ne("")]
            st.session_state["purchasing_locations"] = frame.to_dict("records")
            st.session_state["purchasing_params_version"] += 1

        st.markdown("**Sales locations**")
        st.data_editor(
            locations_editor_frame,
            key=locations_key,
            on_change=apply_locations_changes,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Code": st.column_config.TextColumn(max_chars=12, help="Location code (optional; matched as well as the name)."),
                "Location": st.column_config.TextColumn(required=True, help="Exactly as it appears after LOCATION: in the sales export."),
                "Channel": st.column_config.SelectboxColumn(options=FOS_CHANNELS, required=True),
                "City": st.column_config.SelectboxColumn(options=FOS_CITIES + ["Split"], required=True, help="Split uses the share columns."),
                **{city: st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=1.0, step=0.05, help=f"Share of a Split location's sales going to {FOS_CITY_NAMES[city]}.") for city in FOS_CITIES},
            },
            height=table_height(len(locations_editor_frame), cap=420),
            **FULL_WIDTH,
        )
        split_rows = locations_editor_frame.loc[locations_editor_frame["City"].str.upper().eq("SPLIT")]
        bad_splits = [row.Location for row in split_rows.itertuples() if abs(sum(getattr(row, c) for c in FOS_CITIES) - 1.0) > 0.01]
        if bad_splits:
            st.warning("Split shares do not add to 1 for: " + ", ".join(bad_splits) + ". They are normalised when used.")

        warehouses_editor_frame = pd.DataFrame(
            [{"Site": code, "City": city} for code, city in sorted(st.session_state["purchasing_warehouses"].items())],
            columns=["Site", "City"],
        )
        st.session_state["purchasing_warehouses_frame"] = warehouses_editor_frame
        warehouses_key = f"backend_purchasing_warehouses_{st.session_state['purchasing_params_version']}"

        def apply_warehouse_changes() -> None:
            state = st.session_state.get(warehouses_key, {})
            frame = st.session_state["purchasing_warehouses_frame"].copy()
            for row, changes in state.get("edited_rows", {}).items():
                for column, value in changes.items():
                    frame.loc[int(row), column] = value
            if state.get("added_rows"):
                frame = pd.concat([frame, pd.DataFrame(state["added_rows"])], ignore_index=True)
            if state.get("deleted_rows"):
                frame = frame.drop(index=[int(row) for row in state["deleted_rows"]])
            mapping_out = {}
            for site, city in zip(frame.get("Site", []), frame.get("City", [])):
                code = normalise_site_code(site)
                if code and str(city).upper() in FOS_CITIES:
                    mapping_out[code] = str(city).upper()
            st.session_state["purchasing_warehouses"] = mapping_out
            st.session_state["purchasing_params_version"] += 1

        st.markdown("**Warehouse cities**")
        st.data_editor(
            warehouses_editor_frame,
            key=warehouses_key,
            on_change=apply_warehouse_changes,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Site": st.column_config.TextColumn(required=True, max_chars=12, validate=r"^\s*[A-Za-z0-9_-]+\s*$"),
                "City": st.column_config.SelectboxColumn(options=FOS_CITIES, required=True),
            },
            height=table_height(len(warehouses_editor_frame), cap=260),
            **FULL_WIDTH,
        )
        st.caption(
            "Only sites in the **Warehouse** category above count as stock and as PO destinations in the schedule. "
            "A Warehouse site without a row here is placed by its code suffix (…SY → SYD, …ME → MEL, …PE → PER, …BR → BNE, …AD → ADL)."
        )
        if st.button("Reset purchasing parameters to workbook defaults", key="backend_purchasing_reset"):
            st.session_state["purchasing_locations"] = [dict(r) for r in DEFAULT_SALES_LOCATIONS]
            st.session_state["purchasing_warehouses"] = dict(DEFAULT_WAREHOUSE_CITIES)
            st.session_state["purchasing_settings"] = dict(DEFAULT_PURCHASING_SETTINGS)
            st.session_state["purchasing_params_version"] += 1
            st.rerun()

    with st.container(border=True):
        st.markdown("<p class='card-title'>SKU type rules</p>", unsafe_allow_html=True)
        st.caption("Products are classified from the stock-code pattern, first matching rule wins. Available as a filter and a grouping on the Inventory Analysis tab.")
        rules = sku_type_rules_table()
        if not inventory_raw.empty and detected_mapping.get("sku"):
            codes = inventory_raw[detected_mapping["sku"]].map(sku_text)
            counts = codes.loc[codes.ne("")].drop_duplicates().map(classify_sku).value_counts()
            rules["SKUs in export"] = rules["SKU type"].map(counts).fillna(0).astype(int)
        st.dataframe(rules, hide_index=True, height=table_height(len(rules)), **FULL_WIDTH)

    with st.container(border=True):
        st.markdown("<p class='card-title'>Inventory export columns</p>", unsafe_allow_html=True)
        if inventory_raw.empty:
            st.caption("No inventory export is loaded, so there is nothing to map yet.")
        else:
            st.caption(
                "Detected automatically from the column headings; override any that are wrong. "
                "SKU, on-hand quantity and site are what the stock-health analysis depends on."
            )
            detected = detect_inventory_columns(inventory_raw)
            column_options = ["(auto)", "(none)"] + [str(col) for col in inventory_raw.columns]
            overrides = dict(st.session_state["inventory_column_mapping"])
            chooser_columns = st.columns(3, gap="small")
            for index, (role, label) in enumerate(INVENTORY_ROLE_LABELS.items()):
                current = overrides.get(role, "(auto)")
                if current not in column_options:
                    current = "(auto)"
                auto_label = f"auto → {detected[role]}" if detected[role] else "auto → not found"
                choice = chooser_columns[index % 3].selectbox(
                    label,
                    column_options,
                    index=column_options.index(current),
                    key=f"backend_column_{role}",
                    help=auto_label,
                )
                if choice == "(auto)":
                    overrides.pop(role, None)
                else:
                    overrides[role] = choice
            st.session_state["inventory_column_mapping"] = overrides
            with st.expander("Preview of the inventory export", expanded=False):
                st.dataframe(inventory_raw.head(8), hide_index=True, **FULL_WIDTH)

    loans_raw = prepared_data.get("Loans", pd.DataFrame()) if prepared_data is not None else pd.DataFrame()
    with st.container(border=True):
        st.markdown("<p class='card-title'>Loan file columns</p>", unsafe_allow_html=True)
        if loans_raw.empty:
            st.caption("No loan file is loaded. Put the loan / transfer export under Raw_data/loan (newest file is used).")
        else:
            st.caption(
                "A line with a customer name is a loan; one without is an inter-site transfer, unless it left a Loan site. "
                "SKU, quantity and customer are what the loan analysis depends on."
            )
            detected_loan = detect_loan_columns(loans_raw)
            loan_options = ["(auto)", "(none)"] + [str(col) for col in loans_raw.columns]
            loan_overrides = dict(st.session_state["loan_column_mapping"])
            chooser = st.columns(3, gap="small")
            for index, (role, label) in enumerate(LOAN_ROLE_LABELS.items()):
                current = loan_overrides.get(role, "(auto)")
                if current not in loan_options:
                    current = "(auto)"
                choice = chooser[index % 3].selectbox(
                    label, loan_options, index=loan_options.index(current), key=f"backend_loan_column_{role}",
                    help=f"auto → {detected_loan[role]}" if detected_loan[role] else "auto → not found",
                )
                if choice == "(auto)":
                    loan_overrides.pop(role, None)
                else:
                    loan_overrides[role] = choice
            st.session_state["loan_column_mapping"] = loan_overrides
            with st.expander("Preview of the loan file", expanded=False):
                st.dataframe(loans_raw.head(8), hide_index=True, **FULL_WIDTH)

    with st.container(border=True):
        st.markdown("<p class='card-title'>Average option cost</p>", unsafe_allow_html=True)
        st.caption(
            "The per-unit cost of one unit of a SKU/option. Total stock value = units on hand × this cost; "
            "free stock value = free units × this cost. Gaps in the chosen source are filled from the others."
        )
        inventory_cost_col = resolve_inventory_mapping(inventory_raw).get("cost") if not inventory_raw.empty else None
        options_cost_col = (
            detect_column(options_raw.columns, INVENTORY_COLUMN_CANDIDATES["cost"],
                          exclude=(detect_column(options_raw.columns, ["stock code", "sku", "sku key"]) or "",))
            if options_raw is not None and not options_raw.empty else None
        )
        availability = {
            "inventory": f"inventory export column '{inventory_cost_col}'" if inventory_cost_col else "no cost column found in the inventory export",
            "options": f"options data column '{options_cost_col}'" if options_cost_col else "no cost column found in the options data",
            "sales": "COGS ÷ units sold in the window, for SKUs with sales",
        }
        source_keys = list(COST_SOURCES)
        chosen = st.radio(
            "Cost source",
            source_keys,
            index=source_keys.index(st.session_state["cost_source"]),
            format_func=lambda key: COST_SOURCES[key],
            key="backend_cost_source_choice",
            label_visibility="collapsed",
        )
        st.session_state["cost_source"] = chosen
        st.caption(" · ".join(f"{COST_SOURCES[key].split(' (')[0]}: {text}" for key, text in availability.items()))

# =============================================================================
# Options Data tab
# =============================================================================
elif selected_tab == "Options Data":
    try:
        df_options = load_csv(str(csv_path))
        df_options_filtered = filter_options_data(df_options)

        head_left, head_right = st.columns([5, 2], gap="small", vertical_alignment="center")
        with head_left:
            st.subheader("Options data")
            st.caption(
                f"Showing {min(1000, len(df_options_filtered)):,} of {len(df_options_filtered):,} filtered rows "
                f"({len(df_options):,} total options)"
            )
        with head_right:
            st.download_button(
                "Download full options data", df_options.to_csv(index=False).encode("utf-8"),
                file_name="options_data.csv", mime="text/csv", **FULL_WIDTH,
            )

        with st.container(border=True):
            st.data_editor(df_options_filtered.head(1000), num_rows="dynamic", **FULL_WIDTH)

    except FileNotFoundError:
        st.warning("⚠️ 'options_data.csv' not found under Raw_data.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# =============================================================================
# Formatted Data tab
# =============================================================================
elif selected_tab == "Formatted Data":
    prepared_data, source_names = load_prepared_data()

    if prepared_data is None:
        st.info("Add required source files to view formatted tables.")
    else:
        sheet_names = list(prepared_data.keys())
        head_left, head_right = st.columns([5, 2], gap="small", vertical_alignment="bottom")
        with head_left:
            selected_sheet = st.selectbox("Select table", sheet_names)
        df_sheet = prepared_data[selected_sheet]
        with head_right:
            st.download_button(
                f"Download {selected_sheet} CSV", df_sheet.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_sheet.lower().replace(' ', '_')}.csv", mime="text/csv", **FULL_WIDTH,
            )

        st.caption(f"Showing {min(1000, len(df_sheet)):,} of {len(df_sheet):,} rows")
        with st.container(border=True):
            st.data_editor(df_sheet.head(1000), num_rows="dynamic", **FULL_WIDTH)

# =============================================================================
# Document Status tab
# =============================================================================
elif selected_tab == "Document Status":
    st.subheader("📄 Document Last Modified Tracker")
    st.caption("Every data file under Raw_data plus the application code, with real modification times.")

    def document_record(path: Path, category: str) -> dict:
        stat = path.stat()
        modified = pd.Timestamp(stat.st_mtime, unit="s")
        age_days = (pd.Timestamp.now() - modified).days
        status = "Updated today" if age_days < 1 else ("Updated this week" if age_days < 7 else "Older than 7 days")
        return {
            "Document Name": path.name,
            "Category": category,
            "Last Modified": modified,
            "Size (KB)": round(stat.st_size / 1024, 1),
            "Status": status,
        }

    doc_records = []
    if raw_data_path.exists():
        for path in raw_data_path.rglob("*"):
            if not path.is_file() or path.name in ENGINE_PLACEHOLDERS:
                continue
            if "inventory" in path.parts:
                category = "Inventory Folder"
            elif "PODash" in path.parts:
                category = "PO Dash Folder"
            elif "loan" in path.parts:
                category = "Loan Folder"
            elif "targets" in path.parts:
                category = "Targets Folder"
            elif "purchasing" in path.parts:
                category = "Purchasing Folder"
            elif "Sales Data" in path.parts:
                category = "Sales Folder"
            elif path.name == "options_data.csv":
                category = "Catalog"
            else:
                category = "Raw Data / Source"
            doc_records.append(document_record(path, category))
    for path in sorted(base_dir.glob("*.py")):
        doc_records.append(document_record(path, "System Code"))

    df_docs = pd.DataFrame(doc_records, columns=["Document Name", "Category", "Last Modified", "Size (KB)", "Status"])

    f_col1, f_col2 = st.columns([3, 2], gap="small")
    with f_col1:
        doc_search = st.text_input("Search document name", "").strip()
    with f_col2:
        categories = ["ALL"] + sorted(df_docs["Category"].unique().tolist())
        selected_cat = st.selectbox("Filter by Category", categories)

    filtered_docs = df_docs.copy()
    if doc_search:
        filtered_docs = filtered_docs[filtered_docs["Document Name"].str.contains(doc_search, case=False, na=False)]
    if selected_cat != "ALL":
        filtered_docs = filtered_docs[filtered_docs["Category"] == selected_cat]
    filtered_docs = filtered_docs.sort_values("Last Modified", ascending=False)

    stat_col1, stat_col2, stat_col3 = st.columns(3, gap="small")
    stat_col1.metric("Total Tracked Documents", len(df_docs))
    stat_col2.metric("Filtered Results", len(filtered_docs))
    recent_count = int((df_docs["Last Modified"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum())
    stat_col3.metric("Modified in Last 7 Days", recent_count)

    st.dataframe(
        filtered_docs,
        hide_index=True,
        column_config={
            "Last Modified": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            "Size (KB)": st.column_config.NumberColumn(format="%.1f KB"),
        },
    height=table_height(len(filtered_docs), cap=450),
    **FULL_WIDTH,
)
