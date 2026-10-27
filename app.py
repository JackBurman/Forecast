from pathlib import Path
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
        .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
        [data-testid="stVerticalBlock"] { gap: 0.6rem; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
        h1 { font-size: 1.8rem !important; padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; }
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
    "description": ["description", "item description", "product description", "product name", "name", "product"],
}
INVENTORY_ROLE_LABELS = {
    "sku": "SKU column",
    "on_hand": "On-hand quantity column",
    "free": "Free stock column",
    "site": "Site / location column",
    "value": "Stock value column",
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


def sku_text(value) -> str:
    normalised = normalize_sku(value)
    return "" if pd.isna(normalised) else str(normalised)


@st.cache_data
def clean_sales_data(df_sales: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
    """Apply the Power Query-style transformation to the merged sales exports."""
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
        "City", "Sales Channel", "Sales Channel Net", "Location", "Inv #", "Date", "Month", "Year",
        "SKU", "OptionsData.Supplier", "OptionsData.Class", "Description",
        "OptionsData.Fourth Product Group", "Qty", "Sale", "Cost", "GP%", "GP $",
    ]
    final_columns = [col for col in final_columns if col in sales.columns]
    sales = sales[final_columns]
    if "SKU" in sales.columns:
        sales["SKU"] = sales["SKU"].astype("string")
    sales = sales.sort_values("Date", ascending=False).reset_index(drop=True)
    return sales


def raw_folder_manifest(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Names, sizes and mtimes of every data file; a change invalidates the cache."""
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


# -----------------------------------------------------------------------------
# Workbook reading by content, not by extension
#
# ERP exports are often not what their extension says: a ".xlsx" that is a
# legacy BIFF ".xls", an "Excel" file that is really an HTML table, a CSV with
# a .xls name, or - on Streamlit Community Cloud - a Git LFS pointer instead of
# the workbook. Reading by extension (or forcing one engine) fails on these
# with messages like "File is not a zip file". The first bytes of the file say
# what it really is, so the reader below sniffs them and picks the right
# parser, and the loader checks every workbook up front so a bad file is named
# rather than surfacing as a cryptic error from inside forecast_engine.
# -----------------------------------------------------------------------------

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
    """What a spreadsheet file really contains: xlsx, xls, html, csv, lfs-pointer, empty or unknown."""
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
    """python-calamine is installed and this pandas knows the 'calamine' engine (2.2+)."""
    try:
        import python_calamine  # noqa: F401
    except ImportError:
        return False
    major, minor = (int(part) for part in pd.__version__.split(".")[:2])
    return (major, minor) >= (2, 2)


def workbook_problem(path: Path) -> str | None:
    """A sentence describing why the file cannot be read, or None when it can."""
    kind = sniff_workbook(path)
    if kind == "xls" and not xlrd_available():
        kind = "xls-no-xlrd"
    return WORKBOOK_PROBLEMS.get(kind)


_pandas_read_excel = pd.read_excel


def read_workbook(path: Path, *args, **kwargs) -> pd.DataFrame:
    """pd.read_excel that chooses the parser from the file's content."""
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
    """Route file-path reads through read_workbook; anything else goes to pandas untouched.

    forecast_engine reads the Raw_data workbooks with pd.read_excel, so this is
    the one place its reads can be made content-aware without editing it. An
    explicit engine= argument is respected, except that engine="calamine" (which
    forecast_engine asks for) falls back to content-based reading when the
    python-calamine package is not installed, instead of failing the load.
    """
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
    """Read a sheet; if it is shorter than the requested header offset, take row 0 as the header.

    forecast_engine reads its workbooks with header=1 (a title row above the
    headings). A zero-row placeholder, or an export without the title row, has
    nothing on row 2 and pandas raises "Passed header=[1] ... but only 1 lines
    in file"; the first row is then the header.
    """
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
    """Every .xls/.xlsx under the folder that cannot be read, with the reason."""
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
    """Row index of the header in an export that may start with title lines.

    The first row holding a SKU-like header wins; otherwise the row with the
    most populated cells.
    """
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
    """Read a CSV / Excel export, skipping any title rows above the header."""
    if path.suffix.lower() == ".csv":
        # Title lines above the header have fewer fields than the data, which
        # trips the default parser; read the preview as ragged rows instead.
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
            # Sheet shorter than the detected offset (e.g. a one-line export).
            frame = read_workbook(path, header=0)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame.dropna(how="all").reset_index(drop=True)


def add_alias_columns(frame: pd.DataFrame, aliases: dict[str, list[str]], numeric: tuple[str, ...] = ()) -> pd.DataFrame:
    """Add standard column names alongside the export's own headers.

    Original columns are never renamed, so whatever forecast_engine keyed on
    in the original workbooks is still present; the aliases cover the names it
    is known to look for.
    """
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
    """Create zero-row workbooks the engine's reader expects; returns what was created.

    A zero-byte file of the same name (left behind by an earlier version of the
    loader, or created with `touch`) is not a workbook and is replaced too.
    """
    created = []
    for name, columns in ENGINE_PLACEHOLDERS.items():
        path = raw_data_path / name
        if not path.exists() or path.stat().st_size == 0:
            # Same shape as the real exports: a title row, then the headings
            # (forecast_engine reads these with header=1).
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
    """App-side stand-in for forecast_engine.prepare_forecast_sheets.

    Adds the three derived sales columns the filters rely on (SKU Key, Supplier
    Brand, Model) from the options data, and passes the other sheets through.
    """
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
    """Returns (sheets, source file names, notes to show the user)."""
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

    # Live inventory / PO exports: newest file in each sub-folder, falling back
    # to whatever the engine read from Raw_data itself.
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
    return sheets, source_names, notes


def load_prepared_data():
    """Load Raw_data for a tab; returns (sheets, source_names) or (None, [])."""
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
#
# The engine's helpers are used whenever they work. If one raises on the new
# inventory / PO exports, the app computes the same quantity itself and says
# so, rather than blanking the whole tab.
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
    """Call forecast_engine.<name>(*args); on failure use fallback and return a note."""
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


def column_or_empty(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="string")


# =============================================================================
# Forecast math (unchanged)
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
# Year view (calendar Jan–Dec or Australian financial year Jul–Jun)
# =============================================================================

def fiscal_year_of(period: pd.Period) -> int:
    """Australian FY ends 30 June and is named by its end year (FY2027 = Jul 2026 – Jun 2027)."""
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
    """Re-group a Month × calendar-year profile into Month × financial-year; values are moved, not recomputed."""
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

STATUS_ORDER = ["Dead stock", "Slow moving", "Stock-out risk", "Healthy", "Out of stock", "Inactive"]


@st.cache_data(show_spinner=False)
def inventory_by_sku(inventory: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """One row per SKU with on-hand and free stock summed across sites."""
    frame = pd.DataFrame({"SKU": inventory[mapping["sku"]].map(sku_text)})
    frame["On hand"] = to_number(inventory[mapping["on_hand"]]).fillna(0.0)
    frame["Free stock"] = (
        to_number(inventory[mapping["free"]]).fillna(0.0) if mapping.get("free") else frame["On hand"]
    )
    if mapping.get("value"):
        frame["Stock value"] = to_number(inventory[mapping["value"]]).fillna(0.0)
    if mapping.get("site"):
        frame["Site"] = inventory[mapping["site"]].astype(str).str.strip()
    if mapping.get("description"):
        frame["Inventory description"] = inventory[mapping["description"]].astype(str).str.strip()
    frame = frame.loc[frame["SKU"].ne("")]

    aggregations = {"On hand": "sum", "Free stock": "sum"}
    if "Stock value" in frame.columns:
        aggregations["Stock value"] = "sum"
    if "Inventory description" in frame.columns:
        aggregations["Inventory description"] = "first"
    grouped = frame.groupby("SKU").agg(aggregations)
    if "Site" in frame.columns:
        grouped["Sites with stock"] = (
            frame.loc[frame["On hand"] > 0].groupby("SKU")["Site"].nunique().reindex(grouped.index).fillna(0).astype(int)
        )
    return grouped.reset_index()


@st.cache_data(show_spinner=False)
def sku_movement(sales: pd.DataFrame, window_start: str, window_end: str) -> pd.DataFrame:
    """Per-SKU sales inside the window plus lifetime first/last sale dates."""
    dated = sales.dropna(subset=["Date"]).copy()
    key_column = "SKU Key" if "SKU Key" in dated.columns else "SKU"
    dated["SKU"] = dated[key_column].map(sku_text)
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


def option_attributes(options: pd.DataFrame | None) -> pd.DataFrame:
    """Per-SKU Description / Supplier / Class / Product group from the options sheet."""
    empty = pd.DataFrame(columns=["SKU", "Description", "Supplier", "Class", "Product group"])
    if options is None or options.empty:
        return empty
    sku_col = detect_column(options.columns, ["stock code", "sku", "sku key"])
    if sku_col is None:
        return empty
    frame = pd.DataFrame({"SKU": options[sku_col].map(sku_text)})
    for role, candidates in {
        "Description": INVENTORY_COLUMN_CANDIDATES["description"],
        "Supplier": ["supplier", "supplier brand", "brand", "vendor"],
        "Class": ["class"],
        "Product group": ["fourth product group", "product group", "category"],
    }.items():
        col = detect_column(options.columns, candidates, exclude=(sku_col,))
        frame[role] = options[col].astype(str).str.strip() if col else pd.NA
    return frame.loc[frame["SKU"].ne("")].drop_duplicates(subset=["SKU"])


def open_po_by_sku(purchase_orders: pd.DataFrame | None) -> pd.Series:
    """Open PO quantity per SKU; empty when the PO sheet has no SKU or quantity column."""
    if purchase_orders is None or purchase_orders.empty:
        return pd.Series(dtype=float)
    sku_col = detect_column(purchase_orders.columns, SKU_HEADERS)
    qty_col = detect_column(purchase_orders.columns, OPEN_PO_HEADERS)
    if sku_col is None or qty_col is None:
        return pd.Series(dtype=float)
    keyed = pd.DataFrame(
        {"SKU": purchase_orders[sku_col].map(sku_text), "Open PO qty": to_number(purchase_orders[qty_col]).fillna(0.0)}
    )
    return keyed.loc[keyed["SKU"].ne("")].groupby("SKU")["Open PO qty"].sum()


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
) -> pd.DataFrame:
    """SKU-level stock position, movement, days of supply, turns and status."""
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

    for column in ["On hand", "Free stock", "Units sold", "Sales value", "COGS", "Months with sales"]:
        master[column] = pd.to_numeric(master[column], errors="coerce").fillna(0.0) if column in master.columns else 0.0
    if not open_po.empty:
        master["Open PO qty"] = master["SKU"].map(open_po).fillna(0.0)

    master["Avg monthly units"] = master["Units sold"] / window_months
    daily_rate = master["Units sold"] / window_days
    with np.errstate(divide="ignore", invalid="ignore"):
        master["Days of supply"] = np.where(daily_rate > 0, master["On hand"] / daily_rate, np.inf)
        master["Turns (annualised)"] = np.where(
            master["On hand"] > 0, (master["Units sold"] * 12 / window_months) / master["On hand"], np.nan
        )
        unit_cost = np.where(master["Units sold"] > 0, master["COGS"] / master["Units sold"], np.nan)
        if "Open PO qty" in master.columns:
            master["Days of supply incl. POs"] = np.where(
                daily_rate > 0, (master["On hand"] + master["Open PO qty"]) / daily_rate, np.inf
            )
    master["Est. stock value"] = (
        master["Stock value"] if "Stock value" in master.columns else master["On hand"] * unit_cost
    )

    last_sale = pd.to_datetime(master["Last sale"], errors="coerce")
    master["Last sale"] = last_sale
    master["Days since last sale"] = (today - last_sale).dt.days

    on_hand = master["On hand"]
    dos = master["Days of supply"]
    never_or_stale = last_sale.isna() | (master["Days since last sale"] > dead_days)
    master["Status"] = np.select(
        [
            (on_hand <= 0) & (master["Units sold"] > 0),
            on_hand <= 0,
            (on_hand > 0) & never_or_stale,
            (on_hand > 0) & (dos > slow_dos),
            (on_hand > 0) & (dos < risk_dos),
        ],
        ["Out of stock", "Inactive", "Dead stock", "Slow moving", "Stock-out risk"],
        default="Healthy",
    )
    master["Dead units"] = np.where(master["Status"] == "Dead stock", on_hand, 0.0)
    master["Slow units"] = np.where(master["Status"] == "Slow moving", on_hand, 0.0)
    master["Days of supply"] = master["Days of supply"].replace(np.inf, np.nan)
    if "Days of supply incl. POs" in master.columns:
        master["Days of supply incl. POs"] = master["Days of supply incl. POs"].replace(np.inf, np.nan)
    return master.sort_values("On hand", ascending=False).reset_index(drop=True)


def summarise_inventory(master: pd.DataFrame, group_column: str, window_months: int, window_days: int) -> pd.DataFrame:
    grouped = master.groupby(group_column).agg(
        **{
            "SKUs": ("SKU", "nunique"),
            "SKUs with stock": ("On hand", lambda s: int((s > 0).sum())),
            "On hand": ("On hand", "sum"),
            "Units sold": ("Units sold", "sum"),
            "Sales value": ("Sales value", "sum"),
            "Dead units": ("Dead units", "sum"),
            "Slow units": ("Slow units", "sum"),
            "Stock-out risk SKUs": ("Status", lambda s: int((s == "Stock-out risk").sum())),
            "Est. stock value": ("Est. stock value", "sum"),
        }
    )
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
    """A bordered card with a small heading and one st.metric per tile."""
    with st.container(border=True):
        st.markdown(f"<p class='card-title'>{title}</p>", unsafe_allow_html=True)
        columns = st.columns(len(tiles), gap="small")
        for column, tile in zip(columns, tiles):
            column.metric(**tile)


def chart_formats(metric: str) -> tuple[str, str, str]:
    """Axis format, tooltip format and axis title for a metric."""
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


def choices_for(series: pd.Series) -> list[str]:
    values = sorted(
        {str(value).strip() for value in series.dropna() if str(value).strip()},
        key=str.casefold,
    )
    return ["TOTAL", *values]


def filter_selectbox(label: str, series: pd.Series, default: str = "TOTAL") -> str:
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


def table_height(rows: int, cap: int = 420) -> int:
    return min(cap, int(38 + 35 * rows) + 3)


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
    ["Forecast", "Inventory Analysis", "Options Data", "Formatted Data", "Document Status"],
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
                selected_location = filter_selectbox("Location", column_or_empty(sales_data, "City"))
                selected_channel = filter_selectbox("Sales channel", column_or_empty(sales_data, "Sales Channel Net"))

                st.markdown("**Product**")
                selected_classification = filter_selectbox(
                    "Classification", column_or_empty(sales_data, "OptionsData.Fourth Product Group")
                )
                selected_supplier = filter_selectbox("Supplier", column_or_empty(sales_data, "Supplier Brand"))
                selected_class = filter_selectbox("Class", column_or_empty(sales_data, "OptionsData.Class"))
                selected_model = filter_selectbox("Model", column_or_empty(sales_data, "Model"))
                selected_sku = filter_selectbox("SKU", column_or_empty(sales_data, "SKU Key"))

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

                # The view only decides which twelve months make up "this year";
                # the forecast series are used exactly as produced.
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
                    ],
                )

                # Monthly comparison chart. In the FY view the engine is asked
                # for one extra calendar year so July–December of the first
                # financial year is present, then months are re-grouped.
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
                chart_data = pd.concat(
                    [actual_chart_data[["Month", "Series", "Line Type", "Value"]], forecast_plot_data],
                    ignore_index=True,
                )
                forecast_series = sorted(forecast_plot_data["Series"].unique(), key=label_year_number)
                series_order = list(dict.fromkeys(year_labels + forecast_series))
                palette = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#f0e442", "#000000"]
                series_colors = [palette[index % len(palette)] for index in range(len(series_order))]
                axis_format, value_format, y_title = chart_formats(metric)

                legend_selection = alt.selection_point(
                    name="series_legend",
                    fields=["Series"],
                    bind="legend",
                    toggle="true",
                    clear=False,
                )

                month_axis = alt.X(
                    "Month:N",
                    sort=month_sort_order,
                    title=None,
                    axis=quiet_axis(labelAngle=0, labelExpr="slice(datum.label, 0, 3)"),
                )

                yearly_chart = (
                    alt.Chart(chart_data)
                    .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2)
                    .encode(
                        x=month_axis,
                        y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=False), axis=value_axis(axis_format)),
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
                            scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [7, 4]]),
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
                for month in month_sort_order:
                    rows = annual_profile.loc[annual_profile["Month"].eq(month)]
                    tooltip_row = {"Month": month}
                    tooltip_fields = []
                    for year in year_columns:
                        value = rows.iloc[0][year] if not rows.empty else pd.NA
                        if pd.notna(value):
                            field = f"year_{year}"
                            tooltip_row[field] = float(value)
                            tooltip_fields.append(
                                alt.Tooltip(f"{field}:Q", title=view_label(int(year), fiscal), format=value_format)
                            )
                    month_forecasts = forecast_chart_data.loc[forecast_chart_data["Month"].eq(month)]
                    for year in sorted(month_forecasts["Year"].unique(), key=label_year_number):
                        forecast_value = month_forecasts.loc[month_forecasts["Year"].eq(year), "Forecast"]
                        if forecast_value.empty or pd.isna(forecast_value.iloc[0]):
                            continue
                        field = f"forecast_{label_year_number(year)}"
                        tooltip_row[field] = float(forecast_value.iloc[0])
                        tooltip_fields.append(alt.Tooltip(f"{field}:Q", title=f"Forecast {year}", format=value_format))
                    if tooltip_fields:
                        month_hover_selection = alt.selection_point(
                            name=f"hover_{month.lower()}", fields=["Month"], on="pointerover", clear="pointerout",
                        )
                        month_hover_charts.append(
                            alt.Chart(pd.DataFrame([tooltip_row]))
                            .mark_rule(color="#667085", strokeWidth=12)
                            .encode(
                                x=month_axis,
                                opacity=alt.condition(month_hover_selection, alt.value(0.18), alt.value(0), empty=False),
                                tooltip=tooltip_fields,
                            )
                            .add_params(month_hover_selection)
                        )

                # Legend show/hide and the month hover are handled in the browser by the
                # chart's own selections; no on_select, so a legend click doesn't rerun
                # the whole script (and layered charts can't carry one on Streamlit < 1.41).
                st.altair_chart(
                    alt.layer(yearly_chart, *month_hover_charts).properties(height=360).configure_view(strokeWidth=0),
                    **FULL_WIDTH,
                )
                st.caption(
                    f"{'FY Jul–Jun' if fiscal else 'Calendar Jan–Dec'} · Solid lines: actual · Dashed lines: forecast · "
                    "Click a legend entry to show or hide a series · Hover a month to compare every year."
                )

                if metric == "Sale":
                    target_80 = sale_target * 0.8
                    target_diff = fy_act_fc - sale_target
                    target_pct = (target_diff / sale_target) * 100 if sale_target > 0 else 0.0
                    target_80_diff = fy_act_fc - target_80
                    target_80_pct = (target_80_diff / target_80) * 100 if target_80 > 0 else 0.0

                    with st.expander(f"🎯 Target & performance comparison · {this_label}", expanded=False):
                        t_col1, t_col2, t_col3, t_col4 = st.columns(4, gap="small")
                        t_col1.metric("Full Year Outlook", fmt_str(fy_act_fc), delta=f"{target_diff:+,.0f} ({target_pct:+.1f}% vs Target)")
                        t_col2.metric("100% Sale Target", f"${sale_target:,.0f}", delta=f"{target_pct:+.1f}% to Target")
                        t_col3.metric("80% Sale Target", f"${target_80:,.0f}", delta=f"{target_80_pct:+.1f}% to 80% Target")
                        t_col4.metric(f"{prior_label} Full Year Actuals", fmt_str(prior_fy_act))

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

            attributes = option_attributes(options_raw)
            detected = detect_inventory_columns(inventory_raw)

            with st.sidebar:
                st.header("Inventory filters")
                st.markdown("**Product**")
                filter_sources = {
                    "Supplier": [attributes["Supplier"], sales_data.get("OptionsData.Supplier")],
                    "Class": [attributes["Class"], sales_data.get("OptionsData.Class")],
                    "Product group": [attributes["Product group"], sales_data.get("OptionsData.Fourth Product Group")],
                }
                product_filters: dict[str, list[str]] = {}
                for label, sources in filter_sources.items():
                    values = sorted(
                        {
                            str(value).strip()
                            for source in sources
                            if source is not None
                            for value in source.dropna()
                            if str(value).strip() and str(value).strip().lower() != "nan"
                        },
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

                mapping_needed = detected["sku"] is None or detected["on_hand"] is None
                with st.expander("Inventory column mapping", expanded=mapping_needed):
                    column_options = ["(none)"] + [str(col) for col in inventory_raw.columns]
                    mapping: dict[str, str | None] = {}
                    for role, label in INVENTORY_ROLE_LABELS.items():
                        default = str(detected[role]) if detected[role] is not None else "(none)"
                        choice = st.selectbox(
                            label,
                            column_options,
                            index=column_options.index(default) if default in column_options else 0,
                            key=f"inventory_map_{role}",
                        )
                        mapping[role] = None if choice == "(none)" else choice

            if not mapping["sku"] or not mapping["on_hand"]:
                st.warning(
                    "The SKU and on-hand quantity columns of the inventory sheet could not be identified. "
                    "Choose them under *Inventory column mapping* in the sidebar."
                )
            else:
                window_end = last_complete
                window_start = last_complete - (window_months - 1)
                window_periods = pd.period_range(window_start, window_end, freq="M")
                window_days = int(sum(period.days_in_month for period in window_periods))

                inventory_sku = inventory_by_sku(inventory_raw, mapping)
                movement = sku_movement(sales_data, str(window_start), str(window_end))
                open_po = open_po_by_sku(purchase_orders_raw)
                master = build_inventory_table(
                    inventory_sku, movement, attributes, open_po,
                    window_months, window_days, today, int(dead_days), int(slow_dos), int(risk_dos),
                )

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

                context_col, group_col = st.columns([5, 2], gap="small", vertical_alignment="center")
                with group_col:
                    group_by = st.selectbox(
                        "Group by", ["SKU", "Supplier", "Class", "Product group"], index=0,
                        key="inventory_group_by", label_visibility="collapsed",
                    )
                with context_col:
                    st.caption(f"Sales window {window_label(window_periods)} ({window_days} days) · {len(filtered):,} SKUs")

                if filtered.empty:
                    st.warning("No SKUs match these filters.")
                else:
                    total_on_hand = float(filtered["On hand"].sum())
                    total_units = float(filtered["Units sold"].sum())
                    dead_units = float(filtered["Dead units"].sum())
                    slow_units = float(filtered["Slow units"].sum())
                    status_counts = filtered["Status"].value_counts()
                    overall_daily_rate = total_units / window_days if window_days else 0.0
                    overall_dos = total_on_hand / overall_daily_rate if overall_daily_rate > 0 else np.nan
                    overall_turns = (total_units * 12 / window_months) / total_on_hand if total_on_hand > 0 else np.nan

                    def pct_of_on_hand(units: float) -> str:
                        return f"{units / total_on_hand * 100:.0f}% of on hand" if total_on_hand > 0 else "n/a"

                    health_col, movement_col = st.columns([3, 2], gap="small")
                    with health_col:
                        metric_card(
                            "Stock health",
                            [
                                dict(label="On hand (units)", value=f"{total_on_hand:,.0f}"),
                                dict(label="Dead stock", value=f"{dead_units:,.0f}", delta=pct_of_on_hand(dead_units), delta_color="off"),
                                dict(label="Slow moving", value=f"{slow_units:,.0f}", delta=pct_of_on_hand(slow_units), delta_color="off"),
                                dict(label="Stock-out risk", value=f"{int(status_counts.get('Stock-out risk', 0)):,} SKUs"),
                            ],
                        )
                    with movement_col:
                        metric_card(
                            f"Movement · last {window_months} months",
                            [
                                dict(label="Units sold", value=f"{total_units:,.0f}"),
                                dict(label="Days of supply", value=f"{overall_dos:,.0f}" if pd.notna(overall_dos) else "n/a"),
                                dict(label="Turns (annualised)", value=f"{overall_turns:.1f}×" if pd.notna(overall_turns) else "n/a"),
                            ],
                        )

                    # Stock-health charts: status breakdown, and the status mix
                    # across the attribute chosen in "Group by" (Supplier when
                    # the view is per SKU).
                    chart_group = group_by if group_by != "SKU" else "Supplier"
                    control_left, control_mid, control_right = st.columns([3, 3, 2], gap="small", vertical_alignment="center")
                    with control_left:
                        measure_label = st.radio(
                            "Measure", ["Units on hand", "SKUs", "Stock value"], index=0,
                            horizontal=True, key="inventory_chart_measure", label_visibility="collapsed",
                        )
                    with control_right:
                        show_share = st.toggle("Show as % of group", value=False, key="inventory_chart_normalize")
                    measure = {"Units on hand": "units", "SKUs": "skus", "Stock value": "value"}[measure_label]

                    chart_left, chart_right = st.columns([2, 3], gap="small")
                    with chart_left:
                        st.markdown("<p class='card-title'>Stock health</p>", unsafe_allow_html=True)
                        st.altair_chart(stock_health_breakdown(filtered, measure=measure), **FULL_WIDTH)
                    with chart_right:
                        st.markdown(
                            f"<p class='card-title'>Stock health by {chart_group.lower()} · top 15</p>",
                            unsafe_allow_html=True,
                        )
                        st.altair_chart(
                            stock_health_by_group(filtered, chart_group, measure=measure, top_n=15, normalize=show_share),
                            **FULL_WIDTH,
                        )

                    sku_columns = [
                        "SKU", "Description", "Supplier", "Class", "Product group", "Status",
                        "On hand", "Free stock", "Units sold", "Avg monthly units", "Days of supply",
                        "Turns (annualised)", "Last sale", "Days since last sale", "Sales value", "Est. stock value",
                    ]
                    if "Open PO qty" in filtered.columns:
                        sku_columns += ["Open PO qty", "Days of supply incl. POs"]
                    sku_columns = [c for c in sku_columns if c in filtered.columns]

                    sku_column_config = {
                        "On hand": st.column_config.NumberColumn(format="%d"),
                        "Free stock": st.column_config.NumberColumn(format="%d"),
                        "Units sold": st.column_config.NumberColumn(f"Units sold ({window_months}m)", format="%d"),
                        "Avg monthly units": st.column_config.NumberColumn(format="%.1f"),
                        "Days of supply": st.column_config.NumberColumn(format="%d"),
                        "Days of supply incl. POs": st.column_config.NumberColumn(format="%d"),
                        "Turns (annualised)": st.column_config.NumberColumn(format="%.1f"),
                        "Last sale": st.column_config.DatetimeColumn(format="DD MMM YYYY"),
                        "Days since last sale": st.column_config.NumberColumn(format="%d"),
                        "Sales value": st.column_config.NumberColumn(f"Sales ({window_months}m)", format="$%.0f"),
                        "Est. stock value": st.column_config.NumberColumn(format="$%.0f"),
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
                                    "On hand": st.column_config.NumberColumn(format="%d"),
                                    "Units sold": st.column_config.NumberColumn(f"Units sold ({window_months}m)", format="%d"),
                                    "Sales value": st.column_config.NumberColumn(f"Sales ({window_months}m)", format="$%.0f"),
                                    "Dead units": st.column_config.NumberColumn(format="%d"),
                                    "Slow units": st.column_config.NumberColumn(format="%d"),
                                    "Dead stock %": st.column_config.NumberColumn(format="%.0f%%"),
                                    "Days of supply": st.column_config.NumberColumn(format="%d"),
                                    "Turns (annualised)": st.column_config.NumberColumn(format="%.1f"),
                                    "Est. stock value": st.column_config.NumberColumn(format="$%.0f"),
                                },
                                height=table_height(len(summary)),
                                **FULL_WIDTH,
                            )

                    with st.expander(f"📋 All SKUs · {len(filtered):,}", expanded=False):
                        show_table(filtered, "inventory_analysis.csv", "inventory_download_all")

        except Exception as e:
            st.error(f"Something went wrong while rendering inventory analysis: {e}")

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
