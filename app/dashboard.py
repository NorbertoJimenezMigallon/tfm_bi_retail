"""Dashboard TFM BI Retail.

Aplicación Streamlit para cargar un CSV de ventas, ejecutar ETL con Pandas
 y visualizar KPIs comerciales.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingRegressor = None
    LinearRegression = None
    KNeighborsRegressor = None
    RandomForestRegressor = None
    StandardScaler = None
    make_pipeline = None
    mean_squared_error = None
    SKLEARN_AVAILABLE = False

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.etl import (  # noqa: E402
    EXPECTED_COLUMNS,
    aggregate as etl_aggregate,
    calculate_kpis as etl_calculate_kpis,
    export_to_excel_bytes,
    load_csv,
    prepare_data,
    quality_report as etl_quality_report,
)

from PIL import Image

icono = Image.open("images/favicon.png")

st.set_page_config(
    page_title="TFM BI Retail",
    page_icon=icono,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #0068C9 !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }

    div.stButton > button:hover {
        background-color: #0054A6 !important;
        color: white !important;
        border: none !important;
    }

    .kpi-section-title {
        font-size: 1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.1rem 0 0.25rem 0;
    }

    .kpi-section-subtitle {
        font-size: 0.86rem;
        color: #64748b;
        margin: 0 0 1rem 0;
    }

    .kpi-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        min-height: 168px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.9rem;
        border-top: 4px solid var(--kpi-color, #0ea5e9);
    }

    .kpi-card__eyebrow {
        font-size: 0.69rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
    }

    .kpi-card__title {
        font-size: 0.96rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.55rem;
        line-height: 1.25;
    }

    .kpi-card__value {
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 850;
        color: var(--kpi-color, #0f172a);
        margin-bottom: 0.7rem;
        word-break: break-word;
    }

    .kpi-card__footer {
        margin-top: auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        border-top: 1px solid #edf2f7;
        padding-top: 0.65rem;
    }

    .kpi-card__footer-note {
        font-size: 0.80rem;
        color: #475569;
        font-weight: 650;
        line-height: 1.25;
    }

    .kpi-card__badge {
        white-space: nowrap;
        font-size: 0.72rem;
        font-weight: 800;
        color: var(--kpi-color, #0ea5e9);
        background: color-mix(in srgb, var(--kpi-color, #0ea5e9) 12%, white);
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
    }

    .kpi-subsection {
        font-size: 0.78rem;
        font-weight: 800;
        color: #64748b;
        margin: 0.15rem 0 0.65rem 0.1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-dim-card {
        background: #ffffff;
        border: 1px solid #edf2f7;
        border-radius: 14px;
        padding: 0.65rem 0.75rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
        margin-bottom: 0.3rem;
    }

    .kpi-dim-card__label {
        font-size: 0.73rem;
        color: #94a3b8;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }

    .kpi-dim-card__value {
        font-size: 1.35rem;
        font-weight: 850;
        color: #334155;
        line-height: 1.05;
    }

    .kpi-dim-card__note {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
        line-height: 1.2;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Cabecera con logo a la izquierda y contenido a la derecha
col_logo, col_texto = st.columns([1, 5])
with col_logo:
    st.image("images/logo.png", width=130)

with col_texto:
    st.title("Dashboard BI Retail - TFM")
    st.write("Prototipo BI desarrollado en Python para cargar ventas retail, limpiar datos, calcular indicadores comerciales y visualizar resultados en un dashboard web.")


def build_ticket_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Genera ID_TICKET como FECHA + HORA + NUM.TICKET.

    El número interno de ticket puede reiniciarse, por lo que no debe
    contarse de forma aislada. Esta función conserva NUM.TICKET original
    y añade una clave técnica para contabilizar tickets reales.
    """
    result = df.copy()

    required_cols = {"FECHA", "HORA", "NUM.TICKET"}
    if not required_cols.issubset(result.columns):
        return result

    fecha = pd.to_datetime(result["FECHA"], errors="coerce", dayfirst=True)
    fecha_key = fecha.dt.strftime("%Y-%m-%d").fillna("SIN_FECHA")

    hora_key = (
        result["HORA"]
        .astype(str)
        .str.strip()
        .replace({"": "SIN_HORA", "nan": "SIN_HORA", "NaT": "SIN_HORA", "None": "SIN_HORA"})
    )

    ticket_key = (
        result["NUM.TICKET"]
        .astype(str)
        .str.strip()
        .replace({"": "SIN_TICKET", "nan": "SIN_TICKET", "NaT": "SIN_TICKET", "None": "SIN_TICKET"})
    )

    result["ID_TICKET"] = fecha_key + "_" + hora_key + "_" + ticket_key
    return result


def df_for_ticket_count(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara una copia para cálculos donde NUM.TICKET debe contar tickets únicos."""
    result = build_ticket_identifier(df)

    if "ID_TICKET" in result.columns:
        result = result.copy()
        result["NUM.TICKET"] = result["ID_TICKET"]

    return result


def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs usando FECHA + HORA + NUM.TICKET para contar tickets."""
    return etl_calculate_kpis(df_for_ticket_count(df))


def aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Agrega tablas comerciales usando FECHA + HORA + NUM.TICKET para tickets."""
    return etl_aggregate(df_for_ticket_count(df), group_col)


def quality_report(df: pd.DataFrame):
    """Genera el informe de calidad con tickets únicos corregidos."""
    return etl_quality_report(df_for_ticket_count(df))


def format_currency(value: float) -> str:
    return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def format_number(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def parse_display_number(value) -> float | None:
    """Convierte valores numéricos para mostrarlos, aceptando formato español.

    Streamlit muestra también la vista previa del CSV sin limpiar. Por eso aquí
    admitimos valores como "-29,52", "1.234,56" o "1,234.56".
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    text = text.replace("€", "").replace(" ", "")

    # Formato español habitual: 1.234,56 o -29,52
    if "," in text:
        text = text.replace(".", "").replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def format_currency_cell(value) -> str:
    number = parse_display_number(value)
    if number is None:
        return "" if pd.isna(value) else str(value)
    return format_currency(number)


def format_integer_cell(value) -> str:
    number = parse_display_number(value)
    if number is None:
        return "" if pd.isna(value) else str(value)
    return format_number(number)


def format_postal_cell(value) -> str:
    if pd.isna(value):
        return ""
    digits = "".join(ch for ch in str(value).replace(".0", "") if ch.isdigit())
    return digits.zfill(5) if digits else ""


def format_date_cell(value) -> str:
    if pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%d/%m/%Y")


def anonymize_salesperson(value) -> str:
    """Devuelve las iniciales del comercial sin mostrar su nombre completo."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return ""

    clean = text.replace("_", " ").replace("-", " ")
    tokens = [token for token in clean.split() if token]
    initials = [token[0].upper() for token in tokens if token[0].isalpha()]

    if not initials:
        return text[:1].upper() + "."
    return ".".join(initials) + "."


def anonymize_commercial_column(df: pd.DataFrame) -> pd.DataFrame:
    """Anonimiza COMERCIAL en la capa de dashboard sin cambiar el ETL base."""
    result = df.copy()
    if "COMERCIAL" in result.columns:
        result["COMERCIAL"] = result["COMERCIAL"].apply(anonymize_salesperson)
    return result


CURRENCY_DISPLAY_COLUMNS = [
    "ventas",
    "margen",
    "TOTAL VENTA",
    "IMP.DESCUENTO",
    "IMP.COSTE",
    "MARGEN",
    "TOTAL_TICKET",
    "ticket_medio",
    "venta_media_diaria",
    "real",
    "prediccion_rf",
    "prediccion_media_movil",
    "prediccion",
    "ventas_reales",
    "ventas_predichas",
    "error_absoluto",
    "Random Forest",
    "Regresión Lineal",
    "Gradient Boosting Regressor",
    "KNN Regressor",
    "Media mensual histórica",
]
INTEGER_DISPLAY_COLUMNS = [
    "operaciones",
    "tickets",
    "CANTIDAD",
    "CAJA",
    "AÑO",
    "MES",
    "HORA_NUM",
    "unidades",
]
POSTAL_DISPLAY_COLUMNS = ["COD.POSTAL"]
DATE_DISPLAY_COLUMNS = ["FECHA", "DIA", "MES_FECHA", "Mes"]


def format_percent_cell(value) -> str:
    number = parse_display_number(value)
    if number is None:
        return "" if pd.isna(value) else str(value)
    return f"{number:.2f} %".replace(".", ",")


def show_dataframe(df: pd.DataFrame, max_rows: int = 1000, **kwargs) -> None:
    """Muestra tablas del dashboard sin HTML residual visible.

    Se aplica el formato BI antes de mostrar la tabla con `st.dataframe`,
    evitando que aparezcan etiquetas HTML como `</div>` debajo de las tablas.
    """
    if df is None or df.empty:
        st.info("No hay datos para mostrar.")
        return

    table = df.copy()
    total_rows = len(table)
    if max_rows and total_rows > max_rows:
        table = table.head(max_rows)

    for col in POSTAL_DISPLAY_COLUMNS:
        if col in table.columns:
            table[col] = table[col].apply(format_postal_cell)

    for col in DATE_DISPLAY_COLUMNS:
        if col in table.columns:
            table[col] = table[col].apply(format_date_cell)

    for col in CURRENCY_DISPLAY_COLUMNS:
        if col in table.columns:
            table[col] = table[col].apply(format_currency_cell)

    for col in INTEGER_DISPLAY_COLUMNS:
        if col in table.columns:
            table[col] = table[col].apply(format_integer_cell)

    # Formato específico para la tabla de métricas de modelos.
    if "MSE" in table.columns:
        table["MSE"] = table["MSE"].apply(format_number)
    if "RMSE" in table.columns:
        table["RMSE"] = table["RMSE"].apply(format_currency_cell)
    if "MAPE_%" in table.columns:
        table["MAPE_%"] = table["MAPE_%"].apply(format_percent_cell)
    if "Sesgo medio" in table.columns:
        table["Sesgo medio"] = table["Sesgo medio"].apply(format_currency_cell)

    height = kwargs.pop("height", min(650, max(140, 38 * (len(table) + 1))))

    try:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=height,
            **kwargs,
        )
    except TypeError:
        st.dataframe(
            table,
            use_container_width=True,
            height=height,
            **kwargs,
        )

    if max_rows and total_rows > max_rows:
        st.caption(
            f"Mostrando las primeras {max_rows:,} filas de {total_rows:,}. La exportación incluye todos los registros."
            .replace(",", ".")
        )


def sort_table_by_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena cualquier tabla comercial por ventas de mayor a menor.

    Prioriza la columna agregada `ventas` y, si no existe, usa `TOTAL VENTA`.
    Si la tabla no contiene ninguna de esas columnas, la devuelve sin cambios.
    """
    if "ventas" in df.columns:
        return df.sort_values("ventas", ascending=False)
    if "TOTAL VENTA" in df.columns:
        return df.sort_values("TOTAL VENTA", ascending=False)
    return df


def show_intro() -> None:
       with st.expander("Columnas esperadas del dataset .CSV", expanded=False):
        st.write(", ".join(EXPECTED_COLUMNS))


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros del dashboard")
    filtered = anonymize_commercial_column(df)

    if "FECHA" in filtered.columns and filtered["FECHA"].notna().any():
        min_date = filtered["FECHA"].min().date()
        max_date = filtered["FECHA"].max().date()
        selected_range = st.sidebar.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
            filtered = filtered[
                (filtered["FECHA"].dt.date >= start_date)
                & (filtered["FECHA"].dt.date <= end_date)
            ]

    for col, label in [
        ("COMERCIAL", "Comercial"),
        ("FAMILIA", "Familia"),
        ("PROVEEDOR", "Proveedor"),
        ("COD.POSTAL", "Código postal"),
    ]:
        if col in filtered.columns:
            values = sorted([v for v in filtered[col].dropna().unique()])
            selected = st.sidebar.multiselect(label, values, default=[])
            if selected:
                filtered = filtered[filtered[col].isin(selected)]

    return filtered


def show_kpi_cards(kpis: dict) -> None:
    def kpi_card(title: str, value: str, footer_note: str, *, color: str, badge: str, eyebrow: str = "Periodo filtrado") -> str:
        return f"""
        <div class="kpi-card" style="--kpi-color: {color};">
            <div>
                <div class="kpi-card__eyebrow">{eyebrow}</div>
                <div class="kpi-card__title">{title}</div>
                <div class="kpi-card__value">{value}</div>
            </div>
            <div class="kpi-card__footer">
                <div class="kpi-card__footer-note">{footer_note}</div>
                <div class="kpi-card__badge">{badge}</div>
            </div>
        </div>
        """

    def dimension_card(title: str, value: str, note: str) -> str:
        return f"""
        <div class="kpi-dim-card">
            <div class="kpi-dim-card__label">{title}</div>
            <div class="kpi-dim-card__value">{value}</div>
            <div class="kpi-dim-card__note">{note}</div>
        </div>
        """

    st.markdown('<div class="kpi-section-title">Resumen ejecutivo de KPIs comerciales</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kpi-section-subtitle">Indicadores principales calculados sobre los datos procesados y el rango de filtros activo.</div>',
        unsafe_allow_html=True,
    )

    top_cards = [
        ("Ventas totales", format_currency(kpis["ventas_totales"]), "Facturación acumulada del periodo analizado", "#059669", "Ventas"),
        ("Margen bruto", format_currency(kpis["margen_bruto"]), "Resultado comercial antes de otros costes", "#0284c7", "Margen"),
        ("Ticket medio", format_currency(kpis["ticket_medio"]), "Importe medio generado por cada operación", "#ea580c", "Ticket"),
        ("Margen %", f"{kpis['margen_pct']:.2f} %".replace(".", ","), "Rentabilidad relativa sobre las ventas", "#16a34a", "Ratio"),
    ]
    row1 = st.columns(4)
    for col, (title, value, note, color, badge) in zip(row1, top_cards):
        with col:
            st.markdown(kpi_card(title, value, note, color=color, badge=badge), unsafe_allow_html=True)

    operational_cards = [
        ("Nº tickets", format_number(kpis["tickets"]), "Operaciones únicas identificadas en el dataset", "#2563eb", "Operaciones"),
        ("Unidades vendidas", format_number(kpis["unidades_vendidas"]), "Volumen total de unidades comercializadas", "#7c3aed", "Unidades"),
        ("Descuento total", format_currency(kpis["descuento_total"]), "Importe acumulado de descuentos aplicados", "#dc2626", "Descuentos"),
        ("Líneas de venta", format_number(kpis["lineas"]), "Registros procesados tras la carga y limpieza", "#475569", "Registros"),
    ]
    row2 = st.columns(4)
    for col, (title, value, note, color, badge) in zip(row2, operational_cards):
        with col:
            st.markdown(kpi_card(title, value, note, color=color, badge=badge), unsafe_allow_html=True)

    st.markdown('<div class="kpi-subsection">Dimensiones disponibles para segmentar el análisis</div>', unsafe_allow_html=True)
    dimensional_cards = [
        ("Comerciales", format_number(kpis["comerciales"]), "Anonimizados"),
        ("Proveedores", format_number(kpis["proveedores"]), "Suministro"),
        ("Familias", format_number(kpis["familias"]), "Categorías"),
        ("Artículos", format_number(kpis["articulos"]), "Productos"),
    ]
    row3 = st.columns(4)
    for col, (title, value, note) in zip(row3, dimensional_cards):
        with col:
            st.markdown(dimension_card(title, value, note), unsafe_allow_html=True)



WEEKDAY_NAMES_ES = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}
WEEKDAY_ORDER_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
KEY_PERIOD_ORDER = ["Rebajas invierno", "Rebajas verano", "Black Friday", "Navidad", "Reyes"]


def get_black_friday(year: int) -> pd.Timestamp:
    """Devuelve la fecha de Black Friday para un año concreto."""
    november = pd.Timestamp(year=year, month=11, day=1)
    days_until_thursday = (3 - november.weekday()) % 7
    first_thursday = november + pd.Timedelta(days=days_until_thursday)
    thanksgiving = first_thursday + pd.Timedelta(weeks=3)
    return thanksgiving + pd.Timedelta(days=1)


def classify_key_period(value) -> tuple[str | None, int | None]:
    """Clasifica una fecha en campañas comerciales clave.

    Reyes se asigna al año de cierre de la campaña. Por ejemplo,
    del 25/12/2024 al 05/01/2025 se clasifica como Reyes 2025.
    """
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None, None

    year = int(date.year)
    month = int(date.month)
    day = int(date.day)

    if month == 1 and 1 <= day <= 5:
        return "Reyes", year

    if month == 12 and day >= 25:
        return "Reyes", year + 1

    if (month == 1 and day >= 7) or month == 2:
        return "Rebajas invierno", year

    if month == 7 or (month == 8 and day <= 21):
        return "Rebajas verano", year

    black_friday = get_black_friday(year)
    if black_friday - pd.Timedelta(days=3) <= date.normalize() <= black_friday + pd.Timedelta(days=3):
        return "Black Friday", year

    if month == 12 and 1 <= day <= 24:
        return "Navidad", year

    return None, None


def prepare_temporal_business_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prepara datos temporales con fecha, año, día de semana e ID de ticket."""
    result = build_ticket_identifier(df)
    result = result.dropna(subset=["FECHA"]).copy()

    result["DIA"] = pd.to_datetime(result["FECHA"], errors="coerce", dayfirst=True).dt.normalize()
    result = result.dropna(subset=["DIA"]).copy()
    result["AÑO"] = result["DIA"].dt.year
    result["DIA_SEMANA"] = result["DIA"].dt.weekday.map(WEEKDAY_NAMES_ES)

    ticket_col = "ID_TICKET" if "ID_TICKET" in result.columns else "NUM.TICKET"
    return result, ticket_col


def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla diaria con ventas, margen, tickets y unidades."""
    temporal_df, ticket_col = prepare_temporal_business_df(df)

    if temporal_df.empty:
        return pd.DataFrame()

    daily = (
        temporal_df
        .groupby(["AÑO", "DIA", "DIA_SEMANA"], as_index=False)
        .agg(
            ventas=("TOTAL VENTA", "sum"),
            margen=("MARGEN", "sum"),
            tickets=(ticket_col, "nunique"),
            unidades=("CANTIDAD", "sum"),
        )
    )
    daily["ticket_medio"] = daily["ventas"] / daily["tickets"].replace(0, pd.NA)
    daily["ticket_medio"] = daily["ticket_medio"].fillna(0)
    daily["FECHA"] = daily["DIA"]
    daily["FECHA_LABEL"] = daily["DIA"].dt.strftime("%d/%m/%Y")
    return daily


def show_key_periods_analysis(df: pd.DataFrame) -> None:
    st.subheader("Comparativa de épocas clave")

    temporal_df, ticket_col = prepare_temporal_business_df(df)
    if temporal_df.empty:
        st.info("No hay datos suficientes para comparar épocas clave.")
        return

    period_info = temporal_df["DIA"].apply(classify_key_period)
    temporal_df["EPOCA_CLAVE"] = period_info.apply(lambda item: item[0])
    temporal_df["AÑO_CAMPAÑA"] = period_info.apply(lambda item: item[1])

    campaign_df = temporal_df.dropna(subset=["EPOCA_CLAVE", "AÑO_CAMPAÑA"]).copy()
    if campaign_df.empty:
        st.info("No hay registros dentro de las épocas clave definidas.")
        return

    summary = (
        campaign_df
        .groupby(["AÑO_CAMPAÑA", "EPOCA_CLAVE"], as_index=False)
        .agg(
            ventas=("TOTAL VENTA", "sum"),
            margen=("MARGEN", "sum"),
            tickets=(ticket_col, "nunique"),
            unidades=("CANTIDAD", "sum"),
        )
    )

    summary["AÑO_CAMPAÑA"] = summary["AÑO_CAMPAÑA"].astype(int)
    summary["ticket_medio"] = summary["ventas"] / summary["tickets"].replace(0, pd.NA)
    summary["ticket_medio"] = summary["ticket_medio"].fillna(0)
    summary["EPOCA_CLAVE"] = pd.Categorical(summary["EPOCA_CLAVE"], categories=KEY_PERIOD_ORDER, ordered=True)
    summary = summary.sort_values(["AÑO_CAMPAÑA", "EPOCA_CLAVE"])

    fig = px.bar(
        summary,
        x="AÑO_CAMPAÑA",
        y="ventas",
        color="EPOCA_CLAVE",
        barmode="group",
        title="Ventas por épocas comerciales clave",
    )
    fig.update_layout(xaxis_title="Año", yaxis_title="Ventas", legend_title_text="Época")
    st.plotly_chart(fig, use_container_width=True)

    show_dataframe(summary.rename(columns={"AÑO_CAMPAÑA": "AÑO", "EPOCA_CLAVE": "Época"}))


def show_top_and_weak_days(df: pd.DataFrame) -> None:
    daily = build_daily_sales(df)
    if daily.empty:
        st.info("No hay datos diarios suficientes para calcular días fuertes y flojos.")
        return

    st.subheader("Top 5 días más fuertes de cada año")
    strongest = (
        daily
        .sort_values(["AÑO", "ventas"], ascending=[True, False])
        .groupby("AÑO", as_index=False)
        .head(5)
        .sort_values(["AÑO", "ventas"], ascending=[True, False])
    )

    fig_strong = px.bar(
        strongest,
        x="FECHA_LABEL",
        y="ventas",
        color="AÑO",
        title="Cinco días con mayor facturación de cada año",
    )
    fig_strong.update_layout(xaxis_title="Fecha", yaxis_title="Ventas", legend_title_text="Año")
    st.plotly_chart(fig_strong, use_container_width=True)
    show_dataframe(strongest[["AÑO", "FECHA", "DIA_SEMANA", "ventas", "margen", "tickets", "ticket_medio", "unidades"]])

    st.subheader("Top 5 días más flojos de cada año")
    weakest = (
        daily[daily["tickets"] > 0]
        .sort_values(["AÑO", "ventas"], ascending=[True, True])
        .groupby("AÑO", as_index=False)
        .head(5)
        .sort_values(["AÑO", "ventas"], ascending=[True, True])
    )

    fig_weak = px.bar(
        weakest,
        x="FECHA_LABEL",
        y="ventas",
        color="AÑO",
        title="Cinco días con menor facturación de cada año",
    )
    fig_weak.update_layout(xaxis_title="Fecha", yaxis_title="Ventas", legend_title_text="Año")
    st.plotly_chart(fig_weak, use_container_width=True)
    show_dataframe(weakest[["AÑO", "FECHA", "DIA_SEMANA", "ventas", "margen", "tickets", "ticket_medio", "unidades"]])


def show_low_activity_weekdays(df: pd.DataFrame) -> None:
    st.subheader("Días de la semana con menor actividad")

    daily = build_daily_sales(df)
    if daily.empty:
        st.info("No hay datos diarios suficientes para analizar días de la semana.")
        return

    weekday_summary = (
        daily[daily["tickets"] > 0]
        .groupby("DIA_SEMANA", as_index=False)
        .agg(
            ventas=("ventas", "mean"),
            margen=("margen", "mean"),
            tickets=("tickets", "mean"),
            unidades=("unidades", "mean"),
        )
    )

    if weekday_summary.empty:
        st.info("No hay actividad suficiente para calcular días flojos de la semana.")
        return

    weekday_summary["orden"] = weekday_summary["DIA_SEMANA"].apply(lambda x: WEEKDAY_ORDER_ES.index(x) if x in WEEKDAY_ORDER_ES else 99)
    weekday_summary = weekday_summary.sort_values("ventas")
    weakest_weekdays = weekday_summary.head(3).drop(columns=["orden"])

    fig_weekdays = px.bar(
        weekday_summary.drop(columns=["orden"]),
        x="DIA_SEMANA",
        y="ventas",
        title="Ventas medias por día de la semana",
    )
    fig_weekdays.update_layout(xaxis_title="Día de la semana", yaxis_title="Ventas medias")
    st.plotly_chart(fig_weekdays, use_container_width=True)

    st.markdown("**Tres días de la semana con menor actividad media**")
    show_dataframe(weakest_weekdays)


def show_low_summer_periods(df: pd.DataFrame) -> None:
    st.subheader("Comparativa semanal de verano para vacaciones")

    daily = build_daily_sales(df)
    if daily.empty:
        st.info("No hay datos diarios suficientes para analizar semanas de verano.")
        return

    summer_blocks = [
        ("Jun3", 6, 15, 6, 21),
        ("Jun4", 6, 22, 6, 30),
        ("Jul1", 7, 1, 7, 7),
        ("Jul2", 7, 8, 7, 14),
        ("Jul3", 7, 15, 7, 21),
        ("Jul4", 7, 22, 7, 31),
        ("Ago1", 8, 1, 8, 7),
        ("Ago2", 8, 8, 8, 14),
        ("Ago3", 8, 15, 8, 21),
        ("Ago4", 8, 22, 8, 31),
        ("Sep1", 9, 1, 9, 7),
        ("Sep2", 9, 8, 9, 15),
    ]
    week_order = [block[0] for block in summer_blocks]

    weekly_rows = []
    for year in sorted(daily["AÑO"].dropna().unique()):
        year = int(year)

        for order, (label, start_month, start_day, end_month, end_day) in enumerate(summer_blocks, start=1):
            week_start = pd.Timestamp(year=year, month=start_month, day=start_day)
            week_end = pd.Timestamp(year=year, month=end_month, day=end_day)

            week_data = daily[
                (daily["DIA"] >= week_start)
                & (daily["DIA"] <= week_end)
            ]

            ventas = week_data["ventas"].sum() if not week_data.empty else 0
            margen = week_data["margen"].sum() if not week_data.empty else 0
            tickets = week_data["tickets"].sum() if not week_data.empty else 0
            unidades = week_data["unidades"].sum() if not week_data.empty else 0
            dias_con_ventas = week_data[week_data["tickets"] > 0]["DIA"].nunique() if not week_data.empty else 0
            dias_periodo = (week_end - week_start).days + 1

            weekly_rows.append(
                {
                    "AÑO": year,
                    "SEMANA_VERANO": label,
                    "orden_semana": order,
                    "Inicio": week_start,
                    "Fin": week_end,
                    "Periodo": f"{week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m')}",
                    "ventas": ventas,
                    "margen": margen,
                    "tickets": tickets,
                    "venta_media_diaria": ventas / dias_periodo if dias_periodo else 0,
                    "unidades": unidades,
                    "dias_con_ventas": dias_con_ventas,
                }
            )

    if not weekly_rows:
        st.info("No se han podido calcular semanas entre el 15 de junio y el 15 de septiembre.")
        return

    weekly = pd.DataFrame(weekly_rows)
    weekly["SEMANA_VERANO"] = pd.Categorical(weekly["SEMANA_VERANO"], categories=week_order, ordered=True)
    weekly = weekly.sort_values(["orden_semana", "AÑO"])

    chart_type = st.radio(
        "Tipo de comparativa semanal",
        options=["Líneas por año", "Columnas agrupadas"],
        horizontal=True,
        key="summer_week_chart_type",
    )

    if chart_type == "Líneas por año":
        fig_summer = px.line(
            weekly,
            x="SEMANA_VERANO",
            y="ventas",
            color="AÑO",
            markers=True,
            title="Comparativa anual de ventas por semanas de verano",
        )
    else:
        fig_summer = px.bar(
            weekly,
            x="SEMANA_VERANO",
            y="ventas",
            color="AÑO",
            barmode="group",
            title="Comparativa anual de ventas por semanas de verano",
        )

    fig_summer.update_layout(xaxis_title="Semana de verano", yaxis_title="Ventas", legend_title_text="Año")
    st.plotly_chart(fig_summer, use_container_width=True)

    ranking = weekly.sort_values("ventas", ascending=True).reset_index(drop=True)
    ranking.insert(0, "ranking", range(1, len(ranking) + 1))

    st.markdown("**Ranking de menor a mayor facturación**")
    show_dataframe(
        ranking[
            [
                "ranking",
                "AÑO",
                "SEMANA_VERANO",
                "Inicio",
                "Fin",
                "Periodo",
                "ventas",
                "margen",
                "tickets",
                "venta_media_diaria",
                "unidades",
                "dias_con_ventas",
            ]
        ]
    )

def show_temporal_analysis(df: pd.DataFrame) -> None:
    st.subheader("Evolución temporal")
    if "FECHA" not in df.columns or "TOTAL VENTA" not in df.columns:
        st.info("No hay columnas suficientes para el análisis temporal.")
        return

    df_tickets = build_ticket_identifier(df)

    temporal = (
        df_tickets.dropna(subset=["FECHA"])
        .groupby("DIA", as_index=False)
        .agg(ventas=("TOTAL VENTA", "sum"), margen=("MARGEN", "sum"), tickets=("ID_TICKET", "nunique"))
    )
    if temporal.empty:
        st.info("No se han podido construir series temporales con las fechas disponibles.")
        return

    fig = px.line(temporal, x="DIA", y="ventas", markers=True, title="Ventas por día")
    st.plotly_chart(fig, use_container_width=True)

    if "HORA_NUM" in df.columns:
        hourly = (
            df.dropna(subset=["HORA_NUM"])
            .groupby("HORA_NUM", as_index=False)
            .agg(ventas=("TOTAL VENTA", "sum"), operaciones=("TOTAL VENTA", "size"))
            .sort_values("HORA_NUM")
        )
        fig_hour = px.bar(hourly, x="HORA_NUM", y="ventas", title="Ventas por hora")
        st.plotly_chart(fig_hour, use_container_width=True)

    st.divider()
    show_key_periods_analysis(df_tickets)

    st.divider()
    show_top_and_weak_days(df_tickets)

    st.divider()
    show_low_activity_weekdays(df_tickets)

    st.divider()
    show_low_summer_periods(df_tickets)


def show_ranking(df: pd.DataFrame, group_col: str, title: str) -> None:
    ranking = aggregate(df, group_col)
    if ranking.empty:
        st.info(f"No hay datos suficientes para {title.lower()}.")
        return
    fig = px.bar(ranking, x=group_col, y="ventas", title=title, text_auto=".2s")
    fig.update_layout(xaxis_title="", yaxis_title="Ventas")
    st.plotly_chart(fig, use_container_width=True)
    show_dataframe(sort_table_by_sales(ranking))


def show_margin_analysis(df: pd.DataFrame) -> None:
    st.subheader("Análisis de margen comercial")
    if "MARGEN" not in df.columns:
        st.info("No se puede calcular margen porque faltan TOTAL VENTA o IMP.COSTE.")
        return

    c1, c2 = st.columns(2)
    with c1:
        show_ranking(df, "FAMILIA", "Margen y ventas por familia")
    with c2:
        show_ranking(df, "PROVEEDOR", "Margen y ventas por proveedor")

    if "MARGEN_%" in df.columns:
        fig = px.histogram(df, x="MARGEN_%", nbins=40, title="Distribución del margen porcentual")
        st.plotly_chart(fig, use_container_width=True)


def show_quality(df: pd.DataFrame) -> None:
    st.subheader("Calidad del dato")
    report = quality_report(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", format_number(report.rows))
    c2.metric("Columnas", format_number(report.columns))
    c3.metric("Duplicados", format_number(report.duplicated_rows))
    c4.metric("Nulos", format_number(report.missing_values))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("% nulos", f"{report.missing_percentage:.2f} %".replace(".", ","))
    c6.metric("Tickets únicos", format_number(report.unique_tickets))
    c7.metric("Artículos únicos", format_number(report.unique_products))
    c8.metric("Proveedores únicos", format_number(report.unique_suppliers))

    missing_by_col = df.isna().sum().reset_index()
    missing_by_col.columns = ["columna", "nulos"]
    show_dataframe(missing_by_col)


def show_cleaning_summary(df: pd.DataFrame) -> None:
    st.subheader("Limpieza y normalización aplicada")
    st.markdown(
        """
        - Importes normalizados como valores numéricos y presentados en formato euro con dos decimales.
        - Campo `COD.POSTAL` tratado como texto de 5 cifras, completando con cero a la izquierda cuando procede.
        - Campo `COMERCIAL` anonimizado mediante iniciales para no mostrar nombres completos de vendedores.
        - Tablas comerciales ordenadas por ventas de mayor a menor.
        - Fechas y horas transformadas para permitir análisis temporal.
        - Margen calculado como `TOTAL VENTA - IMP.COSTE`.
        """
    )
    preview_cols = [c for c in ["FECHA", "HORA", "COMERCIAL", "COD.POSTAL", "FAMILIA", "TOTAL VENTA", "IMP.COSTE", "MARGEN"] if c in df.columns]
    if preview_cols:
        show_dataframe(sort_table_by_sales(df[preview_cols].head(20)))



def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calcula MAPE evitando divisiones entre cero."""
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return ((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()).mean() * 100


def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las ventas a nivel mensual."""
    temporal_df, ticket_col = prepare_temporal_business_df(df)

    if temporal_df.empty or "TOTAL VENTA" not in temporal_df.columns:
        return pd.DataFrame()

    temporal_df["MES_FECHA"] = temporal_df["DIA"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        temporal_df
        .groupby("MES_FECHA", as_index=False)
        .agg(
            ventas=("TOTAL VENTA", "sum"),
            margen=("MARGEN", "sum"),
            tickets=(ticket_col, "nunique"),
            unidades=("CANTIDAD", "sum"),
        )
        .sort_values("MES_FECHA")
        .reset_index(drop=True)
    )

    return monthly


def add_prediction_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Crea variables temporales, retardos y medias móviles para Random Forest."""
    data = monthly.copy().sort_values("MES_FECHA").reset_index(drop=True)

    data["AÑO"] = data["MES_FECHA"].dt.year
    data["MES"] = data["MES_FECHA"].dt.month
    data["TRIMESTRE"] = data["MES_FECHA"].dt.quarter
    data["MES_NUM"] = data["AÑO"] * 12 + data["MES"]

    data["ES_NAVIDAD"] = (data["MES"] == 12).astype(int)
    data["ES_VERANO"] = data["MES"].isin([6, 7, 8, 9]).astype(int)
    data["ES_REBAJAS"] = data["MES"].isin([1, 2, 7, 8]).astype(int)

    data["VENTAS_MES_ANTERIOR"] = data["ventas"].shift(1)
    data["VENTAS_2_MESES_ANTES"] = data["ventas"].shift(2)
    data["VENTAS_12_MESES_ANTES"] = data["ventas"].shift(12)
    data["MEDIA_MOVIL_3M"] = data["ventas"].shift(1).rolling(3).mean()
    data["MEDIA_MOVIL_6M"] = data["ventas"].shift(1).rolling(6).mean()

    return data


PREDICTION_FEATURES = [
    "AÑO",
    "MES",
    "TRIMESTRE",
    "MES_NUM",
    "ES_NAVIDAD",
    "ES_VERANO",
    "ES_REBAJAS",
    "VENTAS_MES_ANTERIOR",
    "VENTAS_2_MESES_ANTES",
    "VENTAS_12_MESES_ANTES",
    "MEDIA_MOVIL_3M",
    "MEDIA_MOVIL_6M",
]


def evaluate_prediction_models(monthly: pd.DataFrame):
    """Entrena y evalúa media móvil y Random Forest sobre ventas mensuales."""
    featured = add_prediction_features(monthly)
    model_data = featured.dropna(subset=PREDICTION_FEATURES + ["ventas"]).copy()

    if len(model_data) < 18:
        return None

    test_size = min(12, max(6, int(len(model_data) * 0.2)))
    train = model_data.iloc[:-test_size].copy()
    test = model_data.iloc[-test_size:].copy()

    x_train = train[PREDICTION_FEATURES]
    y_train = train["ventas"]
    x_test = test[PREDICTION_FEATURES]
    y_test = test["ventas"]

    baseline_pred = test["MEDIA_MOVIL_3M"]

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )
    rf_model.fit(x_train, y_train)
    rf_pred = pd.Series(rf_model.predict(x_test), index=test.index)

    mse_baseline = mean_squared_error(y_test, baseline_pred)
    mse_rf = mean_squared_error(y_test, rf_pred)

    metrics = pd.DataFrame(
        [
            {
                "Modelo": "Media móvil 3 meses",
                "MSE": mse_baseline,
                "RMSE": mse_baseline ** 0.5,
                "MAPE_%": safe_mape(y_test, baseline_pred),
            },
            {
                "Modelo": "Random Forest",
                "MSE": mse_rf,
                "RMSE": mse_rf ** 0.5,
                "MAPE_%": safe_mape(y_test, rf_pred),
            },
        ]
    )

    evaluation = test[["MES_FECHA", "ventas"]].copy()
    evaluation = evaluation.rename(columns={"ventas": "real"})
    evaluation["prediccion_media_movil"] = baseline_pred.values
    evaluation["prediccion_rf"] = rf_pred.values
    evaluation["error_absoluto"] = (evaluation["real"] - evaluation["prediccion_rf"]).abs()

    final_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    )
    final_model.fit(model_data[PREDICTION_FEATURES], model_data["ventas"])

    return metrics, evaluation, final_model


def build_future_prediction(monthly: pd.DataFrame, model, horizon_months: int) -> pd.DataFrame:
    """Genera predicción iterativa para los próximos meses."""
    history = monthly[["MES_FECHA", "ventas"]].copy().sort_values("MES_FECHA").reset_index(drop=True)
    future_rows = []

    for _ in range(horizon_months):
        next_month = history["MES_FECHA"].max() + pd.DateOffset(months=1)
        year = int(next_month.year)
        month = int(next_month.month)

        def lag_value(lag: int) -> float:
            if len(history) >= lag:
                return float(history["ventas"].iloc[-lag])
            return float(history["ventas"].mean())

        row = {
            "AÑO": year,
            "MES": month,
            "TRIMESTRE": int(pd.Timestamp(next_month).quarter),
            "MES_NUM": year * 12 + month,
            "ES_NAVIDAD": int(month == 12),
            "ES_VERANO": int(month in [6, 7, 8, 9]),
            "ES_REBAJAS": int(month in [1, 2, 7, 8]),
            "VENTAS_MES_ANTERIOR": lag_value(1),
            "VENTAS_2_MESES_ANTES": lag_value(2),
            "VENTAS_12_MESES_ANTES": lag_value(12),
            "MEDIA_MOVIL_3M": float(history["ventas"].tail(3).mean()),
            "MEDIA_MOVIL_6M": float(history["ventas"].tail(6).mean()),
        }

        prediction = float(model.predict(pd.DataFrame([row])[PREDICTION_FEATURES])[0])

        future_rows.append(
            {
                "MES_FECHA": next_month,
                "prediccion": prediction,
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame([{"MES_FECHA": next_month, "ventas": prediction}]),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)


def train_complementary_future_models(monthly: pd.DataFrame, rf_model) -> dict:
    """Entrena modelos complementarios con todo el histórico mensual válido."""
    featured = add_prediction_features(monthly)
    model_data = featured.dropna(subset=PREDICTION_FEATURES + ["ventas"]).copy()

    if len(model_data) < 18:
        return {}

    x_train = model_data[PREDICTION_FEATURES]
    y_train = model_data["ventas"]
    knn_neighbors = min(5, len(x_train))

    models = {
        "Random Forest": rf_model,
        "Regresión Lineal": LinearRegression(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "KNN Regressor": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=knn_neighbors)),
    }

    for model_name, model in models.items():
        if model_name != "Random Forest":
            model.fit(x_train, y_train)

    return models


def build_monthly_historical_average(monthly: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """Calcula la media histórica del mismo mes para cada mes futuro."""
    history = monthly[["MES_FECHA", "ventas"]].copy().sort_values("MES_FECHA").reset_index(drop=True)
    if history.empty:
        return pd.DataFrame()

    global_mean = float(history["ventas"].mean())
    future_rows = []

    for step in range(1, horizon_months + 1):
        next_month = history["MES_FECHA"].max() + pd.DateOffset(months=step)
        same_month_values = history[history["MES_FECHA"].dt.month == int(next_month.month)]["ventas"]
        monthly_mean = same_month_values.mean()
        if pd.isna(monthly_mean):
            monthly_mean = global_mean

        future_rows.append(
            {
                "MES_FECHA": next_month,
                "Modelo": "Media mensual histórica",
                "prediccion": float(monthly_mean),
            }
        )

    return pd.DataFrame(future_rows)


def build_future_model_comparison(monthly: pd.DataFrame, rf_model, horizon_months: int) -> pd.DataFrame:
    """Genera la comparativa futura entre modelos y media mensual histórica."""
    models = train_complementary_future_models(monthly, rf_model)
    if not models:
        return pd.DataFrame()

    future_parts = []
    for model_name, model in models.items():
        model_future = build_future_prediction(monthly, model, horizon_months)
        if model_future.empty:
            continue
        model_future["Modelo"] = model_name
        future_parts.append(model_future[["MES_FECHA", "Modelo", "prediccion"]])

    historical_average = build_monthly_historical_average(monthly, horizon_months)
    if not historical_average.empty:
        future_parts.append(historical_average)

    if not future_parts:
        return pd.DataFrame()

    return pd.concat(future_parts, ignore_index=True)


def show_prediction_analysis(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="background-color:#E7F6EC;border-left:8px solid #1B8F3A;padding:18px 20px;border-radius:10px;margin-bottom:18px;">
            <h2 style="margin:0;color:#12612A;">Predicción de ventas</h2>
            <p style="margin:6px 0 0 0;color:#1F5130;">
                Módulo de aprendizaje automático para estimar ventas mensuales futuras mediante Random Forest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not SKLEARN_AVAILABLE:
        st.error(
            "No se puede ejecutar la predicción porque falta scikit-learn. "
            "Instálalo con: python -m pip install scikit-learn"
        )
        return

    if "FECHA" not in df.columns or "TOTAL VENTA" not in df.columns:
        st.info("No hay columnas suficientes para generar la predicción.")
        return

    monthly = build_monthly_sales(df)

    if monthly.empty or len(monthly) < 24:
        st.info("Se necesitan al menos 24 meses de datos para entrenar un modelo de predicción mensual.")
        return

    result = evaluate_prediction_models(monthly)
    if result is None:
        st.info("No hay suficientes meses completos para evaluar el modelo tras generar retardos temporales.")
        return

    metrics, evaluation, model = result

    st.subheader("Evaluación del modelo")
    st.write(
        "La predicción se evalúa respetando el orden temporal de los datos. "
        "Los meses más recientes se reservan como conjunto de prueba."
    )

    c1, c2, c3 = st.columns(3)
    rf_metrics = metrics[metrics["Modelo"] == "Random Forest"].iloc[0]
    c1.metric("MSE Random Forest", format_number(rf_metrics["MSE"]))
    c2.metric("RMSE Random Forest", format_currency(rf_metrics["RMSE"]))
    c3.metric("MAPE Random Forest", f"{rf_metrics['MAPE_%']:.2f} %".replace(".", ","))

    show_dataframe(metrics)

    eval_plot = evaluation.melt(
        id_vars="MES_FECHA",
        value_vars=["real", "prediccion_media_movil", "prediccion_rf"],
        var_name="Serie",
        value_name="ventas",
    )

    fig_eval = px.line(
        eval_plot,
        x="MES_FECHA",
        y="ventas",
        color="Serie",
        markers=True,
        title="Ventas reales vs predicción en el periodo de prueba",
    )
    fig_eval.update_layout(xaxis_title="Mes", yaxis_title="Ventas")
    st.plotly_chart(fig_eval, use_container_width=True)

    st.subheader("Predicción de próximos meses")
    horizon = st.selectbox(
        "Horizonte de predicción",
        options=[3, 6, 12],
        index=1,
        key="prediction_horizon_months",
    )

    future = build_future_prediction(monthly, model, horizon)
    show_dataframe(future.rename(columns={"MES_FECHA": "Mes", "prediccion": "prediccion"}))

    historical = monthly[["MES_FECHA", "ventas"]].copy()
    historical["Tipo"] = "Histórico"
    historical = historical.rename(columns={"ventas": "valor"})

    future_plot = future.rename(columns={"prediccion": "valor"}).copy()
    future_plot["Tipo"] = "Predicción"

    combined = pd.concat(
        [
            historical.tail(24),
            future_plot[["MES_FECHA", "valor", "Tipo"]],
        ],
        ignore_index=True,
    )

    fig_future = px.line(
        combined,
        x="MES_FECHA",
        y="valor",
        color="Tipo",
        markers=True,
        title="Histórico reciente y predicción futura de ventas",
    )
    fig_future.update_layout(xaxis_title="Mes", yaxis_title="Ventas")
    st.plotly_chart(fig_future, use_container_width=True)

    st.divider()
    st.subheader("Comparativa complementaria de predicción futura")
    st.write(
        "A partir del mismo horizonte seleccionado, se comparan las predicciones futuras "
        "de distintos modelos supervisados con la media mensual histórica. Esta media toma, "
        "para cada mes previsto, el promedio de ese mismo mes en los años anteriores; por ejemplo, "
        "para julio se calcula la media histórica de los meses de julio disponibles."
    )

    future_comparison = build_future_model_comparison(monthly, model, horizon)
    if future_comparison.empty:
        st.info("No hay datos suficientes para generar la comparativa futura de modelos.")
    else:
        fig_model_comparison = px.bar(
            future_comparison,
            x="MES_FECHA",
            y="prediccion",
            color="Modelo",
            barmode="group",
            title="Predicción futura por modelo y media mensual histórica",
        )
        fig_model_comparison.update_layout(xaxis_title="Mes", yaxis_title="Ventas previstas")
        st.plotly_chart(fig_model_comparison, use_container_width=True)

        st.markdown("**Tabla de predicciones futuras por modelo**")
        future_comparison_table = (
            future_comparison
            .pivot_table(index="MES_FECHA", columns="Modelo", values="prediccion", aggfunc="first")
            .reset_index()
            .rename(columns={"MES_FECHA": "Mes"})
        )
        show_dataframe(future_comparison_table)

    with st.expander("Detalle de la serie mensual utilizada"):
        monthly_detail = monthly.copy()
        monthly_detail["ticket_medio"] = monthly_detail["ventas"] / monthly_detail["tickets"].replace(0, pd.NA)
        monthly_detail["ticket_medio"] = monthly_detail["ticket_medio"].fillna(0)
        show_dataframe(monthly_detail)

def dashboard(df: pd.DataFrame, selected_sections: list[str]) -> None:
    filtered_df = sidebar_filters(df)
    kpis = calculate_kpis(filtered_df)

    st.header("Dashboard de resultados")
    st.caption(f"Registros tras filtros: {len(filtered_df):,}".replace(",", "."))

    if "KPIs generales" in selected_sections:
        show_kpi_cards(kpis)

    tabs = st.tabs([
        "Temporal",
        "Comerciales",
        "Familias",
        "Proveedores",
        "Código postal",
        "Productos",
        "Margen",
        "Limpieza",
        "Calidad",
        "🟢 Predicción",
        "Datos",
    ])

    with tabs[0]:
        if "Evolución temporal" in selected_sections:
            show_temporal_analysis(filtered_df)
        else:
            st.info("Análisis temporal no seleccionado.")

    with tabs[1]:
        if "Ventas por comercial" in selected_sections:
            show_ranking(filtered_df, "COMERCIAL", "Top comerciales por ventas")
        else:
            st.info("Análisis por comercial no seleccionado.")

    with tabs[2]:
        if "Ventas por familia" in selected_sections:
            show_ranking(filtered_df, "FAMILIA", "Top familias por ventas")
        else:
            st.info("Análisis por familia no seleccionado.")

    with tabs[3]:
        if "Ventas por proveedor" in selected_sections:
            show_ranking(filtered_df, "PROVEEDOR", "Top proveedores por ventas")
        else:
            st.info("Análisis por proveedor no seleccionado.")

    with tabs[4]:
        if "Análisis por código postal" in selected_sections:
            show_ranking(filtered_df, "COD.POSTAL", "Top códigos postales por ventas")
        else:
            st.info("Análisis por código postal no seleccionado.")

    with tabs[5]:
        if "Productos más vendidos" in selected_sections:
            product_rank = aggregate(filtered_df, "NOMBRE ARTICULO")
            if not product_rank.empty:
                fig = px.bar(product_rank, x="NOMBRE ARTICULO", y="ventas", title="Top artículos por ventas")
                st.plotly_chart(fig, use_container_width=True)
                show_dataframe(sort_table_by_sales(product_rank))
            else:
                st.info("No hay datos suficientes para productos.")
        else:
            st.info("Análisis de productos no seleccionado.")

    with tabs[6]:
        if "Margen comercial" in selected_sections:
            show_margin_analysis(filtered_df)
        else:
            st.info("Análisis de margen no seleccionado.")

    with tabs[7]:
        show_cleaning_summary(filtered_df)

    with tabs[8]:
        if "Calidad del dato" in selected_sections:
            show_quality(filtered_df)
        else:
            st.info("Calidad del dato no seleccionada.")

    with tabs[9]:
        if "Predicción de ventas" in selected_sections:
            show_prediction_analysis(filtered_df)
        else:
            st.info("Predicción de ventas no seleccionada.")

    with tabs[10]:
        st.subheader("Datos procesados")
        show_dataframe(sort_table_by_sales(filtered_df))
        export_df = sort_table_by_sales(filtered_df)
        csv_bytes = export_df.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            "Descargar datos filtrados en CSV",
            data=csv_bytes,
            file_name="datos_filtrados_tfm.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar datos y KPIs en Excel",
            data=export_to_excel_bytes(export_df, kpis),
            file_name="resultados_tfm_bi_retail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main() -> None:
    show_intro()

    st.header("1. Carga del dataset")
    uploaded_file = st.file_uploader("Sube el archivo CSV de ventas", type=["csv"])

    c1, c2 = st.columns(2)
    sep = c1.selectbox("Separador", options=[";", ",", "|", "\t"], index=0)
    encoding = c2.selectbox("Codificación", options=["latin1", "utf-8", "utf-8-sig", "cp1252"], index=0)

    if uploaded_file is None:
        st.info("Sube un archivo CSV con la misma estructura para comenzar.")
        return

    raw_df = load_csv(uploaded_file, sep=sep, encoding=encoding)
    st.success("Archivo cargado correctamente.")

    st.subheader("Vista previa del archivo original")
    show_dataframe(anonymize_commercial_column(raw_df.head(20)))
    st.write(f"Filas: **{raw_df.shape[0]}** | Columnas: **{raw_df.shape[1]}**")

    st.header("2. Selección de análisis")
    default_sections = [
        "KPIs generales",
        "Evolución temporal",
        "Ventas por comercial",
        "Ventas por familia",
        "Ventas por proveedor",
        "Análisis por código postal",
        "Productos más vendidos",
        "Margen comercial",
        "Calidad del dato",
        "Predicción de ventas",
    ]
    selected_sections = st.multiselect(
        "Elige los bloques analíticos que quieres ejecutar",
        options=default_sections,
        default=default_sections,
    )

    if st.button("Iniciar análisis", type="primary"):
        with st.spinner("Procesando datos y calculando indicadores..."):
            st.session_state["df_processed"] = prepare_data(raw_df)
            st.session_state["selected_sections"] = selected_sections

    if "df_processed" in st.session_state:
        dashboard(st.session_state["df_processed"], st.session_state.get("selected_sections", selected_sections))


if __name__ == "__main__":
    main()
