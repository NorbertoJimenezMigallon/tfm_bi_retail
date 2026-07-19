"""Funciones de carga, limpieza y cálculo de indicadores para el TFM BI Retail."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "TIENDA",
    "FECHA",
    "HORA",
    "CAJA",
    "NUM.TICKET",
    "COMERCIAL",
    "COD.POSTAL",
    "FAMILIA",
    "NOMBRE ARTICULO",
    "CANTIDAD",
    "TOTAL VENTA",
    "IMP.DESCUENTO",
    "IMP.COSTE",
    "PROVEEDOR",
]

NUMERIC_COLUMNS = ["CANTIDAD", "TOTAL VENTA", "IMP.DESCUENTO", "IMP.COSTE"]
CURRENCY_COLUMNS = ["TOTAL VENTA", "IMP.DESCUENTO", "IMP.COSTE", "MARGEN", "TOTAL_TICKET"]
POSTAL_COLUMNS = ["COD.POSTAL"]
TEXT_COLUMNS = [
    "TIENDA",
    "CAJA",
    "NUM.TICKET",
    "COMERCIAL",
    "COD.POSTAL",
    "FAMILIA",
    "NOMBRE ARTICULO",
    "PROVEEDOR",
]


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    duplicated_rows: int
    missing_values: int
    missing_percentage: float
    invalid_sales_rows: int
    invalid_cost_rows: int
    unique_tickets: int
    unique_products: int
    unique_families: int
    unique_suppliers: int
    unique_salespeople: int


def normalize_column_name(column: str) -> str:
    """Normaliza cabeceras sin perder compatibilidad con las columnas originales."""
    column = str(column).strip().upper()
    column = re.sub(r"\s+", " ", column)
    return column


def load_csv(uploaded_file, sep: str = ";", encoding: str = "latin1") -> pd.DataFrame:
    """Carga un CSV subido desde Streamlit."""
    try:
        return pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=sep, encoding="utf-8", errors="replace")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def parse_spanish_number(series: pd.Series) -> pd.Series:
    """Convierte importes españoles: 1.234,56 -> 1234.56."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("€", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
    return df




def normalize_postal_code(series: pd.Series) -> pd.Series:
    """Normaliza códigos postales españoles a texto de 5 cifras.

    Ejemplos:
    - 28013 -> 28013
    - 4512 -> 04512
    - 4512.0 -> 04512
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return cleaned.apply(lambda x: str(x).zfill(5) if pd.notna(x) else np.nan)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y enriquece el dataset para análisis BI."""
    df = normalize_columns(df)
    df = clean_text_columns(df, TEXT_COLUMNS)

    for col in POSTAL_COLUMNS:
        if col in df.columns:
            df[col] = normalize_postal_code(df[col])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = parse_spanish_number(df[col])

    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", dayfirst=True)
        df["AÑO"] = df["FECHA"].dt.year
        df["MES"] = df["FECHA"].dt.month
        df["MES_NOMBRE"] = df["FECHA"].dt.strftime("%Y-%m")
        df["DIA"] = df["FECHA"].dt.date
        df["DIA_SEMANA"] = df["FECHA"].dt.day_name(locale=None)

    if "HORA" in df.columns:
        hora_str = df["HORA"].astype(str).str.extract(r"(\d{1,2})", expand=False)
        df["HORA_NUM"] = pd.to_numeric(hora_str, errors="coerce")

    if {"TOTAL VENTA", "IMP.COSTE"}.issubset(df.columns):
        df["MARGEN"] = df["TOTAL VENTA"] - df["IMP.COSTE"]
        df["MARGEN_%"] = np.where(
            df["TOTAL VENTA"].abs() > 0,
            (df["MARGEN"] / df["TOTAL VENTA"]) * 100,
            np.nan,
        )

    if {"TOTAL VENTA", "NUM.TICKET"}.issubset(df.columns):
        ticket_totals = df.groupby("NUM.TICKET", dropna=False)["TOTAL VENTA"].transform("sum")
        df["TOTAL_TICKET"] = ticket_totals

    return df


def calculate_kpis(df: pd.DataFrame) -> dict[str, float | int | str]:
    total_sales = float(df["TOTAL VENTA"].sum()) if "TOTAL VENTA" in df.columns else 0.0
    total_cost = float(df["IMP.COSTE"].sum()) if "IMP.COSTE" in df.columns else 0.0
    total_margin = float(df["MARGEN"].sum()) if "MARGEN" in df.columns else total_sales - total_cost
    total_discount = float(df["IMP.DESCUENTO"].sum()) if "IMP.DESCUENTO" in df.columns else 0.0
    total_units = float(df["CANTIDAD"].sum()) if "CANTIDAD" in df.columns else 0.0
    ticket_count = int(df["NUM.TICKET"].nunique()) if "NUM.TICKET" in df.columns else len(df)
    average_ticket = total_sales / ticket_count if ticket_count else 0.0
    margin_pct = (total_margin / total_sales) * 100 if total_sales else 0.0

    return {
        "ventas_totales": total_sales,
        "coste_total": total_cost,
        "margen_bruto": total_margin,
        "margen_pct": margin_pct,
        "descuento_total": total_discount,
        "unidades_vendidas": total_units,
        "tickets": ticket_count,
        "ticket_medio": average_ticket,
        "lineas": int(len(df)),
        "comerciales": int(df["COMERCIAL"].nunique()) if "COMERCIAL" in df.columns else 0,
        "proveedores": int(df["PROVEEDOR"].nunique()) if "PROVEEDOR" in df.columns else 0,
        "familias": int(df["FAMILIA"].nunique()) if "FAMILIA" in df.columns else 0,
        "articulos": int(df["NOMBRE ARTICULO"].nunique()) if "NOMBRE ARTICULO" in df.columns else 0,
    }


def quality_report(df: pd.DataFrame) -> DataQualityReport:
    rows, columns = df.shape
    duplicated_rows = int(df.duplicated().sum())
    missing_values = int(df.isna().sum().sum())
    total_cells = rows * columns if rows and columns else 0
    missing_percentage = (missing_values / total_cells) * 100 if total_cells else 0.0
    invalid_sales_rows = int((df.get("TOTAL VENTA", pd.Series(dtype=float)) < 0).sum())
    invalid_cost_rows = int((df.get("IMP.COSTE", pd.Series(dtype=float)) < 0).sum())

    return DataQualityReport(
        rows=rows,
        columns=columns,
        duplicated_rows=duplicated_rows,
        missing_values=missing_values,
        missing_percentage=missing_percentage,
        invalid_sales_rows=invalid_sales_rows,
        invalid_cost_rows=invalid_cost_rows,
        unique_tickets=int(df["NUM.TICKET"].nunique()) if "NUM.TICKET" in df.columns else 0,
        unique_products=int(df["NOMBRE ARTICULO"].nunique()) if "NOMBRE ARTICULO" in df.columns else 0,
        unique_families=int(df["FAMILIA"].nunique()) if "FAMILIA" in df.columns else 0,
        unique_suppliers=int(df["PROVEEDOR"].nunique()) if "PROVEEDOR" in df.columns else 0,
        unique_salespeople=int(df["COMERCIAL"].nunique()) if "COMERCIAL" in df.columns else 0,
    )


def aggregate(df: pd.DataFrame, group_col: str, value_col: str = "TOTAL VENTA", top_n: int = 15) -> pd.DataFrame:
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            ventas=(value_col, "sum"),
            operaciones=(value_col, "size"),
            margen=("MARGEN", "sum") if "MARGEN" in df.columns else (value_col, "sum"),
        )
        .reset_index()
        .sort_values("ventas", ascending=False)
        .head(top_n)
    )
    return grouped


def export_to_excel_bytes(df: pd.DataFrame, kpis: dict) -> bytes:
    """Exporta resultados a Excel ordenando las ventas de mayor a menor."""
    export_df = df.copy()
    if "TOTAL VENTA" in export_df.columns:
        export_df = export_df.sort_values("TOTAL VENTA", ascending=False)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="datos_limpios")
        pd.DataFrame([kpis]).to_excel(writer, index=False, sheet_name="kpis")

        worksheet = writer.sheets["datos_limpios"]
        header_to_col = {cell.value: cell.column for cell in worksheet[1]}
        for col_name in ["TOTAL VENTA", "IMP.DESCUENTO", "IMP.COSTE", "MARGEN", "TOTAL_TICKET"]:
            if col_name in header_to_col:
                for row in worksheet.iter_rows(min_row=2, min_col=header_to_col[col_name], max_col=header_to_col[col_name]):
                    row[0].number_format = '#,##0.00 €'
        if "COD.POSTAL" in header_to_col:
            for row in worksheet.iter_rows(min_row=2, min_col=header_to_col["COD.POSTAL"], max_col=header_to_col["COD.POSTAL"]):
                row[0].number_format = '@'

    return buffer.getvalue()
