from pathlib import Path
import pandas as pd
import re

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No se han encontrado archivos CSV en data/raw")


def extraer_anio(valor):
    """
    Extrae el año de un valor de fecha.
    Soporta formatos tipo:
    - 01/02/2017
    - 2017-02-01
    - 2017
    """
    valor = str(valor).strip()

    # Intento 1: convertir como fecha
    fecha = pd.to_datetime(valor, dayfirst=True, errors="coerce")
    if pd.notna(fecha):
        return fecha.year

    # Intento 2: buscar un año de 4 cifras dentro del texto
    match = re.search(r"(20\d{2}|19\d{2})", valor)
    if match:
        return int(match.group(1))

    raise ValueError(f"No se ha podido extraer el año del valor: {valor}")


def leer_csv(file):
    """
    Lee un CSV con separador ; y codificación latin1.
    Además elimina posibles columnas índice tipo Unnamed.
    """
    df = pd.read_csv(
        file,
        sep=";",
        encoding="latin1",
        dtype=str
    )

    # Eliminar posibles columnas índice generadas por exportaciones anteriores
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    return df


# Primer y último archivo
primer_archivo = csv_files[0]
ultimo_archivo = csv_files[-1]

# Leer primer archivo para obtener A2
df_primero = leer_csv(primer_archivo)

if df_primero.empty:
    raise ValueError(f"El archivo {primer_archivo.name} está vacío")

# A2 en Excel equivale a segunda fila de datos de la primera columna
valor_a2 = df_primero.iloc[1, 1]
anio_inicio = extraer_anio(valor_a2)

# Leer último archivo para obtener última fila no vacía de la columna A
df_ultimo = leer_csv(ultimo_archivo)

if df_ultimo.empty:
    raise ValueError(f"El archivo {ultimo_archivo.name} está vacío")

columna_a = df_ultimo.iloc[:, 1].dropna()

if columna_a.empty:
    raise ValueError(f"La columna A del archivo {ultimo_archivo.name} está vacía")

valor_final_columna_a = columna_a.iloc[-1]
anio_fin = extraer_anio(valor_final_columna_a)

print(f"Año inicial detectado: {anio_inicio}")
print(f"Año final detectado: {anio_fin}")

# Unir todos los CSV
dataframes = []

for file in csv_files:
    print(f"Leyendo: {file.name}")
    df = leer_csv(file)
    dataframes.append(df)

df_total = pd.concat(dataframes, ignore_index=True)

# Nombre automático del archivo final
output_file = PROCESSED_DIR / f"ventas_{anio_inicio}_{anio_fin}.csv"

df_total.to_csv(
    output_file,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("CSV unido correctamente")
print(f"Archivo generado: {output_file}")
print(f"Total de filas: {len(df_total)}")
print(f"Total de columnas: {len(df_total.columns)}")