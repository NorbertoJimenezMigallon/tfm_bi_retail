# TFM BI Retail - Dashboard en Python y Streamlit

Prototipo funcional para el TFM **“Arquitectura de Business Intelligence y Análisis Escalable de Datos Retail en la Nube: Un Enfoque Data-Driven para la Gestión Comercial”**.

La aplicación permite cargar un CSV de ventas, limpiarlo con Pandas, calcular KPIs comerciales y visualizar los resultados en un dashboard web local con Streamlit.

## Estructura del proyecto

```text
tfm_bi_retail/
├── app/
│   ├── __init__.py
│   ├── dashboard.py          # Aplicación Streamlit
│   └── etl.py                # Limpieza, transformación y KPIs
├── data/                     # Carpeta para dejar el CSV si se quiere
├── outputs/                  # Exportaciones generadas
├── notebooks/                # Libretas exploratorias opcionales
├── docs/                     # Capturas y textos para memoria
├── scripts/
│   ├── run_windows.bat       # Lanzador Windows
│   └── run_gitbash.sh        # Lanzador Git Bash / Linux
├── tests/
├── requirements.txt
├── run_app.py                # Lanzador principal en Python
└── README.md
```

## Columnas esperadas

El CSV del TFM debería tener una estructura similar a:

```text
TIENDA;FECHA;HORA;CAJA;NUM.TICKET;COMERCIAL;COD.POSTAL;FAMILIA;NOMBRE ARTICULO;CANTIDAD;TOTAL VENTA;IMP.DESCUENTO;IMP.COSTE;PROVEEDOR
```

## Instalación

Desde la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
```

## Ejecución recomendada

```bash
python run_app.py
```

O directamente:

```bash
python -m streamlit run app/dashboard.py
```

En Windows también puedes hacer doble clic en:

```text
scripts/run_windows.bat
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

## Flujo de uso

1. Abrir la aplicación.
2. Subir el archivo CSV.
3. Seleccionar separador `;` y codificación `latin1`.
4. Elegir los análisis que se quieren ejecutar.
5. Pulsar **Iniciar análisis**.
6. Revisar KPIs, gráficos, tablas y calidad del dato.
7. Descargar resultados en CSV o Excel si se desea.

## Indicadores incluidos

- Ventas totales.
- Coste total.
- Margen bruto.
- Porcentaje de margen.
- Número de tickets.
- Ticket medio.
- Unidades vendidas.
- Descuento total.
- Ventas por comercial.
- Ventas por familia.
- Ventas por proveedor.
- Productos más vendidos.
- Evolución diaria.
- Ventas por hora.
- Calidad del dato.

## Encaje con la memoria del TFM

Este prototipo valida la arquitectura BI propuesta mediante una implementación funcional en entorno local. Python y Pandas realizan la ingesta, limpieza, transformación y cálculo de indicadores, mientras que Streamlit permite construir una interfaz web interactiva para la visualización de resultados.
