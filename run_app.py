"""
Lanzador del dashboard BI Retail del TFM.

Uso recomendado:
    python run_app.py

También se puede lanzar directamente con:
    python -m streamlit run app/dashboard.py
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dashboard_path = project_root / "app" / "dashboard.py"

    if not dashboard_path.exists():
        raise FileNotFoundError(f"No se encuentra el dashboard: {dashboard_path}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
