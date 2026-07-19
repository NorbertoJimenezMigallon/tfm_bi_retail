#!/usr/bin/env bash
# Lanzador para Git Bash / Linux / macOS
cd "$(dirname "$0")/.." || exit 1
python -m streamlit run app/dashboard.py
