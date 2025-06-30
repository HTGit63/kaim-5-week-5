#!/usr/bin/env bash
set -e

echo "🔄 Starting data pipeline..."
python src/data_processing.py
echo "✅ Data pipeline finished. Outputs in data/processed/"
