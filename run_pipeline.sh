#!/bin/bash
set -e

# Configuration
# 优先从 .env 文件读取 Token，如果没有则尝试环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

API_TOKEN=${TUSHARE_TOKEN}

if [ -z "$API_TOKEN" ]; then
    echo "Error: TUSHARE_TOKEN is not set. Please create a .env file or set the environment variable."
    exit 1
fi

START_DATE="2017-01-01"
SCRIPT_DIR="$(dirname "$0")/scripts"
OUTPUT_DIR="csv_data"  

# 1. Data Collection
echo "Starting Data Collection Phase..."
echo "Start Date: $START_DATE"

# 1.1 Fetch Bond Data
~/miniconda3/envs/q_lab/bin/python3 "$SCRIPT_DIR/collect_cb_data.py" --token "$API_TOKEN" --start_date "$START_DATE" --output_dir "$OUTPUT_DIR"

# 1.2 Fetch Index Data
echo "Fetching Benchmark Index Data..."
~/miniconda3/envs/q_lab/bin/python3 "$SCRIPT_DIR/collect_index_only.py"

echo "Pipeline Completed Successfully."
