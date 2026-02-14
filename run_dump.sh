#!/bin/bash
set -e

# Configuration
INPUT_DIR="csv_data"
OUTPUT_DIR="~/.qlib/qlib_data/cb_data"
SCRIPT_DIR="$(dirname "$0")/scripts"

echo "Starting Qlib Data Conversion..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

# 1. Run Dump
# Exclude string columns: issue_rating, symbol
~/miniconda3/envs/q_lab/bin/python3 "$SCRIPT_DIR/dump_bin.py" dump_all \
    --data_path "$INPUT_DIR" \
    --qlib_dir "$OUTPUT_DIR" \
    --exclude_fields "issue_rating,symbol" \
    --symbol_field_name "symbol" \
    --date_field_name "date"

echo "Data Dump Completed."

# 2. Run Verification
echo "Starting Data Verification..."
~/miniconda3/envs/q_lab/bin/python3 "$SCRIPT_DIR/check_dump_bin.py" check \
    --qlib_dir "$OUTPUT_DIR" \
    --csv_path "$INPUT_DIR"

echo "All Processes Completed Successfully."
