#!/bin/bash

# Default values
START_DATE=""
END_DATE=$(date +%Y%m%d)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_date) START_DATE="$2"; shift ;;
        --end_date) END_DATE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Load Token
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

API_TOKEN=${TUSHARE_TOKEN}

if [ -z "$API_TOKEN" ]; then
    echo "Error: TUSHARE_TOKEN is not set. Please create a .env file or set the environment variable."
    exit 1
fi

# Check if start_date is provided
if [ -z "$START_DATE" ]; then
    echo "Error: --start_date is required."
    echo "Usage: ./run_daily_update.sh --start_date YYYYMMDD [--end_date YYYYMMDD]"
    exit 1
fi

echo "Running High-Speed Daily Update from $START_DATE to $END_DATE..."

# 1. Update Bond Data (Tushare)
~/miniconda3/envs/q_lab/bin/python3 scripts/collect_daily_update.py \
    --token "$API_TOKEN" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --output_dir csv_data

# 2. Update Index Data (AkShare)
echo "Updating Index Data..."
~/miniconda3/envs/q_lab/bin/python3 scripts/update_index_daily.py \
    --start_date "$START_DATE"

echo "Done."
