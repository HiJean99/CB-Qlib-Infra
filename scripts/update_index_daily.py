import akshare as ak
import pandas as pd
import os
import argparse
from datetime import datetime

# --- Configuration ---
OUTPUT_DIR = 'csv_data'
INDEX_CODE = '000832' # CSI code for AkShare
INDEX_SYMBOL = 'SH000832' # Qlib symbol

FEATURES = [
    'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'factor', 's_close', 's_volume', 'cb_value', 'cb_over_rate', 
    'bond_value', 'bond_over_rate', 'convert_price', 'acc_convert_ratio', 
    'remain_size', 'remaining_maturity', 'issue_rating'
]

def format_date(date_str):
    if not date_str: return None
    date_str = str(date_str)
    if '-' in date_str: return date_str
    try:
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except:
        return date_str

def map_akshare_to_qlib(df):
    if df.empty:
        return df
    
    # EM columns or Hist columns mapping
    if '日期' in df.columns:
        column_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'}
    else:
        column_map = {'date': 'date', 'open': 'open', 'close': 'close', 'high': 'high', 'low': 'low', 'volume': 'volume', 'amount': 'amount'}
    
    df = df.rename(columns=column_map)
    
    # 1. Unit Conversion
    # amount: already in Yuan in these index interfaces
    df['amount'] = df['amount']
    # volume: Hands -> Units (Bonds indexing)
    df['volume'] = df['volume'] * 10

    # Static info
    df['symbol'] = INDEX_SYMBOL
    df['factor'] = 1.0
    
    # Ensure all FEATURES are present
    for col in FEATURES:
        if col not in df.columns:
            df[col] = None
            
    return df[FEATURES]

def fetch_index_incremental(symbol, start_date=None):
    print(f"Fetching incremental data for {symbol} via AkShare...")
    try:
        # stock_zh_index_daily_em is faster for recent data
        df = ak.stock_zh_index_daily_em(symbol=f"sh{symbol}")
        if df is None or df.empty:
            df = ak.index_zh_a_hist(symbol=symbol, period="daily")
            
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Standardize dates
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        else:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        if start_date:
            formatted_start = format_date(start_date)
            df = df[df['date'] >= formatted_start]
            
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True)
    args = parser.parse_args()
    
    output_file = os.path.join(OUTPUT_DIR, f"{INDEX_SYMBOL}.csv")
    
    new_data = fetch_index_incremental(INDEX_CODE, args.start_date)
    if new_data.empty:
        print("No new data fetched.")
        return
        
    new_data = map_akshare_to_qlib(new_data)
    
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        # Standardize old date format to avoid merge issues
        old_df['date'] = pd.to_datetime(old_df['date']).dt.strftime('%Y-%m-%d')
        
        # Filter old data to before start_date
        cutoff = format_date(args.start_date)
        old_df = old_df[old_df['date'] < cutoff]
        
        combined = pd.concat([old_df, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date')
    else:
        combined = new_data.sort_values('date')
        
    combined.to_csv(output_file, index=False)
    print(f"Successfully updated {output_file}. Total rows: {len(combined)}")

if __name__ == "__main__":
    main()
