import akshare as ak
import pandas as pd
import os
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

def map_akshare_to_qlib(df):
    """
    Map AkShare index_zh_a_hist columns to Qlib schema.
    """
    if df.empty:
        return df
    
    # Mapping
    column_map = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'amount'
    }
    df = df.rename(columns=column_map)
    
    # 1. Unit Conversion
    # amount: already in Yuan in index_zh_a_hist
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

def fetch_index_full_ak(symbol):
    print(f"Fetching full history for CSI Index {symbol} via AkShare...")
    try:
        # Use index_zh_a_hist which covers CSI indices well and has long history
        df = ak.index_zh_a_hist(symbol=symbol, period="daily")
        if df is None or df.empty:
            print("No data returned from AkShare.")
            return pd.DataFrame()
        
        print(f"Fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error fetching from AkShare: {e}")
        return pd.DataFrame()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = fetch_index_full_ak(INDEX_CODE)
    
    if not df.empty:
        df = map_akshare_to_qlib(df)
        df = df.sort_values(by='date')
        output_file = os.path.join(OUTPUT_DIR, f"{INDEX_SYMBOL}.csv")
        df.to_csv(output_file, index=False)
        print(f"Success! Data saved to {output_file}")
    else:
        print("Failed to save data.")

if __name__ == "__main__":
    main()
