import tushare as ts
import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
FEATURES = [
    'symbol', 'date', 
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'factor', 
    's_close', 's_volume', 
    'cb_value', 'cb_over_rate', 'bond_value', 'bond_over_rate', 
    'convert_price', 'acc_convert_ratio', 'remain_size', 
    'remaining_maturity', 'issue_rating'
]

COL_MAPPING = {
    'vol': 'volume',
    'ts_code': 'symbol',
    'trade_date': 'date'
}

def format_symbol(ts_code):
    if not ts_code: return None
    code, exchange = ts_code.split('.')
    return f"{exchange}{code}"

def format_date(date_str):
    if isinstance(date_str, datetime):
        return date_str.strftime('%Y-%m-%d')
    date_str = str(date_str)
    if '-' in date_str:
        return date_str
    try:
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except:
        return date_str

def call_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling {func.__name__}: {e}")
                return pd.DataFrame()
            # Exponential backoff: more aggressive for rate limits
            wait_time = 2 * (attempt + 1)
            time.sleep(wait_time)
    return pd.DataFrame()

def calculate_remaining_maturity(trade_date_str, maturity_date_str):
    if not maturity_date_str: return None
    try:
        def parse_dt(d):
            if isinstance(d, datetime): return d
            d = str(d)
            if '-' in d: return datetime.strptime(d, '%Y-%m-%d')
            return datetime.strptime(d, '%Y%m%d')
        
        trade_date = parse_dt(trade_date_str)
        maturity_date = parse_dt(maturity_date_str)
        delta = maturity_date - trade_date
        return max(0, delta.days / 365.0)
    except:
        return None

def fetch_daily_snapshot(pro, date_str):
    """
    Fetch all market data for a single date.
    Returns merged DataFrame with standardized columns.
    """
    date_formatted = date_str  # YYYYMMDD
    
    # 1. CB Daily (Market Data + Valuation)
    # contains: ts_code, trade_date, open, high, low, close, vol, amount, bond_value, bond_over_rate, cb_value, cb_over_rate
    cb_daily = call_with_retry(pro.cb_daily, trade_date=date_formatted)
    if cb_daily.empty:
        return pd.DataFrame()

    # 2. CB Share (Terms: convert_price, remain_size)
    cb_share = call_with_retry(pro.cb_share, trade_date=date_formatted)
    
    # 3. Stock Daily (Underlying Price)
    # We need mapping from CB to Stock. 
    # pro.daily returns all stocks. 
    # But we don't know which stock corresponds to which CB without cb_basic map.
    # Optimization: Fetch cb_basic once globally?
    pass

    return cb_daily, cb_share

def check_data_freshness(pro, date_str):
    """
    Check if data for the given date is fully available.
    We check a liquid bond (e.g., 113052.SH) to see if 'cb_value' is present.
    """
    # Check if it's a future date
    target_date = datetime.strptime(date_str, '%Y%m%d')
    if target_date > datetime.now():
        return True # Future dates are fine (loop won't process them anyway or user knows what they are doing)

    print(f"Checking data freshness for {date_str}...")
    # 113052.SH is a very liquid bond (Bank of China), usually updates early.
    # Alternatively, use SH110059 (Pufa) or try a few.
    test_code = '113052.SH' 
    
    df = call_with_retry(pro.cb_daily, ts_code=test_code, trade_date=date_str, fields='ts_code,cb_value')
    
    if df.empty:
        # It might be a holiday. Check calendar.
        cal = call_with_retry(pro.trade_cal, exchange='SSE', start_date=date_str, end_date=date_str)
        if not cal.empty and cal.iloc[0]['is_open'] == 0:
            print(f"  {date_str} is a holiday. Skipping freshness check.")
            return True
        
        # If open but no data, likely not ready
        print(f"  WARNING: No data found for reference bond {test_code} on {date_str}.")
        return False
        
    if pd.isna(df.iloc[0]['cb_value']):
        print(f"  WARNING: partial data found (cb_value is NaN) for {date_str}. Data might be updating.")
        return False
        
    print(f"  Data for {date_str} appears ready.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Collect CB data by Date (High Speed Update)")
    parser.add_argument("--token", type=str, required=True, help="Tushare API Token")
    parser.add_argument('--output_dir', type=str, default='csv_data', help='Output directory')
    parser.add_argument('--start_date', type=str, required=True, help='Start date YYYYMMDD')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y%m%d'), help='End date YYYYMMDD')
    args = parser.parse_args()

    # Validate Dates
    try:
        start_dt = datetime.strptime(args.start_date, '%Y%m%d')
        end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        print("Error: Dates must be in YYYYMMDD format.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ts.set_token(args.token)
    pro = ts.pro_api()

    # 0. Pre-fetch Basic Info for mapping (Stock Code, Maturity Date, Rating)
    print("Fetching Basic Info Map...")
    basic_fields = 'ts_code,stk_code,maturity_date,issue_rating,delist_date,first_conv_price,issue_size'
    basic_df = call_with_retry(pro.cb_basic, fields=basic_fields)
    
    # Create Map: ts_code -> {stk_code, maturity_date, issue_rating}
    # We can join this to daily data.
    if basic_df.empty:
        print("Failed to fetch basic info. Exiting.")
        return

    # Enhance basic_df with formatted symbol for easy lookup in the save loop
    # Create a temporary copy or drop the column after
    temp_basic = basic_df.copy()
    temp_basic['symbol'] = temp_basic['ts_code'].apply(format_symbol)
    basic_info_map = temp_basic.set_index('symbol')[['first_conv_price', 'issue_size']].to_dict('index')
    del temp_basic 

    # Data Freshness Check
    if not check_data_freshness(pro, args.end_date):
        print(f"CRITICAL: Data for end_date {args.end_date} is not fully ready on Tushare. Aborting to prevent incomplete updates.")
        return

    # Generate Date Range
    start_dt = datetime.strptime(args.start_date, '%Y%m%d')
    end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    delta = (end_dt - start_dt).days
    
    date_list = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta + 1)]
    print(f"Updating range: {args.start_date} to {args.end_date} ({len(date_list)} days)")
    
    all_daily_data = []

    for d_str in date_list:
        print(f"Processing {d_str} ...")
        
        # 1. Fetch Market Data
        # Explicitly request valuation fields
        daily_fields = 'ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate'
        daily_df = call_with_retry(pro.cb_daily, trade_date=d_str, fields=daily_fields)
        if daily_df.empty:
            print(f"  No trading data for {d_str} (Holiday?)")
            continue
            
        # 2. Fetch Share Data
        share_df = call_with_retry(pro.cb_share, trade_date=d_str)
        if not share_df.empty:
            share_df = share_df.drop_duplicates(subset=['ts_code'])
        
        stk_df = call_with_retry(pro.daily, trade_date=d_str)
        if not stk_df.empty:
             stk_df = stk_df.drop_duplicates(subset=['ts_code'])
             
        # 4. Fetch Adjustment Factor for HFQ calculation
        adj_df = call_with_retry(pro.adj_factor, trade_date=d_str)
        if not adj_df.empty:
            adj_df = adj_df.drop_duplicates(subset=['ts_code'])
        
        # --- Merge Logic ---
        # Base: daily_df
        merged = daily_df.copy()
        
        # Merge Basic (Maturity, Rating, StkCode)
        merged = pd.merge(merged, basic_df, on='ts_code', how='left')
        
        # Merge Share (Convert Price, Remain Size)
        # Note: share_df might specify share/price for specific day
        if not share_df.empty:
            # columns: ts_code, convert_price, acc_convert_ratio, remain_size
            # share_df has detailed columns. We pick relevant.
            cols_to_use = ['ts_code', 'convert_price', 'acc_convert_ratio', 'remain_size']
            # filtering cols if exist
            valid_cols = [c for c in cols_to_use if c in share_df.columns]
            share_subset = share_df[valid_cols]
            merged = pd.merge(merged, share_subset, on='ts_code', how='left')
        
        # Merge Underlying Stock
        if not stk_df.empty:
            # Merge with adj_factor first to get hfq close
            if not adj_df.empty:
                stk_df = pd.merge(stk_df, adj_df[['ts_code', 'adj_factor']], on='ts_code', how='left')
                # Calculate HFQ Close: close * adj_factor
                # Handle cases where adj_factor might be missing (default to 1.0)
                stk_df['adj_factor'] = stk_df['adj_factor'].fillna(1.0)
                stk_df['close'] = stk_df['close'] * stk_df['adj_factor']

            # stk_df: ts_code is STOCK code (e.g. 000001.SZ)
            # merged: has 'stk_code' column from basic_df
            stk_subset = stk_df[['ts_code', 'close', 'vol']].rename(columns={
                'ts_code': 'stk_code', 
                'close': 's_close', 
                'vol': 's_volume'
            })
            merged = pd.merge(merged, stk_subset, on='stk_code', how='left')
            
        # --- Fallback Fill Logic Note ---
        # We NO LONGER fill with initial values here, because it ruins old bonds 
        # that should follow forward-fill logic. 
        # Initial fill will be done at the saving stage if ffill fails.
            
        # --- Feature Engineering ---
        # 1. Volume/Amount Scaling
        if 'vol' in merged.columns: merged['vol'] = merged['vol'] * 10
        if 'amount' in merged.columns: merged['amount'] = merged['amount'] * 10000
        if 's_volume' in merged.columns: merged['s_volume'] = merged['s_volume'] * 100
        
        # 2. Qlib Factor
        merged['factor'] = 1.0
        
        # 3. Remaining Maturity
        # Apply row-wise
        merged['remaining_maturity'] = merged.apply(
            lambda row: calculate_remaining_maturity(row['trade_date'], row.get('maturity_date')), 
            axis=1
        )
        
        # 4. Standardize Columns
        merged = merged.rename(columns=COL_MAPPING)
        merged['symbol'] = merged['symbol'].apply(format_symbol)
        merged['date'] = merged['date'].apply(format_date)
        
        # Ensure features
        for f in FEATURES:
            if f not in merged.columns:
                merged[f] = None
        
        # Keep only features
        final_day_df = merged[FEATURES]
        all_daily_data.append(final_day_df)
        
        # Respect Tushare rate limits (Safe: ~150-180 requests/min)
        # We now call 4 APIs per day, so 0.8s is safer.
        time.sleep(0.8) 

    if not all_daily_data:
        print("No data fetched.")
        return

    print("Concatenating all data...")
    full_df = pd.concat(all_daily_data, ignore_index=True)
    
    # --- Batch Write ---
    print(f"Grouping by symbol ({len(full_df)} rows)...")
    grouped = full_df.groupby('symbol')
    
    # Validation Cols
    check_cols = ['cb_value', 'cb_over_rate', 'bond_value', 'bond_over_rate']
    missing_alerts = []

    count = 0
    total = len(grouped)
    for symbol, group_df in grouped:
        count += 1
        if count % 100 == 0:
            print(f"Writing {count}/{total} ...")
            
        # Check for missing values in critical columns
        # Filter: Only check rows where volume > 0 (active trading)
        if 'volume' in group_df.columns:
            active_df = group_df[group_df['volume'] > 0]
        else:
            active_df = group_df

        for col in check_cols:
            if col in active_df.columns:
                nans = active_df[col].isnull().sum()
                if nans > 0:
                    missing_alerts.append(f"{symbol}: {nans} missing in {col}")
        
        file_path = os.path.join(args.output_dir, f"{symbol}.csv")
        
        # Overwrite Logic
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            try:
                old_df = pd.read_csv(file_path)
                # Parse date to compare
                # Parse date to compare
                old_df['date_dt'] = pd.to_datetime(old_df['date'], errors='coerce')
                cutoff_dt = datetime.strptime(args.start_date, '%Y%m%d')
                
                # Filter: keep rows strictly before the new start date
                old_df = old_df[old_df['date_dt'] < cutoff_dt]
                
                # Drop temp col
                old_df = old_df.drop(columns=['date_dt'])
                
                # Append
                save_df = pd.concat([old_df, group_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_path}: {e}. Overwriting entirely.")
                save_df = group_df
        else:
            save_df = group_df
            
        # Sort and ffill
        # This ensures that old bond status is carried forward if no new info today
        save_df = save_df.sort_values('date')
        
        # We only ffill critical terms
        fill_cols = ['convert_price', 'remain_size', 'acc_convert_ratio']
        for col in fill_cols:
            if col in save_df.columns:
                save_df[col] = save_df[col].ffill()
        
        # Final Fallback: If still NaN (truly new bond), use basic_df's initial values
        # Use the pre-built basic_info_map
        if symbol in basic_info_map:
             init_vals = basic_info_map[symbol]
             
             if 'convert_price' in save_df.columns:
                 save_df['convert_price'] = save_df['convert_price'].fillna(init_vals['first_conv_price'])
             if 'remain_size' in save_df.columns:
                 save_df['remain_size'] = save_df['remain_size'].fillna(init_vals['issue_size'])
             if 'acc_convert_ratio' in save_df.columns:
                 save_df['acc_convert_ratio'] = save_df['acc_convert_ratio'].fillna(0.0)

        # Drop temporary helper columns
        drop_cols = ['first_conv_price', 'issue_size', 'stk_code', 'maturity_date', 'issue_rating', 'delist_date']
        save_df = save_df.drop(columns=[c for c in drop_cols if c in save_df.columns], errors='ignore')

        save_df.to_csv(file_path, index=False)

    print("Daily update complete.")

    if missing_alerts:
        print("\n" + "="*40)
        print("WARNING: MISSING CRITICAL DATA DETECTED")
        print("="*40)
        for alert in missing_alerts[:20]:
            print(alert)
        if len(missing_alerts) > 20:
            print(f"... and {len(missing_alerts) - 20} more.")
        print("Please check source data.")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()
