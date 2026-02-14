
import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime
import argparse
import warnings

# Suppress Tushare/Pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global mapping for renaming columns
COL_MAPPING = {
    'ts_code': 'symbol',
    'trade_date': 'date',
    'vol': 'volume'
}

# Features to keep
FEATURES = [
    'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'factor',
    's_close', 's_volume',
    'cb_value', 'cb_over_rate', 'bond_value', 'bond_over_rate',
    'convert_price', 'acc_convert_ratio', 'remain_size', 'remaining_maturity', 'issue_rating'
]

def get_tushare_api(token):
    if not token:
        token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        raise ValueError("Tushare token is required. Please provide it via --token or TUSHARE_TOKEN env var.")
    return ts.pro_api(token)

def format_symbol(ts_code):
    """
    Convert 113503.SH -> SH113503
    """
    code, exchange = ts_code.split('.')
    return f"{exchange}{code}"

def format_date(date_str):
    """
    Convert 20200101 -> 2020-01-01
    """
    if isinstance(date_str, datetime):
        return date_str.strftime('%Y-%m-%d')
    date_str = str(date_str)
    # Handle YYYY-MM-DD
    if '-' in date_str:
        return date_str
    # Handle YYYYMMDD
    try:
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except:
        return date_str # Fallback

def calculate_remaining_maturity(trade_date_str, maturity_date_str):
    """
    Calculate remaining maturity in years.
    Returns None if maturity_date is missing.
    """
    if not maturity_date_str:
        return None
    try:
        # Helper to parse multiple formats
        def parse_dt(d):
            if isinstance(d, datetime): return d
            d = str(d)
            if '-' in d: return datetime.strptime(d, '%Y-%m-%d')
            return datetime.strptime(d, '%Y%m%d')

        trade_date = parse_dt(trade_date_str)
        maturity_date = parse_dt(maturity_date_str)
        delta = maturity_date - trade_date
        return max(0, delta.days / 365.0)
    except Exception as e:
        print(f"Error calculating maturity: {e}")
        return None

def call_with_retry(func, *args, max_retries=3, **kwargs):
    """
    Retry API call 3 times before giving up.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e # invalid
            # print(f"    Retry {attempt+1}/{max_retries} due to: {e}")
            time.sleep(1 * (attempt + 1))
    return None

def log_failure(ts_code, msg):
    """
    Log failure to file for easier post-mortem.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("download_errors.log", "a") as f:
        f.write(f"{timestamp} - {ts_code} - {msg}\n")

def fetch_all_cb_basic(pro):
    print("Fetching CB list...")
    # Get basic info including maturity date and issue_rating
    # Using retry for completeness, though basic usually works once
    df = call_with_retry(pro.cb_basic, fields='ts_code,symbol,bond_short_name,maturity_date,list_date,delist_date,stk_code,issue_rating,first_conv_price,issue_size')
    return df

def fetch_cb_share(pro, ts_code):
    """
    Fetch CB share/size change info for remaining size interpolation.
    """
    try:
        # fetch enough history
        df = call_with_retry(pro.cb_share, ts_code=ts_code)
        if df is None or df.empty:
            return pd.DataFrame()
        # Ensure date format

        # Ensure date format
        df['end_date'] = pd.to_datetime(df['end_date'])
        df = df.sort_values(by='end_date')
        return df[['end_date', 'remain_size', 'convert_price', 'acc_convert_ratio']]
    except Exception as e:
        print(f"  Warning: failed to fetch cb_share for {ts_code}: {e}")
        return pd.DataFrame()

def fetch_stock_daily(ts_code, start_date=None):
    try:
        # Use ts.pro_bar to get adjusted close (hfq)
        # Note: ts.pro_bar requires ts.set_token to be effective if not explicitly passed data?
        # Actually ts.pro_bar uses the global token or pro_api passed? 
        # ts.pro_bar input signature doesn't take 'pro' object, it uses 'api' or global setting.
        # We will ensure ts.set_token is called in main.
        
        # Use hfq (Posterior Adjustment) for continuous price series
        # ts.pro_bar handles splitting dates if range is too large, but for single stock usually fine.
        df = call_with_retry(ts.pro_bar, ts_code=ts_code, adj='hfq', start_date=start_date)
        
        if df is None or df.empty:
             return pd.DataFrame()
             
        # Select and rename columns immediately to avoid conflict
        # trade_date, close -> s_close, vol -> s_volume
        df = df[['trade_date', 'close', 'vol']]
        df = df.rename(columns={'close': 's_close', 'vol': 's_volume'})
        
        return df
    except Exception as e:
        print(f"  Warning: failed to fetch stock daily for {ts_code}: {e}")
        return pd.DataFrame()

def fetch_daily_data(pro, ts_code, maturity_date=None, start_date=None, end_date=None):
    """
    Fetch daily transaction data for a convertible bond.
    Tushare limit: 2000 rows per request. For daily data, 2000 rows > 8 years.
    Most CBs live less than 6 years. So one request is usually enough.
    We will add a loop just in case or keep it simple if efficient.
    Actually, cb_daily allows startDate/endDate. 
    Let's try fetching all without dates first, and check length. 
    If hits 2000 limit, we need logic. But usually fetching by specific ts_code returns all.
    Wait, Tushare doc says "Single max 2000". If a CB has > 2000 days, we miss data.
    Safest way: fetch in chunks or use a large enough loop.
    Given typical CB life span < 6 years, 2000 records (approx 8 years) is usually enough.
    BUT, to be robust, we can simply fetch.
    """
    

    
    # Define fields including optional ones
    fields = 'ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate'

    # Try fetching all
    s_date = None
    e_date = None
    if start_date:
        s_date = start_date.replace('-', '')
    if end_date:
        e_date = end_date.replace('-', '')

    if s_date and e_date:
        df = call_with_retry(pro.cb_daily, ts_code=ts_code, start_date=s_date, end_date=e_date, fields=fields)
    elif s_date:
        df = call_with_retry(pro.cb_daily, ts_code=ts_code, start_date=s_date, fields=fields)
    else:
        df = call_with_retry(pro.cb_daily, ts_code=ts_code, fields=fields)
    
    # Check if we hit the limit (this is a simple heuristic, if 2000 rows returned, might have more)
    # But for CB, >2000 days is rare. 
    # Let's assume one shot is enough for 99% cases. 
    # If user has very old CBs, we might need more complex logic.
    # For now, simplistic approach.
    
    if df.empty:
        return df

    # Calculate remaining maturity
    # Note: df['trade_date'] is YYYYMMDD
    if maturity_date:
        df['remaining_maturity'] = df['trade_date'].apply(lambda x: calculate_remaining_maturity(x, maturity_date))
    else:
        df['remaining_maturity'] = None

    return df

def main():
    parser = argparse.ArgumentParser(description="Collect CB Data")
    parser.add_argument("--token", type=str, help="Tushare API Token")
    parser.add_argument('--output_dir', type=str, default='csv_data', help='Output directory for CSV files')
    parser.add_argument('--test', action='store_true', help='Test mode: run only for 3 items')
    parser.add_argument("--mock", action="store_true", help="Mock mode (generate sample data without API)")
    parser.add_argument('--start_date', type=str, default='20180101', help='Start date YYYYMMDD. Default: 20180101')
    parser.add_argument('--end_date', type=str, default=None, help='End date for data collection YYYYMMDD')
    args = parser.parse_args()

    if args.mock:
        print("Mock mode: Generating sample data...")
        os.makedirs(args.output_dir, exist_ok=True)
        # Create a sample DataFrame
        mock_data = {
            'symbol': ['SH113001'] * 5,
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'open': [100.0, 101.0, 100.5, 102.0, 101.5],
            'high': [102.0, 103.0, 101.0, 103.0, 102.5],
            'low': [99.0, 100.5, 99.5, 101.0, 100.0],
            'close': [101.0, 102.0, 100.8, 102.5, 101.8],
            'volume': [1000, 1200, 900, 1500, 1100],
            'amount': [101000, 122400, 90720, 153750, 111980],
            'bond_value': [90.0, 90.1, 90.2, 90.3, 90.4],
            'bond_over_rate': [12.2, 13.2, 11.7, 13.5, 12.6],
            'cb_value': [110.0, 112.0, 111.0, 113.0, 112.5],
            'cb_over_rate': [-8.1, -8.9, -9.1, -9.2, -9.5],
            'remaining_maturity': [5.5, 5.497, 5.494, 5.491, 5.488]
        }
        df = pd.DataFrame(mock_data)
        save_path = os.path.join(args.output_dir, "SH113001.csv")
        df.to_csv(save_path, index=False)
        print(f"Mock data saved to {save_path}")
        return

    # 1. Init API
    # Init Tushare
    ts.set_token(args.token) # Important for ts.pro_bar
    try:
        pro = ts.pro_api(args.token)
    except Exception as e:
        print(f"Error initializing API: {e}")
        return

    # 2. Get Basic Info map (for maturity dates)
    try:
        basic_df = fetch_all_cb_basic(pro)
    except Exception as e:
        print(f"Failed to fetch basic info: {e}")
        return

    print(f"Found {len(basic_df)} convertibles.")
    
    # Filter by start_date if provided
    if args.start_date:
        print(f"Filtering bonds delisted before {args.start_date}...")
        # Ensure delist_date is treated as string for comparison, fast filter
        # Format user start_date to YYYYMMDD for string comparison
        s_date_str = args.start_date.replace('-', '')
        
        # Keep if delist_date is null (still active) or delist_date >= start_date
        # basic_df['delist_date'] might be None or NaN
        original_count = len(basic_df)
        basic_df = basic_df[basic_df['delist_date'].fillna('99999999') >= s_date_str]
        basic_df = basic_df.reset_index(drop=True)
        print(f"Filtered: {original_count} -> {len(basic_df)} bonds remaining.")

    if args.test:
        print("Test mode enabled. limiting to 3 items.")
        basic_df = basic_df.head(3)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Loop and Fetch
    # To respect rate limit (200 requests/min -> ~3 requests/sec), we sleep slightly.
    # Safe bet: 0.3s sleep per request.
    
    for idx, row in basic_df.iterrows():
        ts_code = row['ts_code']
        maturity_date = row['maturity_date']
        issue_rating = row['issue_rating']
        init_conv_price = row.get('first_conv_price')
        init_issue_size = row.get('issue_size')
        
        if not ts_code: 
            continue

        if not ts_code: 
            continue

        # Check if file exists (Resume Logic & Update Logic)
        run_symbol = format_symbol(ts_code)
        output_file = os.path.join(args.output_dir, f"{run_symbol}.csv")
        file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 100
        
        # Resume Logic: Skip if file already exists
        if file_exists:
            print(f"[{idx+1}/{len(basic_df)}] Skipping {ts_code}, file exists.")
            continue
        
        current_start_date = args.start_date
        print(f"[{idx+1}/{len(basic_df)}] Fetching {ts_code} ...")
        
        try:
            # 3.1 Fetch CB Daily
            df = fetch_daily_data(pro, ts_code, maturity_date, start_date=current_start_date, end_date=args.end_date)
            
            if df.empty:
                print(f"  No data for {ts_code}")
                time.sleep(0.3) 
                continue

            # 3.2 Fetch Remaining Size (cb_share) and Merge
            # Logic: Merge on date, ffill.
            share_df = fetch_cb_share(pro, ts_code)
            if not share_df.empty:
                # Ensure daily date is datetime
                df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
                
                # Ensure share_df 'end_date' is datetime
                share_df['end_date'] = pd.to_datetime(share_df['end_date'])

                # Merge asof logic: 
                # We need to assign 'remain_size' to daily records based on the latest available share record
                # sort both
                df = df.sort_values('trade_date_dt')
                share_df = share_df.sort_values('end_date')
                
                # use merge_asof
                df = pd.merge_asof(
                    df, 
                    share_df, 
                    left_on='trade_date_dt', 
                    right_on='end_date', 
                    direction='backward'
                )
                
                # Fix: Backfill initial missing values using ACTUAL initial values
                # 1. convert_price -> first_conv_price
                if 'convert_price' in df.columns and init_conv_price is not None:
                    df['convert_price'] = df['convert_price'].fillna(init_conv_price)
                
                # 2. remain_size -> issue_size
                if 'remain_size' in df.columns and init_issue_size is not None:
                    df['remain_size'] = df['remain_size'].fillna(init_issue_size)
                    
                # 3. acc_convert_ratio -> 0
                if 'acc_convert_ratio' in df.columns:
                    df['acc_convert_ratio'] = df['acc_convert_ratio'].fillna(0.0)
            else:
                 log_failure(ts_code, "Missing cb_share data (remain_size)")
                 # Fallback to initial if share data completely missing
                 df['remain_size'] = init_issue_size
                 df['convert_price'] = init_conv_price
                 df['acc_convert_ratio'] = 0.0

            # 3.4 Static Features
            df['issue_rating'] = issue_rating
            df['factor'] = 1.0

            # 3.3 Fetch Stock Vol & Price (s_close)
            stk_code = row['stk_code']
            if stk_code:
                stk_df = fetch_stock_daily(stk_code, start_date=args.start_date)
                if not stk_df.empty:
                    # Rename is already done in fetch_stock_daily: s_close, s_volume
                    # Merge
                    # stock_daily from pro_bar has date in YYYYMMDD string usually? 
                    # ts.pro_bar returns 'trade_date' as string YYYYMMDD by default.
                    
                    # Ensure trade_date coversions if needed?
                    # basic df['trade_date'] is likely object/string YYYYMMDD.
                    # Let's assume compat.
                    df = pd.merge(df, stk_df, on='trade_date', how='left')
                else:
                    log_failure(ts_code, "Missing stock data (s_close/s_volume)")
                    df['s_volume'] = None
                    df['s_close'] = None
            else:
                df['s_volume'] = None
                df['s_close'] = None

            # 4. Format Data
            # Unit Conversion for Standardization
            # amount: 10k Yuan -> Yuan
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 10000
                
            # volume: Hands (10 bonds) -> Units (1 bond)
            if 'volume' in df.columns: 
                # Rename columns happens below, but 'vol' logic was confusing in previous edits.
                # Actually, df current col is 'vol' from cb_daily. 
                pass

            # Rename columns (vol -> volume)
            df = df.rename(columns=COL_MAPPING) 
            
            # Now apply conversion on 'volume'
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * 10 
            
            # s_volume: Hands (100 shares) -> Shares
            if 's_volume' in df.columns:
                df['s_volume'] = df['s_volume'] * 100
            
            # Format Symbol
            df['symbol'] = df['symbol'].apply(lambda x: format_symbol(ts_code) if x else format_symbol(ts_code))
            
            # Format Date
            # Format Date
            df['date'] = df['date'].apply(format_date)
            
            # 5. Save
            # Reorder columns
            # Ensure all FEATURES are present
            for col in FEATURES:
                if col not in df.columns:
                    df[col] = None
            
            save_df = df[FEATURES].copy()
            save_df = save_df.sort_values(by='date')
            
            # Write mode
            save_df.to_csv(output_file, index=False)
            print(f"  Saved {len(save_df)} rows to {output_file}")
            
            time.sleep(0.3) # Rate limit logic passed to here

        except Exception as e:
            print(f"  Failed to fetch {ts_code}: {e}")

        # Rate limit sleep
        time.sleep(0.3)

if __name__ == "__main__":
    main()
