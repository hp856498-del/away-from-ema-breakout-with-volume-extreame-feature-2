import streamlit as st
import ccxt
import pandas as pd
from datetime import datetime, time
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# Page configuration
st.set_page_config(page_title="EMA Breakout Crypto Screener", layout="wide")
st.title("📈 Crypto EMA Away & Breakout Screener")

# Initialize Binance exchange via CCXT
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

IST = pytz.timezone('Asia/Kolkata')

@st.cache_data(ttl=300)
def get_top_binance_symbols(limit):
    """Fetch top active USDT spot trading pairs sorted by 24h volume."""
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and ticker.get('quoteVolume') is not None:
                if not any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR/', 'BULL/']):
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume']
                    })
        usdt_pairs = sorted(usdt_pairs, key=lambda x: x['volume'], reverse=True)
        return [p['symbol'] for p in usdt_pairs[:limit]]
    except Exception as e:
        st.error(f"Error fetching symbols from Binance: {e}")
        return []

def fetch_ohlcv_data(symbol, timeframe, limit=300):
    """Fetch OHLCV data from Binance and prepare dataframe."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime_ist'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(IST)
        # Drop current unclosed live candle
        df = df.iloc[:-1].reset_index(drop=True)
        return df
    except Exception:
        return None

def get_ema_at_timestamp(df, ema_period, target_time):
    """Find the EMA value of the candle corresponding to or just before target_time."""
    if df is None or len(df) < ema_period:
        return None
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()

    valid_rows = df[df['datetime_ist'] <= target_time]
    if not valid_rows.empty:
        return valid_rows.iloc[-1]['ema']
    return None

def calculate_closest_ema_distance(high, low, ema_val):
    """Calculate whether High or Low is closer to EMA and return percentage string."""
    if ema_val is None or pd.isna(ema_val) or ema_val == 0:
        return "N/A"

    diff_high = abs(high - ema_val)
    diff_low = abs(low - ema_val)

    if diff_high <= diff_low:
        pct = ((high - ema_val) / ema_val) * 100
        sign = "+" if pct >= 0 else ""
        return f"High({sign}{pct:.2f}%)"
    else:
        pct = ((low - ema_val) / ema_val) * 100
        sign = "+" if pct >= 0 else ""
        return f"Low({sign}{pct:.2f}%)"

def check_away_and_breakout(df_scan, df_1h, df_4h, ema_period, breakout_type, vol_mode, vol_condition, vol_lookback, date_filter_enabled, start_datetime, end_datetime):
    """Scan dataframe for EMA away + chosen Volume filter + Breakout."""
    min_required = ema_period + vol_lookback + 5
    if df_scan is None or len(df_scan) < min_required:
        return []

    df_scan['ema'] = df_scan['close'].ewm(span=ema_period, adjust=False).mean()
    matches = []

    start_index = max(ema_period, vol_lookback if vol_mode == "Away Candle Vol vs Previous Candles" else 0)

    for i in range(start_index, len(df_scan) - 1):
        row = df_scan.iloc[i]
        c_time = row['datetime_ist']

        if date_filter_enabled:
            if not (start_datetime <= c_time <= end_datetime):
                continue

        high, low, open_p, close_p, vol = row['high'], row['low'], row['open'], row['close'], row['volume']
        ema = row['ema']

        # 1. Check Away condition (Touch Nothing)
        is_away_above = (low > ema) and (high > ema) and (open_p > ema) and (close_p > ema)
        is_away_below = (high < ema) and (low < ema) and (open_p < ema) and (close_p < ema)

        if not (is_away_above or is_away_below):
            continue

        # 2. Away Candle Volume Lookback Filter Check
        if vol_mode == "Away Candle Vol vs Previous Candles":
            prev_volumes = df_scan.iloc[i - vol_lookback : i]['volume']
            if not (vol > prev_volumes).all():
                continue

        # 3. Check next 1 to 3 candles for breakout
        for offset in range(1, 4):
            breakout_idx = i + offset
            if breakout_idx >= len(df_scan):
                break

            b_row = df_scan.iloc[breakout_idx]
            b_high, b_low, b_close, b_vol = b_row['high'], b_row['low'], b_row['close'], b_row['volume']

            # Breakout Vol vs Away Vol Filter Check
            if vol_mode == "Breakout Vol vs Away Vol":
                if vol_condition == "Higher Volume (Breakout Vol > Away Vol)" and b_vol <= vol:
                    continue
                elif vol_condition == "Lower Volume (Breakout Vol < Away Vol)" and b_vol >= vol:
                    continue

            # Breakout Vol vs Previous Candles Filter Check
            if vol_mode == "Breakout Candle Vol vs Previous Candles":
                if breakout_idx - vol_lookback < 0:
                    continue
                b_prev_volumes = df_scan.iloc[breakout_idx - vol_lookback : breakout_idx]['volume']
                if not (b_vol > b_prev_volumes).all():
                    continue

            breakout_detected = False
            direction = ""

            # Bullish Breakout Check
            if breakout_type == "High/Low Breakout" and b_high > high:
                breakout_detected = True
                direction = "Bullish High Break"
            elif breakout_type == "Close Breakout" and b_close > high:
                breakout_detected = True
                direction = "Bullish Close Break"

            # Bearish Breakout Check
            if not breakout_detected:
                if breakout_type == "High/Low Breakout" and b_low < low:
                    breakout_detected = True
                    direction = "Bearish Low Break"
                elif breakout_type == "Close Breakout" and b_close < low:
                    breakout_detected = True
                    direction = "Bearish Close Break"

            if breakout_detected:
                position_tag = "(U)" if is_away_above else "(D)"

                ema_1h_51 = get_ema_at_timestamp(df_1h, 51, c_time)
                ema_4h_51 = get_ema_at_timestamp(df_4h, 51, c_time)
                ema_4h_101 = get_ema_at_timestamp(df_4h, 101, c_time)

                trend_symbol = "🟢" if (ema_4h_51 is not None and ema_4h_101 is not None and ema_4h_51 > ema_4h_101) else "🔴"

                dist_1h_51 = calculate_closest_ema_distance(high, low, ema_1h_51)
                dist_4h_51 = calculate_closest_ema_distance(high, low, ema_4h_51)
                dist_4h_101 = calculate_closest_ema_distance(high, low, ema_4h_101)

                matches.append({
                    'trend_symbol': trend_symbol,
                    'position_tag': position_tag,
                    'Away Candle Time (IST)': c_time.strftime('%Y-%m-%d %H:%M'),
                    'Away Price (High/Low)': f"{high} / {low}",
                    'Away Vol': f"{vol:,.2f}",
                    '1h 51 EMA Dist': dist_1h_51,
                    '4h 51 EMA Dist': dist_4h_51,
                    '4h 101 EMA Dist': dist_4h_101,
                    'Breakout Candle Time (IST)': b_row['datetime_ist'].strftime('%Y-%m-%d %H:%M'),
                    'Breakout Offset': f"{offset} candle(s) after",
                    'Breakout Close': b_close,
                    'Breakout Vol': f"{b_vol:,.2f}",
                    'Type': direction
                })
                break

    return matches

def process_single_symbol(symbol, tf_option, ema_option, breakout_option, vol_mode, vol_condition, vol_lookback, enable_date, start_datetime, end_datetime):
    """Helper function to process a single symbol for multithreading."""
    df_scan = fetch_ohlcv_data(symbol, tf_option)
    if df_scan is None:
        return []

    df_1h = fetch_ohlcv_data(symbol, "1h") if tf_option != "1h" else df_scan
    df_4h = fetch_ohlcv_data(symbol, "4h") if tf_option != "4h" else df_scan

    matches = check_away_and_breakout(
        df_scan, df_1h, df_4h, ema_option, breakout_option, vol_mode, vol_condition, vol_lookback, enable_date, start_datetime, end_datetime
    )

    symbol_results = []
    for match in matches:
        match['Symbol'] = f"{match['trend_symbol']} {symbol} {match['position_tag']}"
        symbol_results.append(match)

    return symbol_results

# Sidebar UI Options
st.sidebar.header("⚙️ Screener Controls")

# 1. EMA Period Selection
ema_option = st.sidebar.selectbox("Select EMA Period:", [2, 3, 4, 5, 6], index=1)

# 2. Number of Top Binance Coins
top_coins_count = st.sidebar.selectbox("Select Top Coins Count (by Volume):", [50, 100, 200, 400, 500, 600, 700], index=1)

# 3. Timeframe Selection
tf_option = st.sidebar.selectbox("Select Timeframe:", ["15m", "30m", "1h", "2h", "4h", "1d"], index=0)

# 4. Breakout Type
breakout_option = st.sidebar.selectbox(
    "Select Breakout Type:", 
    ["Close Breakout", "High/Low Breakout"], 
    help="Close Breakout requires candle to close beyond away candle high/low. High/Low Breakout checks if wick/touch breaks high/low."
)

# 5. Volume Filter Options
st.sidebar.subheader("📊 Volume Filter Options")
vol_mode = st.sidebar.radio(
    "Select Volume Filter Type:",
    [
        "Breakout Vol vs Away Vol", 
        "Away Candle Vol vs Previous Candles",
        "Breakout Candle Vol vs Previous Candles"
    ]
)

vol_condition = None
vol_lookback = 0

if vol_mode == "Breakout Vol vs Away Vol":
    vol_condition = st.sidebar.selectbox(
        "Select Condition:",
        [
            "Higher Volume (Breakout Vol > Away Vol)",
            "Lower Volume (Breakout Vol < Away Vol)"
        ]
    )
elif vol_mode == "Away Candle Vol vs Previous Candles":
    vol_lookback = st.sidebar.selectbox(
        "Away Vol Higher Than Prev N Candles:",
        [2, 3, 4, 5, 7],
        index=1,
        help="Away candle ka volume picchli N candles ke volume se zyada hona chahiye."
    )
elif vol_mode == "Breakout Candle Vol vs Previous Candles":
    vol_lookback = st.sidebar.selectbox(
        "Breakout Vol Higher Than Prev N Candles:",
        [2, 3, 4, 5, 7],
        index=2,
        help="Breakout candle ka volume apni picchli N candles ke volume se zyada hona chahiye."
    )

# 6. Date & Time Filter (IST)
st.sidebar.subheader("📅 Date & Time Filter (IST)")
enable_date = st.sidebar.checkbox("Enable Specific Date & Time Filter")

start_datetime, end_datetime = None, None

if enable_date:
    selected_date = st.sidebar.date_input("Select Date", datetime.now(IST))
    enable_time = st.sidebar.checkbox("Specify Time Range")

    if enable_time:
        start_time_val = st.sidebar.time_input("Start Time (IST)", time(0, 0))
        end_time_val = st.sidebar.time_input("End Time (IST)", time(23, 59))
        start_datetime = IST.localize(datetime.combine(selected_date, start_time_val))
        end_datetime = IST.localize(datetime.combine(selected_date, end_time_val))
    else:
        start_datetime = IST.localize(datetime.combine(selected_date, time(0, 0)))
        end_datetime = IST.localize(datetime.combine(selected_date, time(23, 59)))

# Run Screener Button
if st.button("🚀 Start Scanning"):
    st.info(f"Fetching Binance Top {top_coins_count} pairs and scanning...")
    symbols = get_top_binance_symbols(top_coins_count)

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_symbols = len(symbols)
    completed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(
                process_single_symbol, symbol, tf_option, ema_option, breakout_option, vol_mode, vol_condition, vol_lookback, enable_date, start_datetime, end_datetime
            ): symbol for symbol in symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            status_text.text(f"Scanning ({completed}/{total_symbols}): {symbol}")
            progress_bar.progress(completed / total_symbols)

            try:
                res = future.result()
                if res:
                    results.extend(res)
            except Exception:
                pass

    status_text.text("Scanning Completed!")
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        cols = [
            'Symbol', 'Type', 'Away Candle Time (IST)', 'Away Price (High/Low)', 'Away Vol', 
            '1h 51 EMA Dist', '4h 51 EMA Dist', '4h 101 EMA Dist',
            'Breakout Candle Time (IST)', 'Breakout Offset', 'Breakout Close', 'Breakout Vol'
        ]
        res_df = res_df[cols]

        def style_symbol(row):
            symbol = row['Symbol']
            breakout_type = row['Type']
            if '(D)' in symbol and breakout_type.startswith('Bullish'):
                style = 'color: green; font-weight: bold;'
            elif '(U)' in symbol and breakout_type.startswith('Bearish'):
                style = 'color: red; font-weight: bold;'
            else:
                style = ''
            return [style if column == 'Symbol' else '' for column in row.index]

        st.success(f"Found {len(res_df)} matching setup(s)!")
        st.dataframe(res_df.style.apply(style_symbol, axis=1), use_container_width=True)
    else:
        st.warning("Selected criteria par koi coin scan me nahi mila.")