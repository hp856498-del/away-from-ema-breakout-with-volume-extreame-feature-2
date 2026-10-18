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

def is_large_breakout_candle(away_high, away_low, breakout_high, breakout_low, multiplier=1.70):
    """Whether breakout candle's full range is at least the requested multiple."""
    away_range = float(away_high) - float(away_low)
    breakout_range = float(breakout_high) - float(breakout_low)
    return away_range > 0 and breakout_range >= away_range * multiplier

def get_breakout_size_markers(
    away_high,
    away_low,
    previous_high,
    previous_low,
    breakout_high,
    breakout_low
):
    """Return star flags; prior-candle comparison applies only to 1.70x+ setups."""
    away_range = float(away_high) - float(away_low)
    previous_range = float(previous_high) - float(previous_low)
    breakout_range = float(breakout_high) - float(breakout_low)

    star = is_large_breakout_candle(
        away_high,
        away_low,
        breakout_high,
        breakout_low,
        multiplier=1.70
    )
    if not star:
        return False, False, False

    two_x_away = is_large_breakout_candle(
        away_high,
        away_low,
        breakout_high,
        breakout_low,
        multiplier=2.30
    )
    two_stars = two_x_away or (
        previous_range > 0
        and breakout_range >= previous_range * 1.60
    )

    return star, two_stars, two_x_away

def check_away_and_breakout(df_scan, df_1h, df_4h, ema_period, breakout_type, vol_mode, vol_condition, vol_lookback, date_filter_enabled, start_datetime, end_datetime):
    """Scan dataframe for EMA away + chosen Volume filter + Breakout."""
    min_required = ema_period + vol_lookback + 5
    if df_scan is None or len(df_scan) < min_required:
        return []

    df_scan['ema'] = df_scan['close'].ewm(span=ema_period, adjust=False).mean()
    bb_middle = df_scan['close'].rolling(window=20).mean()
    bb_std = df_scan['close'].rolling(window=20).std(ddof=0)
    df_scan['bb_upper'] = bb_middle + (2 * bb_std)
    df_scan['bb_lower'] = bb_middle - (2 * bb_std)
    matches = []

    start_index = max(ema_period, vol_lookback if "Previous Candles" in vol_mode else 0)

    for i in range(start_index, len(df_scan) - 1):
        row = df_scan.iloc[i]
        c_time = row['datetime_ist']

        if date_filter_enabled:
            if not (start_datetime <= c_time <= end_datetime):
                continue

        high, low, open_p, close_p, vol = row['high'], row['low'], row['open'], row['close'], row['volume']
        ema = row['ema']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        bollinger_tags = []
        if pd.notna(bb_upper) and high >= bb_upper:
            bollinger_tags.append("(BU)")
        if pd.notna(bb_lower) and low <= bb_lower:
            bollinger_tags.append("(BD)")
        bollinger_tag = " ".join(bollinger_tags)

        # 1. Check Away condition (Touch Nothing)
        is_away_above = (low > ema) and (high > ema) and (open_p > ema) and (close_p > ema)
        is_away_below = (high < ema) and (low < ema) and (open_p < ema) and (close_p < ema)

        if not (is_away_above or is_away_below):
            continue

        # 2. Volume Filter Checks
        prev_volumes = df_scan.iloc[i - vol_lookback : i]['volume'] if vol_lookback > 0 else pd.Series([])

        if vol_mode == "Away Candle Vol vs Previous Candles (Higher)":
            if not (vol > prev_volumes).all():
                continue
        elif vol_mode == "Away Candle Vol vs Previous Candles (Lower)":
            if not (vol < prev_volumes).all():
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
                previous_row = df_scan.iloc[i - 1]
                large_breakout_candle, double_large_breakout_candle, very_large_breakout_candle = get_breakout_size_markers(
                    high,
                    low,
                    previous_row["high"],
                    previous_row["low"],
                    b_high,
                    b_low
                )

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
                    'bollinger_tag': bollinger_tag,
                    'large_breakout_candle': large_breakout_candle,
                    'double_large_breakout_candle': double_large_breakout_candle,
                    'very_large_breakout_candle': very_large_breakout_candle,
                    'is_away_below': is_away_below,
                    'is_away_above': is_away_above,
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
        match['coin_name'] = symbol
        label_parts = [symbol]
        if match.get("very_large_breakout_candle"):
            label_parts.insert(0, "(2)⭐⭐")
        elif match.get("double_large_breakout_candle"):
            label_parts.insert(0, "⭐⭐")
        elif match.get("large_breakout_candle"):
            label_parts.insert(0, "⭐")
        if match['bollinger_tag']:
            label_parts.append(match['bollinger_tag'])
        label_parts.append(match['position_tag'])
        coin_label = " ".join(label_parts)
        base_symbol_text = f"{match['trend_symbol']} {coin_label}"
        is_bullish = "Bullish" in match['Type']
        is_bearish = "Bearish" in match['Type']

        if match['is_away_below'] and is_bullish:
            match['Symbol'] = f"🟢 {coin_label}"
        elif match['is_away_above'] and is_bearish:
            match['Symbol'] = f"🔴 {coin_label}"
        else:
            match['Symbol'] = base_symbol_text

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
        "Away Candle Vol vs Previous Candles (Higher)",
        "Away Candle Vol vs Previous Candles (Lower)",
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
elif "Away Candle Vol vs Previous Candles" in vol_mode:
    vol_lookback = st.sidebar.selectbox(
        "Away Candles Lookback Count (N):",
        [2, 3, 4, 5, 7],
        index=3, # Default 5
        help="Away candle ka volume pichhli N candles ke mukable comparison ke liye."
    )
elif vol_mode == "Breakout Candle Vol vs Previous Candles":
    vol_lookback = st.sidebar.selectbox(
        "Breakout Vol Higher Than Prev N Candles:",
        [2, 3, 4, 5, 7],
        index=2,
        help="Breakout candle ka volume apni pichhli N candles ke volume se zyada hona chahiye."
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

def style_breakout_symbol(row):
    """Color and bold only the requested Symbol cells in the results table."""
    styles = [""] * len(row)
    symbol_column = row.index.get_loc("Symbol")
    symbol = str(row.get("Symbol", ""))
    breakout_type = str(row.get("Type", ""))

    if "(D)" in symbol and "Bullish" in breakout_type:
        styles[symbol_column] = "color: #16803c; font-weight: bold;"
    elif "(U)" in symbol and "Bearish" in breakout_type:
        styles[symbol_column] = "color: #dc2626; font-weight: bold;"

    return styles


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

        final_df = res_df[cols]

        st.success(f"Found {len(final_df)} matching setup(s)!")
        red_mask = (
            res_df['is_away_above'].fillna(False)
            & res_df['Type'].str.contains("Bearish", na=False)
        )
        green_mask = (
            res_df['is_away_below'].fillna(False)
            & res_df['Type'].str.contains("Bullish", na=False)
        )

        def get_result_counts(group):
            coins = group.groupby('coin_name')[
                [
                    'large_breakout_candle',
                    'double_large_breakout_candle',
                    'very_large_breakout_candle'
                ]
            ].any()
            is_double_star = coins['double_large_breakout_candle']
            is_very_large = coins['very_large_breakout_candle']
            is_star = coins['large_breakout_candle']
            return (
                len(coins),
                int((is_star & ~is_double_star).sum()),
                int((is_double_star & ~is_very_large).sum()),
                int(is_very_large.sum())
            )

        red_total, red_star, red_double_star, red_very_large = get_result_counts(res_df.loc[red_mask])
        green_total, green_star, green_double_star, green_very_large = get_result_counts(res_df.loc[green_mask])
        st.markdown(
            f'<div style="color:#dc2626;font-weight:700">Red text: total {red_total} | '
            f'⭐ {red_star} | ⭐⭐ {red_double_star} | (2)⭐⭐ {red_very_large}</div>'
            f'<div style="color:#16803c;font-weight:700">Green text: total {green_total} | '
            f'⭐ {green_star} | ⭐⭐ {green_double_star} | (2)⭐⭐ {green_very_large}</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "⭐ = away candle se 1.70x+. ⭐⭐ = away candle se 1.70x+ aur usse pehle wali candle se 1.60x+. "
            "(2)⭐⭐ = away candle se 2.30x+."
        )

        # Keep the table interactive and style only the requested Symbol cells.
        styled_results = final_df.style.apply(
            style_breakout_symbol,
            axis=1
        )
        st.dataframe(styled_results, use_container_width=True)
    else:
        st.warning("Selected criteria par koi coin scan me nahi mila.")
