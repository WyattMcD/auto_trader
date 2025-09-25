import pandas as pd

def check_signal(data, short=10, long=50):
    """
    data: DataFrame with 'close' prices
    Returns: 'buy', 'sell', or None
    """
    data['fast_sma'] = data['close'].rolling(short).mean()
    data['slow_sma'] = data['close'].rolling(long).mean()

    if data['fast_sma'].iloc[-2] < data['slow_sma'].iloc[-2] and data['fast_sma'].iloc[-1] > data['slow_sma'].iloc[-1]:
        return "buy"
    elif data['fast_sma'].iloc[-2] > data['slow_sma'].iloc[-2] and data['fast_sma'].iloc[-1] < data['slow_sma'].iloc[-1]:
        return "sell"
    else:
        return None


