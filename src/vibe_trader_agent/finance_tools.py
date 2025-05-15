"""Finance tools for the agent."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool

# Financial parameters
#    'minimal': 90,        # 3-4 months, for basic indicators
#    'standard': 252,      # 1 year, for robust analysis
#    'comprehensive': 504, # 2 years, for trend analysis
#    'advanced': 1260      # 5 years, for cycle analysis
TIME_WINDOW = 504


def get_stock_historical_data(
    tickers: List[str],
    time_window: Union[int, str] = 30,
    window_type: str = 'days',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, List[float]]:
    """Fetch historical closing prices data for multiple stock tickers.
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        time_window (Union[int, str]): Time window size (default: 30)
        window_type (str): Type of time window ('days', 'months', 'years') (default: 'days')
        start_date (Optional[str]): Start date in YYYY-MM-DD format (overrides time_window)
        end_date (Optional[str]): End date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict: Dictionary with tickers as keys and historical data as values
    """
    result = {}

    if isinstance(time_window, str):
        time_window = int(time_window)
    
    # Set the date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        if window_type == 'days':
            start_date = (datetime.now() - timedelta(days=time_window)).strftime('%Y-%m-%d')
        elif window_type == 'months':
            # Approximate months (30 days per month)
            start_date = (datetime.now() - timedelta(days=time_window * 30)).strftime('%Y-%m-%d')
        elif window_type == 'years':
            start_date = (datetime.now() - timedelta(days=time_window * 365)).strftime('%Y-%m-%d')
        else:
            raise ValueError("Invalid window_type. Use 'days', 'months', or 'years'")
    
    for ticker in tickers:
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                result[ticker] = None
                continue

            result[ticker] = df['Close'].tolist()

        except Exception:
            result[ticker] = None
        
    return result


def calculate_financial_indicators(
    closing_prices: List[float],
    ticker: str
) -> Dict[str, Any]:
    """Calculate financial indicators for forecasting based on closing prices.
    
    Args:
        closing_prices (List[float]): List of closing prices
        ticker (str): Stock ticker symbol
    
    Returns:
        Dict[str, Any]: Dictionary containing various financial indicators
    """
    # Convert to pandas Series for easier calculations
    prices = pd.Series(closing_prices)
    
    indicators = {}
    
    try:
        # 1. Moving Averages
        indicators['SMA_20'] = round(float(prices.rolling(window=20).mean().iloc[-1]), 2)
        indicators['SMA_50'] = round(float(prices.rolling(window=50).mean().iloc[-1]), 2)
        indicators['EMA_20'] = round(float(prices.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        indicators['EMA_50'] = round(float(prices.ewm(span=50, adjust=False).mean().iloc[-1]), 2)
        
        # 2. RSI (Relative Strength Index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = round(float(100 - (100 / (1 + rs)).iloc[-1]), 2)
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        indicators['MACD'] = float(macd_line.iloc[-1])
        indicators['MACD_Signal'] = float(signal_line.iloc[-1])
        indicators['MACD_Histogram'] = float((macd_line - signal_line).iloc[-1])
        
        # 4. Bollinger Bands
        # sma_20 = prices.rolling(window=20).mean()
        # std_20 = prices.rolling(window=20).std()
        # indicators['Bollinger_Upper'] = float((sma_20 + 2 * std_20).iloc[-1])
        # indicators['Bollinger_Lower'] = float((sma_20 - 2 * std_20).iloc[-1])
        # indicators['Bollinger_Middle'] = float(sma_20.iloc[-1])
        # indicators['Bollinger_Width'] = float(((sma_20 + 2 * std_20) - (sma_20 - 2 * std_20)).iloc[-1])
        
        # 5. Volatility Measures
        daily_returns = prices.pct_change()
        volatility_20d = daily_returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        volatility_50d = daily_returns.rolling(window=50).std() * np.sqrt(252)
        indicators['Volatility_20d'] = round(float(volatility_20d.iloc[-1]), 2)  # Current volatility only
        indicators['Volatility_50d'] = round(float(volatility_50d.iloc[-1]), 2)  # Current volatility only
        indicators['Current_Volatility'] = indicators['Volatility_20d']  # Use last value
        
        # 6. Rate of Change (ROC)
        # indicators['ROC_10d'] = float((prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10] * 100)
        indicators['ROC_20d'] = round(float((prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] * 100), 2)
        
        # 7. Momentum
        # indicators['Momentum_10d'] = float(prices.iloc[-1] - prices.iloc[-10])
        indicators['Momentum_20d'] = round(float(prices.iloc[-1] - prices.iloc[-20]), 2)
        
        # 8. Statistical Measures
        indicators['Mean_Price'] = round(float(prices.mean()), 2)
        indicators['Std_Dev'] = round(float(prices.std()), 2)
        indicators['Coefficient_Variation'] = round(float(indicators['Std_Dev'] / indicators['Mean_Price']), 2)
        
        # 9. Support and Resistance Levels
        rolling_max = prices.rolling(window=20).max()
        rolling_min = prices.rolling(window=20).min()
        indicators['Resistance_20d'] = round(float(rolling_max.iloc[-1]), 2)
        indicators['Support_20d'] = round(float(rolling_min.iloc[-1]), 2)
        
        # 10. Average Daily Returns
        # indicators['Avg_Daily_Return'] = float(daily_returns.mean() * 100)
        # indicators['Avg_Daily_Return_20d'] = float(daily_returns.tail(20).mean() * 100)
        
        # 11. Maximum Drawdown
        rolling_max = prices.expanding().max()
        drawdown = prices / rolling_max - 1
        indicators['Max_Drawdown'] = round(float(drawdown.min() * 100), 2)
        
        # 12. Golden/Death Cross Indicators
        indicators['Golden_Cross'] = bool(indicators['SMA_20'] > indicators['SMA_50'])
        indicators['Death_Cross'] = bool(indicators['SMA_20'] < indicators['SMA_50'])
        
        # 13. Price Position - convert to float
        indicators['Price_vs_SMA20'] = round(float((prices.iloc[-1] - indicators['SMA_20']) / indicators['SMA_20'] * 100), 2)
        indicators['Price_vs_SMA50'] = round(float((prices.iloc[-1] - indicators['SMA_50']) / indicators['SMA_50'] * 100), 2)
        
        # 14. Stochastic Oscillator
        low_14 = prices.rolling(window=14).min()
        high_14 = prices.rolling(window=14).max()
        stochastic_k = 100 * (prices - low_14) / (high_14 - low_14)
        indicators['Stochastic_K_Current'] = round(float(stochastic_k.iloc[-1]), 2)  # Current value only
        indicators['Stochastic_D'] = round(float(stochastic_k.rolling(window=3).mean().iloc[-1]), 2)
        
        # 15. Current Market Position
        # indicators['Current_Price'] = float(prices.iloc[-1], 2)
        # indicators['Price_Change_1d'] = float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100, 2)
        # indicators['Price_Change_5d'] = float((prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] * 100, 2)
        indicators['Price_Change_20d'] = round(float((prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] * 100), 2)
        
        # 16. Trend Strength
        adx_value = calculate_adx(prices)
        indicators['ADX'] = round(float(adx_value), 2) if adx_value is not None else None
        
    except Exception as e:
        indicators['Error'] = str(e)
    
    return indicators


def calculate_adx(prices: pd.Series, period: int = 14) -> float:
    """Calculate Average Directional Index (ADX) for trend strength."""
    try:
        high = prices
        low = prices
        
        plus_dm = high.diff()
        minus_dm = low.diff()

        # Convert to numeric to fix type issues
        plus_dm = pd.to_numeric(plus_dm, errors='coerce')
        minus_dm = pd.to_numeric(minus_dm, errors='coerce')
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = prices.diff().abs()
        tr = tr1.rolling(window=period).sum()
        
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr)
        minus_di = 100 * (abs(minus_dm.rolling(window=period).sum()) / tr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1]
    except (IndexError, ZeroDivisionError, ValueError):
        return None


@tool
def calculate_financial_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Calculate comprehensive financial indicators for given tickers.
    
    Returns key metrics including ratios, historical performance, and risk indicators for portfolio analysis.

    Args:
        tickers (List[str]): List of ticker symbols
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping each ticker to its financial indicators
    """
    indicators_dict = {}
    
    for ticker in tickers:

        closing_prices = get_stock_historical_data([ticker], time_window=TIME_WINDOW).get(ticker, [])

        if closing_prices:
            indicators_dict[ticker] = calculate_financial_indicators(closing_prices, ticker)
        else:
            indicators_dict[ticker] = {'Error': 'No data available. Is {ticker} correct?'}
    
    return indicators_dict


if __name__ == "__main__":
    tickers = ['AAPL', 'PLTR']
    data = get_stock_historical_data(tickers)

    indicators = calculate_financial_metrics(tickers)    
    
