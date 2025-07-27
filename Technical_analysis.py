import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

class TechnicalAnalysis:
    def __init__(self):
        pass

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        try:
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            hist = macd - signal_line
            return {
                'macd': macd,
                'signal': signal_line,
                'hist': hist
            }
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return {}

    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict:
        try:
            emas = {}
            for period in periods:
                emas[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            return emas
        except Exception as e:
            print(f"Error calculating EMA: {e}")
            return {}

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series(dtype=float)

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict:
        try:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return {
                'upper': upper,
                'middle': sma,
                'lower': lower
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return {}

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return pd.Series(dtype=float)

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
            minus_di = abs(100 * (minus_dm.rolling(window=period).sum() / atr))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=period).mean()
            return adx
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return pd.Series(dtype=float)

    def calculate_stoch_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            rsi = self.calculate_rsi(df, period)
            min_rsi = rsi.rolling(window=period).min()
            max_rsi = rsi.rolling(window=period).max()
            stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
            return stoch_rsi
        except Exception as e:
            print(f"Error calculating Stochastic RSI: {e}")
            return pd.Series(dtype=float)

    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        try:
            atr = self.calculate_atr(df, period)
            hl2 = (df['high'] + df['low']) / 2
            final_upperband = hl2 + (multiplier * atr)
            final_lowerband = hl2 - (multiplier * atr)

            in_uptrend = pd.Series(True, index=df.index)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > final_upperband.iloc[i - 1]:
                    in_uptrend.iloc[i] = True
                elif df['close'].iloc[i] < final_lowerband.iloc[i - 1]:
                    in_uptrend.iloc[i] = False
                else:
                    in_uptrend.iloc[i] = in_uptrend.iloc[i - 1]
                    if in_uptrend.iloc[i] and final_lowerband.iloc[i] < final_lowerband.iloc[i - 1]:
                        final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]
                    if not in_uptrend.iloc[i] and final_upperband.iloc[i] > final_upperband.iloc[i - 1]:
                        final_upperband.iloc[i] = final_upperband.iloc[i - 1]
            supertrend = pd.DataFrame({
                'supertrend': (final_upperband + final_lowerband) / 2,
                'in_uptrend': in_uptrend
            }, index=df.index)
            return supertrend
        except Exception as e:
            print(f"Error calculating SuperTrend: {e}")
            return pd.DataFrame({'supertrend': pd.Series(dtype=float), 'in_uptrend': pd.Series(dtype=bool)})

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap
        except Exception as e:
            print(f"Error calculating VWAP: {e}")
            return pd.Series(dtype=float)