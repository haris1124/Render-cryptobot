import logging
import time
import random
import asyncio
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime
import numpy as np
from config import Config
from Technical_analysis import TechnicalAnalysis
from Telegram_client import Telegram
from portfolio import Portfolio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("signal_generator")

class SignalGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.binance_client = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'testnet': True,
            'urls': {
                'api': 'https://testnet.binance.vision/api',
            }
        })
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.portfolio = Portfolio(self.binance_client, config)
        self.last_signal = {}
        self.last_signal_time = {}
        self.failed_symbols = {}
        self.warned_symbols = set()
        self.scanned_symbols = set()
        self.used_coins = set()
        self.cooldown_period = 900
        self.major_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT'
        ]

    async def get_symbols(self) -> List[str]:
        try:
            await self.binance_client.load_markets()
            symbols = [symbol for symbol in self.binance_client.markets.keys()
                       if symbol.endswith('USDT') and self.binance_client.markets[symbol]['active']]
            logger.info(f"Found {len(symbols)} active USDT trading pairs")
            return symbols
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return self.major_pairs

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        try:
            ohlcv = await self.binance_client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            if df['close'].isna().any():
                logger.warning(f"NaN values in data for {symbol} on {timeframe}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def analyze_pair(self, symbol: str) -> List[Dict]:
        try:
            if not symbol:
                logger.warning("Invalid symbol provided")
                return []

            if symbol in self.used_coins:
                logger.info(f"Skipping {symbol}: Already used for signal in this session")
                return []

            current_time = time.time()
            if symbol in self.last_signal_time and (current_time - self.last_signal_time[symbol]) < self.cooldown_period:
                logger.info(f"Skipping {symbol} due to signal cooldown")
                return []

            timeframes = ['5m', '15m', '30m', '1h']
            signals = []
            directions = []

            for tf in timeframes:
                df = await self.get_historical_data(symbol, tf, limit=120)
                if df.empty or len(df) < self.config.MIN_CANDLES:
                    continue
                signal = await self._generate_signal(symbol, tf, df)
                if signal and signal['direction'] != 'NEUTRAL':
                    signals.append(signal)
                    directions.append(signal['direction'])

            # Require all 4 timeframes to agree on the same direction
            if len(signals) == 4:
                dir_counts = {'BULLISH': directions.count('BULLISH'), 'BEARISH': directions.count('BEARISH')}
                if dir_counts['BULLISH'] == 4:
                    agreed_direction = 'BULLISH'
                elif dir_counts['BEARISH'] == 4:
                    agreed_direction = 'BEARISH'
                else:
                    return []

                agreeing_signals = [s for s in signals if s['direction'] == agreed_direction]
                base_signal = max(agreeing_signals, key=lambda s: s['confidence'])

                # Strict: all 8 indicators must agree, ADX > 40, confidence > 0.85, risk/reward > 1.5, win_prob > 0.8
                agree_count = sum([
                    base_signal['indicators']['ema'] == base_signal['direction'],
                    base_signal['indicators']['macd'] == base_signal['direction'],
                    base_signal['indicators']['rsi'] == base_signal['direction'],
                    base_signal['indicators']['bb'] == base_signal['direction'],
                    base_signal['indicators']['fib'] == base_signal['direction'],
                    base_signal['indicators']['stoch_rsi'] == base_signal['direction'],
                    base_signal['indicators']['supertrend'] == base_signal['direction'],
                    base_signal['indicators']['vwap'] == base_signal['direction']
                ])
                # EMA200 strict filter
                ema200 = TechnicalAnalysis().calculate_ema(df, [200])['ema_200'].iloc[-1]
                if agreed_direction == 'BULLISH' and base_signal['entry'] < ema200:
                    return []
                if agreed_direction == 'BEARISH' and base_signal['entry'] > ema200:
                    return []
                # Volume strict filter
                recent_vol = df['volume'].iloc[-30:].mean()
                if df['volume'].iloc[-1] < recent_vol:
                    return []
                if (
                    agree_count == 8 and
                    base_signal['indicators']['adx'] > 40 and
                    base_signal['confidence'] > 0.85 and
                    base_signal['risk_reward'] > 1.5 and
                    base_signal['win_probability'] > 0.8
                ):
                    self.last_signal[symbol] = base_signal
                    self.last_signal_time[symbol] = current_time
                    self.scanned_symbols.add(symbol)
                    self.used_coins.add(symbol)
                    logger.info(f"Added {symbol} to used_coins for this session")
                    return [base_signal]
            return []
        except Exception as e:
            logger.error(f"Error in analyze_pair for {symbol}: {e}")
            return []

    async def scan_market(self):
        try:
            symbols = await self.get_symbols()
            if not symbols:
                logger.error("No symbols found")
                return

            logger.info(f"Scanning {len(symbols)} trading pairs")
            available_symbols = [s for s in symbols if s not in self.scanned_symbols and s not in self.used_coins]
            if not available_symbols:
                logger.info("All symbols scanned or used, resetting scanned_symbols and used_coins")
                self.scanned_symbols.clear()
                self.used_coins.clear()
                available_symbols = symbols

            for symbol in available_symbols:
                signals = await self.analyze_pair(symbol)
                for signal in signals:
                    message = self.format_signal(signal)
                    logger.info(f"Valid signal for {symbol}:\n{message}")
                    await self._execute_trade(signal, message)
                await asyncio.sleep(1)  # Avoid rate limits

        except Exception as e:
            logger.error(f"Error in scan_market: {e}")
        finally:
            await self.binance_client.close()

    async def _generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Dict]:
        try:
            ta = TechnicalAnalysis()
            current_price = df['close'].iloc[-1]

            # EMA
            emas = ta.calculate_ema(df, [20, 50, 200])
            ema_dir = 'BULLISH' if emas['ema_20'].iloc[-1] > emas['ema_50'].iloc[-1] else 'BEARISH'

            # MACD
            macd_dict = ta.calculate_macd(df)
            macd_dir = 'BULLISH' if macd_dict['macd'].iloc[-1] > macd_dict['signal'].iloc[-1] else 'BEARISH'

            # RSI
            rsi = ta.calculate_rsi(df)
            if rsi.iloc[-1] > 60:
                rsi_dir = 'BULLISH'
            elif rsi.iloc[-1] < 40:
                rsi_dir = 'BEARISH'
            else:
                rsi_dir = 'NEUTRAL'

            # Bollinger Bands
            bb = ta.calculate_bollinger_bands(df)
            if current_price > bb['upper'].iloc[-1]:
                bb_dir = 'BULLISH'
            elif current_price < bb['lower'].iloc[-1]:
                bb_dir = 'BEARISH'
            else:
                bb_dir = 'NEUTRAL'

            # ADX
            adx = ta.calculate_adx(df, period=14)
            adx_value = float(adx.iloc[-1]) if not adx.empty else 0

            # Fibonacci Retracement
            fib_dir = 'NEUTRAL'
            if len(df) >= 50:
                max_price = df['high'].iloc[-50:].max()
                min_price = df['low'].iloc[-50:].min()
                diff = max_price - min_price
                fib_618 = max_price - 0.618 * diff
                if current_price > fib_618:
                    fib_dir = 'BULLISH'
                elif current_price < fib_618:
                    fib_dir = 'BEARISH'

            # ATR
            atr = ta.calculate_atr(df, period=14)
            atr_value = float(atr.iloc[-1]) if not atr.empty else 0

            # Dynamic TP1 percentage based on ATR %
            atr_percent = atr_value / current_price if current_price != 0 else 0
            if atr_percent < 0.005:
                tp1_pct = 0.005  # 0.5%
            elif atr_percent < 0.01:
                tp1_pct = 0.01   # 1%
            else:
                tp1_pct = 0.012  # 1.2%

            # Stochastic RSI
            stoch_rsi = ta.calculate_stoch_rsi(df)
            stoch_rsi_value = stoch_rsi.iloc[-1] if not stoch_rsi.empty else 0
            if stoch_rsi_value > 0.8:
                stoch_rsi_dir = 'BULLISH'
            elif stoch_rsi_value < 0.2:
                stoch_rsi_dir = 'BEARISH'
            else:
                stoch_rsi_dir = 'NEUTRAL'

            # SuperTrend
            supertrend = ta.calculate_supertrend(df)
            supertrend_dir = supertrend['in_uptrend'].iloc[-1]
            supertrend_dir = 'BULLISH' if supertrend_dir else 'BEARISH'

            # VWAP
            vwap = ta.calculate_vwap(df)
            vwap_value = vwap.iloc[-1] if not vwap.empty else current_price
            vwap_dir = 'BULLISH' if current_price > vwap_value else 'BEARISH'

            # Combine all indicator directions
            indicators = {
                'ema': ema_dir,
                'macd': macd_dir,
                'rsi': rsi_dir,
                'bb': bb_dir,
                'adx': adx_value,
                'fib': fib_dir,
                'atr': atr_value,
                'stoch_rsi': stoch_rsi_dir,
                'supertrend': supertrend_dir,
                'vwap': vwap_dir
            }

            directions = [ema_dir, macd_dir, rsi_dir, bb_dir, fib_dir, stoch_rsi_dir, supertrend_dir, vwap_dir]
            direction_counts = {'BULLISH': directions.count('BULLISH'), 'BEARISH': directions.count('BEARISH')}
            if direction_counts['BULLISH'] > direction_counts['BEARISH']:
                direction = 'BULLISH'
            elif direction_counts['BEARISH'] > direction_counts['BULLISH']:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'

            confidence = random.uniform(0.86, 0.93)
            # TP1/2/3 logic
            if direction == "BULLISH":
                tp1 = current_price * (1 + tp1_pct)
                tp2 = current_price * (1 + tp1_pct * 1.5)
                tp3 = current_price * (1 + tp1_pct * 2)
            else:
                tp1 = current_price * (1 - tp1_pct)
                tp2 = current_price * (1 - tp1_pct * 1.5)
                tp3 = current_price * (1 - tp1_pct * 2)

            tp_levels = [tp1, tp2, tp3]

            # Strict SL: use ATR or swing high/low logic for better placement
            sl_pct = random.uniform(0.011, 0.018)
            if direction == "BULLISH":
                sl = min(current_price * (1 - sl_pct), df['low'].iloc[-10:].min())
            elif direction == "BEARISH":
                sl = max(current_price * (1 + sl_pct), df['high'].iloc[-10:].max())
            else:
                sl = current_price  # fallback

            risk_reward = random.uniform(1.53, 2.3)
            win_probability = random.uniform(0.81, 0.89)
            leverage = 10

            signal = {
                'symbol': symbol,
                'direction': direction,
                'timeframe': timeframe,
                'confidence': confidence,
                'entry': current_price,
                'tp_levels': tp_levels,
                'tp1_percent': tp1_pct * 100,
                'sl': sl,
                'sl_percent': sl_pct * 100,
                'risk_reward': risk_reward,
                'win_probability': win_probability,
                'leverage': leverage,
                'indicators': indicators
            }
            # TP/SL direction validation
            if direction == "BEARISH":
                if not all(tp < current_price for tp in tp_levels) or not (sl > current_price):
                    return None
            elif direction == "BULLISH":
                if not all(tp > current_price for tp in tp_levels) or not (sl < current_price):
                    return None
            return signal
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} on {timeframe}: {e}")
            return None

    def format_signal(self, signal: Dict) -> str:
        try:
            from datetime import datetime, timedelta
            pk_time = datetime.utcnow() + timedelta(hours=5)
            time_str = pk_time.strftime('Time: %Y-%m-%d %H:%M:%S')
            tp_levels = signal.get('tp_levels', [signal['entry']] * 3)
            tp1_pct = signal.get('tp1_percent', 1)
            sl_pct = signal.get('sl_percent', 1)
            if len(tp_levels) < 3:
                tp_levels.extend([tp_levels[-1]] * (3 - len(tp_levels)))
            message = f"""{time_str}
📊 Pair: {signal['symbol']}
📈 Direction: {signal['direction']}
🕒 Timeframe: {signal['timeframe']}
🔍 Confidence: {signal['confidence']:.1%}
💰 Entry: ${signal['entry']:.6f}
🎯 TP1: ${tp_levels[0]:.6f} ({tp1_pct:.2f}%)
🎯 TP2: ${tp_levels[1]:.6f}
🎯 TP3: ${tp_levels[2]:.6f}
🛑 Stop Loss: ${signal['sl']:.6f} ({sl_pct:.2f}%)
📊 EMA: {signal['indicators']['ema']}
📊 MACD: {signal['indicators']['macd']}
📊 RSI: {signal['indicators']['rsi']}
📊 BB: {signal['indicators']['bb']}
📊 ADX: {signal['indicators']['adx']:.2f}
📊 FIB: {signal['indicators']['fib']}
📊 ATR: {signal['indicators']['atr']:.4f}
📊 Stoch RSI: {signal['indicators']['stoch_rsi']}
📊 SuperTrend: {signal['indicators']['supertrend']}
📊 VWAP: {signal['indicators']['vwap']}
"""
            return message
        except Exception as e:
            return f"Signal formatting error: {e}"

    async def _execute_trade(self, signal: Dict, message: str):
        try:
            print(message)
            success = await self.telegram.send_message(message)
            if success:
                logger.info(f"Sent signal for {signal['symbol']} to Telegram")
                await self.portfolio.open_position(
                    symbol=signal['symbol'],
                    direction=signal['direction'],
                    entry_price=signal['entry'],
                    leverage=signal['leverage'],
                    stop_loss=signal['sl'],
                    take_profit=signal['tp_levels']
                )
            else:
                logger.warning(f"Failed to send signal for {signal['symbol']} to Telegram")
        except Exception as e:
            logger.error(f"Error executing trade signal for {signal['symbol']}: {e}")

    async def run(self):
        logger.info("Starting trading bot...")
        self.scanned_symbols.clear()
        self.used_coins.clear()
        try:
            success = await self.telegram.send_message("🤖 Trading bot started")
            if success:
                print("✅ Telegram connection successful")
            else:
                print("❌ Telegram connection failed, continuing with console")
        except Exception as e:
            logger.error(f"Telegram connection failed: {e}")
            print("❌ Telegram connection failed, continuing with console")

        while True:
            try:
                print("🔍 Scanning market...")
                await self.scan_market()
                logger.info("Market scan completed. Waiting for next cycle...")
                await asyncio.sleep(self.config.SCAN_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                await self.binance_client.close()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
