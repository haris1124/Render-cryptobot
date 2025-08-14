import os

class Config:
    def __init__(self):
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'NgGWZfcUgyB6QVFOzCQVJmkOSa1cfusNi1w6emTZPapbrYf60xoL6olA4B70Eb3h')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'oIUp076Xd5YVwXFo82E8DE2fOwIcrmFSvePHH1YHJI31NlGZqv5j5soL7MdlbZoU')
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7894745130:AAHlFYuDAuRlhStlVQ6UDsNlUMiYzar0Bvo')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-1002574858930')
        self.CAPITAL = float(os.getenv('CAPITAL', 10000))
        self.RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
        self.MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', 40))
        self.MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', 1.2))
        self.MIN_WIN_PERCENTAGE = float(os.getenv('MIN_WIN_PERCENTAGE', 0.2))
        self.MAX_PAIRS_PER_CYCLE = int(os.getenv('MAX_PAIRS_PER_CYCLE', 50))
        self.SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', 120))
        self.MIN_CANDLES = int(os.getenv('MIN_CANDLES', 20))
        self.MAX_VOLATILITY = float(os.getenv('MAX_VOLATILITY', 0.15))
        self.MAX_DRAWDOWN_PERCENT = float(os.getenv('MAX_DRAWDOWN_PERCENT', 20))
        self.MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', 0.8))         # Minimum confidence for signals
        self.MIN_RISK_REWARD = float(os.getenv('MIN_RISK_REWARD', 1.7))       # Minimum risk/reward ratio
        self.MIN_WIN_PROBABILITY = float(os.getenv('MIN_WIN_PROBABILITY', 0.7)) # Minimum win probability
        self.MIN_TREND_STRENGTH = float(os.getenv('MIN_TREND_STRENGTH', 25))  # Minimum ADX/trend strength
        self.MIN_INDICATORS = int(os.getenv('MIN_INDICATORS', 4))             # Minimum indicators in agreement
        self.MAX_DAILY_LOSS_PERCENT = float(os.getenv('MAX_DAILY_LOSS_PERCENT', 5))
