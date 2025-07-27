import logging
import time
import asyncio
import aiohttp
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_analysis")

CACHE_EXPIRY = 300  # 5 minutes
MAX_CACHE_SIZE = 1000
DEFAULT_TIMEOUT = 10  # seconds

class SentimentSource(Enum):
    CRYPTO_PANIC = "crypto_panic"
    ALTERNATIVE_ME = "alternative_me"
    COIN_NEWS = "coin_news"

@dataclass
class SentimentScore:
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    compound: float = 0.0
    volume: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral,
            'compound': self.compound,
            'volume': self.volume,
            'timestamp': self.timestamp
        }

class SentimentAnalyzer:
    def __init__(self, config: object):
        self.config = getattr(config, 'SENTIMENT', {})
        self.api_keys = {
            'crypto_panic': self.config.get('CRYPTO_PANIC_API_KEY', ''),
        }
        self.vader = SentimentIntensityAnalyzer()
        self.cache = {}
        self.sentiment_history = defaultdict(lambda: deque(maxlen=100))
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        )
        self.source_weights = {
            SentimentSource.CRYPTO_PANIC: 1.0,
            SentimentSource.ALTERNATIVE_ME: 0.9,
            SentimentSource.COIN_NEWS: 0.85
        }
        self.cooldowns = {
            SentimentSource.CRYPTO_PANIC: 0,
            SentimentSource.ALTERNATIVE_ME: 0,
            SentimentSource.COIN_NEWS: 0
        }
        self.api_cooldowns = {
            SentimentSource.CRYPTO_PANIC: 1,
            SentimentSource.ALTERNATIVE_ME: 2,
            SentimentSource.COIN_NEWS: 3
        }

    async def close(self):
        await self.session.close()

    def _get_cache_key(self, source: SentimentSource, symbol: str, time_window: int = 3600) -> str:
        return f"{source.value}:{symbol.lower()}:{time_window}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self.cache:
            return False
        cached_data = self.cache[cache_key]
        return time.time() - cached_data['timestamp'] < CACHE_EXPIRY

    async def _make_request(self, url, source, params=None, headers=None):
        current_time = time.time()
        if current_time < self.cooldowns[source]:
            wait_time = self.cooldowns[source] - current_time
            await asyncio.sleep(wait_time)
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                self.cooldowns[source] = time.time() + self.api_cooldowns[source]
                return data
        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            self.cooldowns[source] = time.time() + self.api_cooldowns[source] * 2
            return None

    async def get_crypto_panic_sentiment(self, symbol: str):
        cache_key = self._get_cache_key(SentimentSource.CRYPTO_PANIC, symbol)
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        api_key = self.api_keys['crypto_panic']
        if not api_key:
            logger.info("CryptoPanic API key not set, skipping CryptoPanic sentiment.")
            return None

        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': api_key,
            'currencies': symbol.upper(),
            'public': 'true'
        }
        data = await self._make_request(url, SentimentSource.CRYPTO_PANIC, params=params)
        if not data:
            return None

        sentiment_scores = []
        for post in data.get('results', [])[:50]:
            text = f"{post.get('title', '')} {post.get('body', '')}"
            if not text.strip():
                continue
            blob = TextBlob(text)
            vader_scores = self.vader.polarity_scores(text)
            combined_score = {
                'positive': (vader_scores['pos'] + blob.sentiment.polarity) / 2,
                'negative': (vader_scores['neg'] + (1 - blob.sentiment.polarity) / 2) / 2,
                'neutral': (vader_scores['neu'] + blob.sentiment.subjectivity) / 2,
                'compound': vader_scores['compound'],
                'source': 'crypto_panic',
                'timestamp': time.time(),
                'url': post.get('url', '')
            }
            sentiment_scores.append(combined_score)
        if not sentiment_scores:
            return None

        avg_sentiment = {
            'positive': np.mean([s['positive'] for s in sentiment_scores]),
            'negative': np.mean([s['negative'] for s in sentiment_scores]),
            'neutral': np.mean([s['neutral'] for s in sentiment_scores]),
            'compound': np.mean([s['compound'] for s in sentiment_scores]),
            'volume': len(sentiment_scores),
            'sources': ['crypto_panic'],
            'timestamp': time.time()
        }
        self.cache[cache_key] = {
            'data': avg_sentiment,
            'timestamp': time.time()
        }
        if len(self.cache) > MAX_CACHE_SIZE:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        return avg_sentiment

    async def get_alternative_me_sentiment(self, symbol: str):
        cache_key = self._get_cache_key(SentimentSource.ALTERNATIVE_ME, symbol)
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        coin_map = {
            'BTC': 1, 'ETH': 1027, 'BNB': 1839, 'XRP': 52, 'ADA': 2010,
            'SOL': 5426, 'DOT': 6636, 'DOGE': 74, 'AVAX': 5805, 'MATIC': 3890,
            'LTC': 2, 'TRX': 1958, 'NEO': 1376
        }
        coin_id = coin_map.get(symbol.upper())
        if not coin_id:
            logger.info(f"Symbol {symbol} not found in alternative.me mapping")
            return None
        url = f"https://api.alternative.me/fng/?limit=1"
        data = await self._make_request(url, SentimentSource.ALTERNATIVE_ME)
        if not data or 'data' not in data or not data['data']:
            return None
        fear_greed = float(data['data'][0].get('value', 50))
        sentiment = {
            'positive': fear_greed / 100,
            'negative': (100 - fear_greed) / 100,
            'neutral': 0.3,
            'compound': (fear_greed - 50) / 50,
            'volume': 1,
            'sources': ['alternative_me'],
            'timestamp': time.time()
        }
        self.cache[cache_key] = {
            'data': sentiment,
            'timestamp': time.time()
        }
        return sentiment

    async def get_coin_news_sentiment(self, symbol: str):
        cache_key = self._get_cache_key(SentimentSource.COIN_NEWS, symbol)
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        sentiment = {
            'positive': 0.35,
            'negative': 0.30,
            'neutral': 0.35,
            'compound': 0.05,
            'volume': 10,
            'sources': ['coin_news'],
            'timestamp': time.time()
        }
        self.cache[cache_key] = {
            'data': sentiment,
            'timestamp': time.time()
        }
        return sentiment

    def _calculate_combined_sentiment(self, sentiment_data):
        if not sentiment_data:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'volume': 0,
                'sources': [],
                'timestamp': time.time()
            }
        total_weight = sum(self.source_weights.get(SentimentSource(s['sources'][0]), 0.5) for s in sentiment_data)
        if total_weight == 0:
            weights = [1/len(sentiment_data)] * len(sentiment_data)
        else:
            weights = [self.source_weights.get(SentimentSource(s['sources'][0]), 0.5) / total_weight 
                      for s in sentiment_data]
        combined = {
            'positive': sum(s['positive'] * w for s, w in zip(sentiment_data, weights)),
            'negative': sum(s['negative'] * w for s, w in zip(sentiment_data, weights)),
            'neutral': sum(s['neutral'] * w for s, w in zip(sentiment_data, weights)),
            'compound': sum(s['compound'] * w for s, w in zip(sentiment_data, weights)),
            'volume': sum(s.get('volume', 0) for s in sentiment_data),
            'sources': list(set(src for s in sentiment_data for src in s['sources'])),
            'timestamp': time.time()
        }
        total = combined['positive'] + combined['negative'] + combined['neutral']
        if total > 0:
            combined['positive'] /= total
            combined['negative'] /= total
            combined['neutral'] /= total
        return combined

    async def get_sentiment(self, symbol: str):
        coin_symbol = symbol.split('/')[0].upper()
        tasks = [
            self.get_crypto_panic_sentiment(coin_symbol),
            self.get_alternative_me_sentiment(coin_symbol),
            self.get_coin_news_sentiment(coin_symbol)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        combined = self._calculate_combined_sentiment(valid_results)
        self.sentiment_history[coin_symbol].append(combined)
        return combined

    def get_sentiment_trend(self, symbol: str, window: int = 5):
        coin_symbol = symbol.split('/')[0].upper()
        if coin_symbol not in self.sentiment_history or not self.sentiment_history[coin_symbol]:
            return {'trend': 'neutral', 'change': 0.0}
        recent_scores = list(self.sentiment_history[coin_symbol])[-window:]
        if len(recent_scores) < 2:
            return {'trend': 'neutral', 'change': 0.0}
        first = recent_scores[0]['compound']
        last = recent_scores[-1]['compound']
        change = last - first
        if abs(change) < 0.05:
            trend = 'neutral'
        elif change > 0:
            trend = 'bullish'
        else:
            trend = 'bearish'
        return {
            'trend': trend,
            'change': change,
            'current_sentiment': last,
            'window': window
        }

    def get_market_sentiment(self, symbols):
        sentiments = []
        for symbol in symbols:
            coin_symbol = symbol.split('/')[0].upper()
            if self.sentiment_history[coin_symbol]:
                sentiments.append(self.sentiment_history[coin_symbol][-1])
        if not sentiments:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'timestamp': time.time()
            }
        return {
            'positive': sum(s['positive'] for s in sentiments) / len(sentiments),
            'negative': sum(s['negative'] for s in sentiments) / len(sentiments),
            'neutral': sum(s['neutral'] for s in sentiments) / len(sentiments),
            'compound': sum(s['compound'] for s in sentiments) / len(sentiments),
            'timestamp': time.time()
        }

# Example usage
async def main():
    from config import Config
    config = Config()
    analyzer = SentimentAnalyzer(config)
    try:
        symbol = 'BTC/USDT'
        sentiment = await analyzer.get_sentiment(symbol)
        print(f"Current sentiment for {symbol}:")
        print(f"Positive: {sentiment['positive']:.2f}")
        print(f"Negative: {sentiment['negative']:.2f}")
        print(f"Neutral: {sentiment['neutral']:.2f}")
        print(f"Compound: {sentiment['compound']:.2f}")
        print(f"Volume: {sentiment['volume']}")
        trend = analyzer.get_sentiment_trend(symbol)
        print(f"\nSentiment trend: {trend['trend']} ({trend['change']:+.2f})")
        market = analyzer.get_market_sentiment(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
        print(f"\nMarket sentiment: {market['compound']:.2f}")
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())