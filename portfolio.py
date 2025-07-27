import logging
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ccxt.async_support as ccxt
from decimal import Decimal, ROUND_DOWN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Portfolio:
    """
    Handles position management, risk calculation, and portfolio tracking
    for the trading bot.
    """

    def __init__(self, exchange: ccxt.Exchange, config: object):
        """
        Initialize the Portfolio manager.

        Args:
            exchange: Initialized ccxt exchange instance
            config: Configuration object with trading parameters
        """
        self.exchange = exchange
        self.config = config
        self.positions: Dict[str, Dict] = {}  # Active positions
        self.trade_history: List[Dict] = []   # Historical trades
        self.balance = {
            'total': Decimal('0'),
            'free': Decimal('0'),
            'used': Decimal('0'),
            'usd_value': Decimal('0')
        }
        self.initial_balance = Decimal('0')
        self.leverage = config.MAX_LEVERAGE
        self.max_drawdown_pct = Decimal(str(config.MAX_DRAWDOWN_PERCENT)) / Decimal('100')
        self.daily_loss_limit = Decimal(str(config.MAX_DAILY_LOSS_PERCENT)) / Decimal('100')

        # Initialize risk parameters
        self.risk_per_trade = Decimal(str(config.RISK_PER_TRADE))
        self.min_risk_reward = Decimal(str(config.MIN_RISK_REWARD))

        # Track daily performance
        self.daily_pnl = {
            'date': datetime.utcnow().date(),
            'starting_balance': Decimal('0'),
            'current_balance': Decimal('0'),
            'closed_pnl': Decimal('0'),
            'open_pnl': Decimal('0'),
            'fees': Decimal('0')
        }

    async def initialize(self) -> bool:
        """
        Initialize the portfolio with current exchange data.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Load markets if not already loaded
            await self.exchange.load_markets()

            # Set exchange leverage if futures trading
            if 'futures' in self.exchange.urls:
                await self.exchange.set_leverage(self.leverage, 'BTC/USDT')
                await self.exchange.set_margin_mode('isolated', 'BTC/USDT')

            # Get initial balance
            await self.update_balance()
            self.initial_balance = self.balance['total']
            self.daily_pnl['starting_balance'] = self.balance['total']
            self.daily_pnl['current_balance'] = self.balance['total']

            logger.info(f"Portfolio initialized with balance: {self.balance['total']:.2f} USDT")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize portfolio: {e}")
            return False

    async def update_balance(self) -> None:
        """Update the current balance from exchange"""
        try:
            balance = await self.exchange.fetch_balance()
            if 'USDT' in balance['total']:
                self.balance = {
                    'total': Decimal(str(balance['total']['USDT'])),
                    'free': Decimal(str(balance['free']['USDT'])),
                    'used': Decimal(str(balance['used']['USDT'])),
                    'usd_value': Decimal(str(balance['total']['USDT']))
                }
        except Exception as e:
            logger.error(f"Error updating balance: {e}")
            # Use demo balance if API fails
            if self.balance['total'] == Decimal('0'):
                self.balance = {'total': Decimal('10000.0'), 'free': Decimal('10000.0'), 'used': Decimal('0.0'), 'usd_value': Decimal('10000.0')}

    async def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float, 
        risk_percent: float = None
    ) -> Tuple[float, float, float]:
        """
        Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            risk_percent: Risk percentage per trade (overrides config)

        Returns:
            Tuple[float, float, float]: (position_size, quantity, risk_amount)
        """
        try:
            # Convert to Decimal for precise calculations
            entry = Decimal(str(entry_price))
            stop = Decimal(str(stop_loss))

            # Use provided risk or config default
            risk_pct = Decimal(str(risk_percent)) if risk_percent is not None else self.risk_per_trade

            # Calculate risk amount
            account_balance = self.balance['total']
            risk_amount = account_balance * risk_pct

            # Calculate position size
            if entry > stop:  # Long position
                risk_per_share = entry - stop
            else:  # Short position
                risk_per_share = stop - entry

            if risk_per_share <= 0:
                logger.warning(f"Invalid stop loss: {stop} for entry: {entry}")
                return 0.0, 0.0, 0.0

            position_size = (risk_amount / risk_per_share) * entry
            quantity = float((Decimal(str(position_size)) / entry).quantize(
                Decimal('0.00000001'), rounding=ROUND_DOWN
            ))

            # Ensure we don't exceed available balance
            max_position_size = float(account_balance * Decimal('0.9'))  # Max 90% of balance
            position_size = min(position_size, Decimal(str(max_position_size)))

            # Recalculate quantity with capped position size
            quantity = float((Decimal(str(position_size)) / entry).quantize(
                Decimal('0.00000001'), rounding=ROUND_DOWN
            ))

            return float(position_size), quantity, float(risk_amount)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0, 0.0, 0.0

    async def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: List[float],
        leverage: int = None,
        risk_percent: float = None
    ) -> Dict:
        """
        Open a new position.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            direction: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: List of take profit levels
            leverage: Leverage to use (overrides config)
            risk_percent: Risk percentage (overrides config)

        Returns:
            Dict: Trade details or error
        """
        try:
            # Input validation
            if direction.lower() not in ['long', 'short']:
                return {'error': 'Invalid direction. Must be "long" or "short"'}

            if symbol in self.positions:
                return {'error': f'Position for {symbol} already exists'}

            # Calculate position size
            position_size, quantity, risk_amount = await self.calculate_position_size(
                entry_price, stop_loss, risk_percent
            )

            if position_size <= 0 or quantity <= 0:
                return {'error': 'Invalid position size calculation'}

            # Apply leverage
            lev = leverage if leverage is not None else self.leverage
            position_size *= lev
            risk_amount *= lev

            # Check available balance
            if position_size > float(self.balance['free']):
                return {'error': 'Insufficient balance'}

            # Create order
            order_type = 'limit' if entry_price > 0 else 'market'
            side = 'buy' if direction.lower() == 'long' else 'sell'

            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=entry_price if order_type == 'limit' else None,
                params={
                    'leverage': lev,
                    'stopLoss': {
                        'price': stop_loss,
                        'type': 'stopMarket'
                    },
                    'takeProfit': {
                        'price': take_profit[0] if take_profit else None,
                        'type': 'takeProfitMarket'
                    }
                }
            )

            # Record position
            self.positions[symbol] = {
                'symbol': symbol,
                'direction': direction.lower(),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': lev,
                'quantity': quantity,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'entry_time': datetime.utcnow(),
                'order_id': order['id'],
                'status': 'open'
            }

            # Update balance
            await self.update_balance()

            logger.info(f"Opened {direction} position for {symbol}: {quantity} @ {entry_price}")
            return self.positions[symbol]

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return {'error': str(e)}

    async def close_position(
        self, 
        symbol: str, 
        reason: str = 'manual',
        price: float = None
    ) -> Dict:
        """
        Close an open position.

        Args:
            symbol: Trading pair
            reason: Reason for closing
            price: Optional exit price (None for market)

        Returns:
            Dict: Close details or error
        """
        try:
            if symbol not in self.positions:
                return {'error': f'No open position for {symbol}'}

            position = self.positions[symbol]

            # Determine exit price if not provided
            if price is None:
                ticker = await self.exchange.fetch_ticker(symbol)
                price = ticker['last']

            # Calculate P&L
            entry = Decimal(str(position['entry_price']))
            exit_price = Decimal(str(price))
            quantity = Decimal(str(position['quantity']))

            if position['direction'] == 'long':
                pnl = (exit_price - entry) * quantity
            else:
                pnl = (entry - exit_price) * quantity

            pnl_pct = (pnl / (entry * quantity)) * Decimal('100')

            # Create close order
            side = 'sell' if position['direction'] == 'long' else 'buy'

            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position['quantity'],
                params={'reduceOnly': True}
            )

            # Record trade
            trade = {
                'symbol': symbol,
                'direction': position['direction'],
                'entry_price': float(entry),
                'exit_price': float(exit_price),
                'quantity': float(quantity),
                'pnl': float(pnl),
                'pnl_pct': float(pnl_pct),
                'leverage': position['leverage'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.utcnow(),
                'duration': (datetime.utcnow() - position['entry_time']).total_seconds() / 60,  # in minutes
                'reason': reason,
                'status': 'closed'
            }

            # Update trade history
            self.trade_history.append(trade)

            # Update daily P&L
            self.daily_pnl['closed_pnl'] += pnl
            self.daily_pnl['current_balance'] += pnl

            # Remove from active positions
            del self.positions[symbol]

            # Update balance
            await self.update_balance()

            logger.info(f"Closed {symbol} position: P&L {float(pnl):.2f} USDT ({float(pnl_pct):.2f}%)")
            return trade

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'error': str(e)}

    async def update_positions(self) -> None:
        """Update all open positions with current market data."""
        try:
            for symbol in list(self.positions.keys()):
                await self.update_position(symbol)
        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    async def update_position(self, symbol: str) -> Dict:
        """
        Update a single position with current market data.

        Args:
            symbol: Trading pair

        Returns:
            Dict: Updated position or error
        """
        try:
            if symbol not in self.positions:
                return {'error': f'No open position for {symbol}'}

            position = self.positions[symbol]
            ticker = await self.exchange.fetch_ticker(symbol)

            # Calculate current P&L
            current_price = Decimal(str(ticker['last']))
            entry = Decimal(str(position['entry_price']))
            quantity = Decimal(str(position['quantity']))

            if position['direction'] == 'long':
                pnl = (current_price - entry) * quantity
            else:
                pnl = (entry - current_price) * quantity

            pnl_pct = (pnl / (entry * quantity)) * Decimal('100')

            # Update position
            position['current_price'] = float(current_price)
            position['unrealized_pnl'] = float(pnl)
            position['unrealized_pnl_pct'] = float(pnl_pct)

            # Check if stop loss or take profit hit
            if position['stop_loss'] is not None:
                if ((position['direction'] == 'long' and current_price <= Decimal(str(position['stop_loss']))) or
                    (position['direction'] == 'short' and current_price >= Decimal(str(position['stop_loss'])))):
                    await self.close_position(symbol, 'stop_loss')
                    return {'status': 'closed', 'reason': 'stop_loss'}

            if position['take_profit'] and len(position['take_profit']) > 0:
                for i, tp in enumerate(position['take_profit']):
                    tp = Decimal(str(tp))
                    if ((position['direction'] == 'long' and current_price >= tp) or
                        (position['direction'] == 'short' and current_price <= tp)):
                        await self.close_position(symbol, f'take_profit_{i+1}')
                        return {'status': 'closed', 'reason': f'take_profit_{i+1}'}

            return position

        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return {'error': str(e)}

    async def get_portfolio_summary(self) -> Dict:
        """
        Get a summary of the portfolio.

        Returns:
            Dict: Portfolio summary
        """
        try:
            await self.update_balance()
            await self.update_positions()

            # Calculate total P&L
            total_pnl = Decimal('0')
            total_invested = Decimal('0')

            for position in self.positions.values():
                total_pnl += Decimal(str(position.get('unrealized_pnl', 0)))
                total_invested += Decimal(str(position.get('position_size', 0)))

            # Calculate daily P&L
            today = datetime.utcnow().date()
            if self.daily_pnl['date'] != today:
                # Reset daily P&L for new day
                self.daily_pnl = {
                    'date': today,
                    'starting_balance': self.balance['total'],
                    'current_balance': self.balance['total'],
                    'closed_pnl': Decimal('0'),
                    'open_pnl': total_pnl,
                    'fees': Decimal('0')
                }
            else:
                self.daily_pnl['open_pnl'] = total_pnl
                self.daily_pnl['current_balance'] = self.balance['total']

            # Calculate performance metrics
            initial_balance = Decimal(str(self.initial_balance))
            current_balance = self.balance['total']
            total_return = ((current_balance - initial_balance) / initial_balance * Decimal('100')) if initial_balance > 0 else Decimal('0')

            return {
                'balance': {
                    'total': float(current_balance),
                    'free': float(self.balance['free']),
                    'used': float(self.balance['used']),
                    'initial': float(initial_balance),
                    'total_return_pct': float(total_return)
                },
                'positions': {
                    'count': len(self.positions),
                    'total_invested': float(total_invested),
                    'total_unrealized_pnl': float(total_pnl),
                    'open_positions': list(self.positions.keys())
                },
                'daily': {
                    'date': self.daily_pnl['date'].isoformat(),
                    'starting_balance': float(self.daily_pnl['starting_balance']),
                    'current_balance': float(self.daily_pnl['current_balance']),
                    'closed_pnl': float(self.daily_pnl['closed_pnl']),
                    'open_pnl': float(self.daily_pnl['open_pnl']),
                    'fees': float(self.daily_pnl['fees']),
                    'daily_pnl_pct': float((
                        (self.daily_pnl['current_balance'] - self.daily_pnl['starting_balance']) / 
                        self.daily_pnl['starting_balance'] * Decimal('100')
                    ) if self.daily_pnl['starting_balance'] > 0 else 0)
                },
                'risk': {
                    'max_drawdown_pct': float(self.max_drawdown_pct * Decimal('100')),
                    'daily_loss_limit_pct': float(self.daily_loss_limit * Decimal('100')),
                    'risk_per_trade_pct': float(self.risk_per_trade * Decimal('100')),
                    'min_risk_reward': float(self.min_risk_reward)
                }
            }

        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {'error': str(e)}

    async def check_risk_limits(self) -> Dict:
        """
        Check if any risk limits have been exceeded.

        Returns:
            Dict: Risk status and any actions taken
        """
        try:
            summary = await self.get_portfolio_summary()

            # Check daily loss limit
            starting_balance = summary['daily']['starting_balance']
            if starting_balance > 0:
                daily_pnl_pct = (summary['daily']['current_balance'] - starting_balance) / starting_balance * 100
            else:
                daily_pnl_pct = 0
            if daily_pnl_pct < -float(self.daily_loss_limit * Decimal('100')):
                # Close all positions
                for symbol in list(self.positions.keys()):
                    await self.close_position(symbol, 'daily_loss_limit')
                return {
                    'status': 'limit_exceeded',
                    'limit': 'daily_loss',
                    'value': daily_pnl_pct,
                    'limit_value': -float(self.daily_loss_limit * Decimal('100')),
                    'action': 'closed_all_positions'
                }

            # Check max drawdown
            max_drawdown = (1 - (summary['balance']['total'] / max(
                summary['balance']['initial'],
                summary['daily']['starting_balance']
            ))) * 100

            if max_drawdown > float(self.max_drawdown_pct * 100):
                # Close all positions
                for symbol in list(self.positions.keys()):
                    await self.close_position(symbol, 'max_drawdown')
                return {
                    'status': 'limit_exceeded',
                    'limit': 'max_drawdown',
                    'value': max_drawdown,
                    'limit_value': float(self.max_drawdown_pct * 100),
                    'action': 'closed_all_positions'
                }

            return {'status': 'within_limits'}

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {'error': str(e)}

    async def close_all_positions(self, reason: str = 'manual') -> Dict:
        """
        Close all open positions.

        Args:
            reason: Reason for closing

        Returns:
            Dict: Results of closing positions
        """
        results = {}
        for symbol in list(self.positions.keys()):
            results[symbol] = await self.close_position(symbol, reason)
        return results

    async def get_trade_history(
        self, 
        symbol: str = None, 
        limit: int = 100,
        start_time: int = None,
        end_time: int = None
    ) -> List[Dict]:
        """
        Get trade history with filtering options.

        Args:
            symbol: Filter by symbol
            limit: Maximum number of trades to return
            start_time: Filter trades after this timestamp (ms)
            end_time: Filter trades before this timestamp (ms)

        Returns:
            List[Dict]: Filtered trade history
        """
        try:
            history = self.trade_history.copy()

            # Apply filters
            if symbol:
                history = [t for t in history if t['symbol'] == symbol]

            if start_time:
                history = [t for t in history if int(t['exit_time'].timestamp() * 1000) >= start_time]

            if end_time:
                history = [t for t in history if int(t['exit_time'].timestamp() * 1000) <= end_time]

            # Sort by exit time (newest first)
            history.sort(key=lambda x: x['exit_time'], reverse=True)

            return history[:limit]

        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []

    async def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the portfolio.

        Returns:
            Dict: Performance metrics
        """
        try:
            if not self.trade_history:
                return {'error': 'No trade history available'}

            # Calculate metrics
            wins = [t for t in self.trade_history if t['pnl'] > 0]
            losses = [t for t in self.trade_history if t['pnl'] <= 0]

            total_trades = len(self.trade_history)
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

            avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else float('inf')

            total_pnl = sum(t['pnl'] for t in self.trade_history)
            max_drawdown = min(t['pnl'] for t in self.trade_history) if self.trade_history else 0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': 0,  # Would require returns data
                'sortino_ratio': 0   # Would require returns data
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self.exchange.close()
        except Exception as e:
            logger.error(f"Error during portfolio cleanup: {e}")

# Example usage
if __name__ == "__main__":
    import asyncio
    from config import Config

    async def main():
        config = Config()
        exchange = ccxt.binance({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_API_SECRET,
            'enableRateLimit': True
        })

        portfolio = Portfolio(exchange, config)
        await portfolio.initialize()

        try:
            # Get portfolio summary
            summary = await portfolio.get_portfolio_summary()
            print("Portfolio Summary:", summary)

            # Update and check positions
            await portfolio.update_positions()

            # Check risk limits
            risk = await portfolio.check_risk_limits()
            print("Risk check:", risk)

        finally:
            await portfolio.close()

    asyncio.run(main())