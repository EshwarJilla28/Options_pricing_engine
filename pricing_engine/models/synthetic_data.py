import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from .black_scholes import BlackScholesModel

class EnhancedSyntheticDataGenerator:
    """Generate realistic synthetic options data with market microstructure"""
    
    def __init__(self):
        self.bs_model = BlackScholesModel()
    
    def generate_realistic_options_chain(self, symbol: str, spot_price: float, 
                                       risk_free_rate: float = 0.05) -> Dict:
        """Generate a realistic options chain with market-like characteristics"""
        
        # Multiple expiration dates
        base_date = datetime.now()
        expiry_dates = [
            base_date + timedelta(days=7),   # Weekly
            base_date + timedelta(days=14),  # Bi-weekly
            base_date + timedelta(days=30),  # Monthly
            base_date + timedelta(days=60),  # 2-month
            base_date + timedelta(days=90),  # 3-month
            base_date + timedelta(days=180), # 6-month
        ]
        
        options_data = []
        
        for expiry in expiry_dates:
            time_to_expiry = max((expiry - base_date).days / 365.0, 1/365)
            
            # Generate strikes around spot price
            strike_range = self._generate_realistic_strikes(spot_price)
            
            for strike in strike_range:
                for option_type in ['call', 'put']:
                    # Generate realistic implied volatility with smile
                    iv = self._generate_iv_with_smile(spot_price, strike, time_to_expiry, option_type)
                    
                    # Calculate theoretical price using IV
                    if option_type == 'call':
                        theoretical_price = self.bs_model.call_price(
                            spot_price, strike, time_to_expiry, risk_free_rate, iv
                        )
                    else:
                        theoretical_price = self.bs_model.put_price(
                            spot_price, strike, time_to_expiry, risk_free_rate, iv
                        )
                    
                    # Add realistic market microstructure
                    market_data = self._add_market_microstructure(
                        theoretical_price, strike, spot_price, time_to_expiry, option_type
                    )
                    
                    options_data.append({
                        'symbol': symbol,
                        'strike': float(strike),
                        'option_type': option_type,
                        'expiry_date': expiry.strftime('%Y-%m-%d'),
                        'time_to_expiry': time_to_expiry,
                        'implied_volatility': round(iv, 4),
                        'market_price': market_data['market_price'],
                        'bid': market_data['bid'],
                        'ask': market_data['ask'],
                        'volume': market_data['volume'],
                        'open_interest': market_data['open_interest'],
                        'last_trade': market_data['last_trade'],
                        'bid_ask_spread': market_data['bid_ask_spread'],
                        'moneyness': round(spot_price / strike if option_type == 'call' else strike / spot_price, 4)
                    })
        
        return {
            'symbol': symbol,
            'current_price': spot_price,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'enhanced_synthetic',
            'total_contracts': len(options_data),
            'expiry_dates': [exp.strftime('%Y-%m-%d') for exp in expiry_dates],
            'options': options_data
        }
    
    def _generate_realistic_strikes(self, spot_price: float) -> np.ndarray:
        """Generate realistic strike distribution"""
        # Create strikes with different spacing for different moneyness levels
        
        # Close to ATM: $1 intervals
        atm_range = np.arange(
            max(5, spot_price * 0.9),
            spot_price * 1.1,
            1 if spot_price < 100 else 5
        )
        
        # Further OTM: wider intervals
        otm_low = np.arange(
            max(5, spot_price * 0.7),
            spot_price * 0.9,
            5 if spot_price < 100 else 10
        )
        
        otm_high = np.arange(
            spot_price * 1.1,
            spot_price * 1.4,
            5 if spot_price < 100 else 10
        )
        
        # Combine and clean
        all_strikes = np.concatenate([otm_low, atm_range, otm_high])
        all_strikes = np.unique(np.round(all_strikes).astype(int))
        
        return all_strikes[all_strikes > 0]
    
    def _generate_iv_with_smile(self, spot_price: float, strike: float, 
                              time_to_expiry: float, option_type: str) -> float:
        """Generate realistic implied volatility with smile/skew"""
        
        # Base volatility
        base_vol = 0.25
        
        # Moneyness effect (volatility smile/skew)
        if option_type == 'call':
            moneyness = np.log(spot_price / strike)
        else:
            moneyness = np.log(strike / spot_price)
        
        # Volatility smile parameters
        smile_curvature = 0.15  # How pronounced the smile is
        skew_factor = -0.1      # Negative skew (higher IV for OTM puts)
        
        # Calculate smile effect
        smile_effect = smile_curvature * moneyness**2 + skew_factor * moneyness
        
        # Time to expiry effect (term structure)
        if time_to_expiry < 0.25:  # Less than 3 months
            time_effect = 0.05 / time_to_expiry  # Higher vol for short-term
        else:
            time_effect = 0.02
        
        # Random noise for realism
        noise = np.random.normal(0, 0.02)
        
        # Combine effects
        iv = base_vol + smile_effect + time_effect + noise
        
        # Keep within reasonable bounds
        return max(0.05, min(2.0, iv))
    
    def _add_market_microstructure(self, theoretical_price: float, strike: float, 
                                 spot_price: float, time_to_expiry: float, 
                                 option_type: str) -> Dict:
        """Add realistic market microstructure effects"""
        
        # Bid-ask spread based on liquidity
        moneyness = abs(np.log(spot_price / strike))
        
        # Spread widens for OTM options and short time to expiry
        base_spread_pct = 0.02  # 2% base spread
        moneyness_penalty = moneyness * 0.01  # Additional spread for OTM
        time_penalty = max(0, (0.1 - time_to_expiry) * 0.05) if time_to_expiry < 0.1 else 0
        
        spread_pct = base_spread_pct + moneyness_penalty + time_penalty
        spread_pct = min(spread_pct, 0.15)  # Cap at 15%
        
        # Market price with slight deviation from theoretical
        market_noise = np.random.uniform(-0.02, 0.02)  # ±2% noise
        market_price = max(0.01, theoretical_price * (1 + market_noise))
        
        # Bid-ask around market price
        half_spread = market_price * spread_pct / 2
        bid = max(0.01, market_price - half_spread)
        ask = market_price + half_spread
        
        # Volume and open interest based on moneyness and time
        # ATM options have higher volume
        volume_factor = np.exp(-2 * moneyness**2)  # Gaussian around ATM
        base_volume = int(np.random.exponential(200) * volume_factor)
        volume = max(0, base_volume)
        
        # Open interest tends to be higher for liquid strikes
        oi_factor = volume_factor * (1 + time_to_expiry)  # Longer time = more OI
        open_interest = int(np.random.exponential(1000) * oi_factor)
        
        # Last trade time (some options don't trade every day)
        if np.random.random() > 0.3:  # 70% chance of recent trade
            hours_ago = int(np.random.exponential(4))  # Average 4 hours ago
            last_trade = datetime.now() - timedelta(hours=hours_ago)
        else:
            days_ago = int(np.random.exponential(2)) + 1  # 1-7 days ago typically
            last_trade = datetime.now() - timedelta(days=days_ago)
        
        return {
            'market_price': round(market_price, 2),
            'bid': round(bid, 2),
            'ask': round(ask, 2),
            'volume': volume,
            'open_interest': open_interest,
            'last_trade': last_trade.isoformat(),
            'bid_ask_spread': round(ask - bid, 2),
            'spread_pct': round(spread_pct * 100, 2)
        }
    
    def generate_historical_iv_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate historical implied volatility data for analysis"""
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        historical_data = []
        base_iv = 0.25
        
        for date in dates:
            # Add trend and noise to IV
            trend = 0.001 * np.sin(2 * np.pi * len(historical_data) / 20)  # 20-day cycle
            noise = np.random.normal(0, 0.02)
            
            current_iv = max(0.05, base_iv + trend + noise)
            
            historical_data.append({
                'date': date.date(),
                'symbol': symbol,
                'implied_volatility': round(current_iv, 4),
                'realized_volatility': round(current_iv * np.random.uniform(0.8, 1.2), 4)
            })
            
            # Update base for next day (mean reversion)
            base_iv = base_iv * 0.95 + current_iv * 0.05
        
        return pd.DataFrame(historical_data)
