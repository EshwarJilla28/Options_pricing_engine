import numpy as np
from scipy.stats import norm
from typing import Dict, Union
import math

class BlackScholesModel:
    """Enhanced Black-Scholes implementation with comprehensive Greeks"""
    
    def __init__(self):
        self.model_name = "Black-Scholes"
    
    def calculate_d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> tuple:
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def call_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return max(call_price, 0)
    
    def put_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price"""
        if T <= 0:
            return max(K - S, 0)
        
        d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
        
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        return max(put_price, 0)
    
    def calculate_all_prices(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """Calculate both call and put prices"""
        return {
            "call_price": self.call_price(S, K, T, r, sigma),
            "put_price": self.put_price(S, K, T, r, sigma)
        }
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = "call", max_iter: int = 100) -> float:
        """Calculate implied volatility using Brent method"""
        from scipy.optimize import brentq
        
        def objective(vol):
            if option_type.lower() == "call":
                theoretical_price = self.call_price(S, K, T, r, vol)
            else:
                theoretical_price = self.put_price(S, K, T, r, vol)
            return theoretical_price - market_price
        
        try:
            iv = brentq(objective, 0.01, 5.0, maxiter=max_iter)
            return iv
        except ValueError:
            return np.nan
