import numpy as np
from scipy.stats import norm
from .black_scholes import BlackScholesModel
from typing import Dict

class GreeksCalculator:
    """Calculate all Greeks for options"""
    
    def __init__(self):
        self.bs_model = BlackScholesModel()
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = "call") -> Dict[str, float]:
        """Calculate all first and second order Greeks"""
        
        if T <= 0:
            return self._zero_greeks()
        
        d1, d2 = self.bs_model.calculate_d1_d2(S, K, T, r, sigma)
        
        # Common calculations
        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        phi_d1 = norm.pdf(d1)
        phi_d2 = norm.pdf(d2)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        
        if option_type.lower() == "call":
            delta = N_d1
            theta = (-S * phi_d1 * sigma / (2 * sqrt_T) - r * K * exp_neg_rT * N_d2) / 365
            rho = K * T * exp_neg_rT * N_d2 / 100
        else:  # put
            delta = N_d1 - 1
            theta = (-S * phi_d1 * sigma / (2 * sqrt_T) + r * K * exp_neg_rT * norm.cdf(-d2)) / 365
            rho = -K * T * exp_neg_rT * norm.cdf(-d2) / 100
        
        # Greeks that are same for calls and puts
        gamma = phi_d1 / (S * sigma * sqrt_T)
        vega = S * phi_d1 * sqrt_T / 100
        
        # Second-order Greeks
        vanna = -phi_d1 * d2 / sigma
        volga = S * phi_d1 * sqrt_T * d1 * d2 / sigma
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4),
            "vanna": round(vanna, 4),
            "volga": round(volga, 4)
        }
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks for expired options"""
        return {
            "delta": 0.0, "gamma": 0.0, "theta": 0.0,
            "vega": 0.0, "rho": 0.0, "vanna": 0.0, "volga": 0.0
        }
