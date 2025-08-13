import numpy as np
from typing import Dict, Tuple
import time

class MonteCarloModel:
    """Monte Carlo option pricing with variance reduction techniques"""
    
    def __init__(self, n_simulations: int = 10000, n_steps: int = 100):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.model_name = "Monte Carlo"
    
    def simulate_paths(self, S0: float, T: float, r: float, sigma: float, 
                      n_paths: int = None) -> np.ndarray:
        """Generate stock price paths using geometric Brownian motion"""
        if n_paths is None:
            n_paths = self.n_simulations
        
        dt = T / self.n_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((self.n_steps, n_paths))
        
        # Calculate drift and diffusion
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Generate paths
        returns = drift + diffusion * Z
        log_paths = np.cumsum(returns, axis=0)
        paths = S0 * np.exp(log_paths)
        
        return paths
    
    def european_option_price(self, S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = "call") -> Dict[str, float]:
        """Price European options using Monte Carlo"""
        start_time = time.time()
        
        # Generate final stock prices
        final_prices = self._generate_final_prices(S, T, r, sigma)
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount back to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate confidence interval
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error * np.exp(-r * T)
        
        execution_time = time.time() - start_time
        
        return {
            "price": round(option_price, 4),
            "std_error": round(std_error, 4),
            "confidence_interval": round(confidence_interval, 4),
            "execution_time_ms": round(execution_time * 1000, 2),
            "simulations": self.n_simulations
        }
    
    def _generate_final_prices(self, S0: float, T: float, r: float, sigma: float) -> np.ndarray:
        """Generate final stock prices at maturity"""
        # Use antithetic variance reduction
        n_half = self.n_simulations // 2
        
        Z = np.random.standard_normal(n_half)
        Z_anti = np.concatenate([Z, -Z])
        
        # Geometric Brownian Motion final prices
        final_prices = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_anti)
        
        return final_prices
    
    def calculate_greeks_mc(self, S: float, K: float, T: float, r: float, 
                          sigma: float, option_type: str = "call") -> Dict[str, float]:
        """Calculate Greeks using finite difference method"""
        # Bump size for finite differences
        dS = 0.01 * S
        dT = 1/365  # 1 day
        dr = 0.0001  # 1 basis point
        dsigma = 0.01  # 1%
        
        # Base price
        base_price = self.european_option_price(S, K, T, r, sigma, option_type)["price"]
        
        # Delta (sensitivity to spot price)
        price_up = self.european_option_price(S + dS, K, T, r, sigma, option_type)["price"]
        price_down = self.european_option_price(S - dS, K, T, r, sigma, option_type)["price"]
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma (second derivative wrt spot)
        gamma = (price_up - 2 * base_price + price_down) / (dS**2)
        
        # Theta (time decay)
        if T > dT:
            price_theta = self.european_option_price(S, K, T - dT, r, sigma, option_type)["price"]
            theta = -(price_theta - base_price) / dT
        else:
            theta = 0
        
        # Vega (volatility sensitivity)
        price_vega = self.european_option_price(S, K, T, r, sigma + dsigma, option_type)["price"]
        vega = (price_vega - base_price) / dsigma
        
        # Rho (interest rate sensitivity)
        price_rho = self.european_option_price(S, K, T, r + dr, sigma, option_type)["price"]
        rho = (price_rho - base_price) / dr
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4)
        }
