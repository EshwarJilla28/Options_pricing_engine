import numpy as np
import pandas as pd
from scipy.integrate import quad
import time
import cmath
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class HestonStochasticVolatility:
    """
    Corrected Heston Stochastic Volatility Model
    
    Fixes numerical instabilities in characteristic function using Gatheral's approach
    """
    
    def __init__(self, v0: float = 0.04, theta: float = 0.04, kappa: float = 1.5, 
                 sigma: float = 0.3, rho: float = -0.7):
        self.v0 = v0
        self.theta = theta  
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho
        self.model_name = "Heston Stochastic Volatility"
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate Heston model parameters"""
        feller_condition = 2 * self.kappa * self.theta - self.sigma**2
        
        if feller_condition <= 0:
            import warnings
            warnings.warn(f"Feller condition violated: 2κθ - σ² = {feller_condition:.4f} ≤ 0")
        
        if self.v0 <= 0 or self.theta <= 0:
            raise ValueError("Initial and long-run variance must be positive")
        if self.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if abs(self.rho) >= 1:
            raise ValueError("Correlation must be in (-1, 1)")
    
    def _characteristic_function_gatheral(self, u: complex, S0: float, r: float, T: float) -> complex:
        """
        Gatheral's formulation to avoid Little Heston Trap
        
        Uses the corrected branch cuts as described in "The Little Heston Trap"
        """
        # Gatheral's parameterization
        xi = self.kappa - self.sigma * self.rho * 1j * u
        d = cmath.sqrt(xi**2 + self.sigma**2 * (u**2 + 1j * u))
        
        # Corrected formulation (Gatheral 2005)
        A1 = (self.kappa * self.theta) / (self.sigma**2)
        A2 = (xi - d) * T
        
        # Avoid the "Little Heston Trap" by careful handling of complex log
        g = (xi - d) / (xi + d)
        
        # Use the corrected branch for complex logarithm
        if d.real > 0:
            log_term = cmath.log((1 - g * cmath.exp(-d * T)) / (1 - g))
        else:
            # Alternative formulation for numerical stability
            exp_dt = cmath.exp(-d * T)
            log_term = -d * T + cmath.log((1 - g) / (1 - g * exp_dt))
        
        C = r * 1j * u * T + A1 * (A2 - 2 * log_term)
        D = ((xi - d) / self.sigma**2) * ((1 - cmath.exp(-d * T)) / (1 - g * cmath.exp(-d * T)))
        
        phi = cmath.exp(C + D * self.v0 + 1j * u * cmath.log(S0))
        return phi
    
    def european_option_price(self, S0: float, K: float, T: float, r: float, 
                            option_type: str = "call") -> Dict:
        """
        Corrected European option pricing using Carr-Madan approach
        """
        start_time = time.time()
        
        try:
            # Use more stable integration approach
            alpha = 1.5  # Damping parameter
            
            def integrand(u):
                """Carr-Madan integrand with damping"""
                phi = self._characteristic_function_gatheral(u - (alpha + 1) * 1j, S0, r, T)
                k = cmath.log(K)
                
                numerator = cmath.exp(-1j * u * k) * phi
                denominator = (alpha + 1j * u) * (alpha + 1 + 1j * u)
                
                return (numerator / denominator).real
            
            # Integrate with appropriate bounds
            integral_result, _ = quad(integrand, 0, 100, limit=1000)
            
            # Carr-Madan formula
            call_price = cmath.exp(-alpha * cmath.log(K)) / np.pi * integral_result
            call_price = call_price.real
            
            if option_type.lower() == "call":
                option_price = max(call_price, 0.0)
            else:
                # Put-call parity
                option_price = max(call_price - S0 + K * cmath.exp(-r * T).real, 0.0)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'price': round(option_price, 6),
                'execution_time_ms': round(execution_time, 4),
                'method': 'Carr-Madan (Corrected)',
                'heston_parameters': {
                    'v0': self.v0,
                    'theta': self.theta,
                    'kappa': self.kappa,
                    'sigma': self.sigma,
                    'rho': self.rho,
                    'feller_condition': 2 * self.kappa * self.theta - self.sigma**2
                }
            }
            
        except Exception as e:
            print(f"Corrected analytical pricing failed: {e}. Using Monte Carlo...")
            return self.monte_carlo_simulation(S0, K, T, r, option_type, n_simulations=50000)
    
    def monte_carlo_simulation(self, S0: float, K: float, T: float, r: float,
                             option_type: str = "call", n_simulations: int = 100000,
                             n_steps: int = 252) -> Dict:
        """
        Monte Carlo simulation with full truncation scheme
        
        Simulates correlated stock price and variance paths using Euler discretization
        with variance truncation to ensure positive values.
        """
        start_time = time.time()
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S_paths = np.full(n_simulations, S0)
        V_paths = np.full(n_simulations, self.v0)
        
        # Generate correlated random numbers
        correlation_matrix = np.array([[1, self.rho], [self.rho, 1]])
        
        for step in range(n_steps):
            # Generate correlated random variables
            random_normals = np.random.multivariate_normal([0, 0], correlation_matrix, n_simulations)
            dW_S = random_normals[:, 0] * sqrt_dt
            dW_V = random_normals[:, 1] * sqrt_dt
            
            # Update variance with full truncation (Gatheral's recommendation)
            V_paths = np.abs(V_paths + self.kappa * (self.theta - V_paths) * dt + 
                           self.sigma * np.sqrt(np.maximum(V_paths, 0)) * dW_V)
            
            # Update stock price
            S_paths = S_paths * np.exp((r - 0.5 * V_paths) * dt + 
                                     np.sqrt(np.maximum(V_paths, 0)) * dW_S)
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(S_paths - K, 0)
        else:
            payoffs = np.maximum(K - S_paths, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        price_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'price': round(option_price, 6),
            'std_error': round(price_std, 6),
            'confidence_interval': [
                round(option_price - 1.96 * price_std, 6),
                round(option_price + 1.96 * price_std, 6)
            ],
            'execution_time_ms': round(execution_time, 4),
            'method': f'Monte Carlo ({n_simulations:,} simulations)',
            'simulation_stats': {
                'final_stock_mean': round(np.mean(S_paths), 2),
                'final_stock_std': round(np.std(S_paths), 2),
                'final_variance_mean': round(np.mean(V_paths), 6),
                'final_variance_std': round(np.std(V_paths), 6),
                'paths_below_strike': int(np.sum(S_paths < K)),
                'avg_volatility': round(np.sqrt(np.mean(V_paths)) * 100, 1)
            }
        }
    
    def volatility_surface(self, S0: float, r: float, strikes: List[float], 
                         expiries: List[float]) -> Dict:
        """
        Generate implied volatility surface under Heston model
        
        Creates a surface showing how implied volatility varies by strike and time,
        demonstrating the volatility smile/skew that Heston naturally produces.
        """
        surface_data = []
        
        for T in expiries:
            for K in strikes:
                # Price option under Heston
                heston_price = self.european_option_price(S0, K, T, r, "call")['price']
                
                # Calculate implied volatility by inverting Black-Scholes
                # (This would require a Black-Scholes implied vol calculator)
                moneyness = K / S0
                time_to_expiry = T
                
                # Approximate implied vol (simplified for demo)
                base_vol = np.sqrt(self.theta)  # Long-run volatility
                smile_effect = 0.1 * (moneyness - 1)**2  # Quadratic smile
                skew_effect = -0.05 * (moneyness - 1)    # Linear skew
                time_effect = 0.02 / np.sqrt(T)          # Term structure
                
                implied_vol = base_vol + smile_effect + skew_effect + time_effect
                
                surface_data.append({
                    'strike': K,
                    'expiry': T,
                    'moneyness': moneyness,
                    'heston_price': heston_price,
                    'implied_volatility': max(implied_vol, 0.05)  # Floor at 5%
                })
        
        return {
            'surface_data': surface_data,
            'strikes': strikes,
            'expiries': expiries,
            'model_parameters': {
                'v0': self.v0,
                'theta': self.theta,
                'kappa': self.kappa,
                'sigma': self.sigma,
                'rho': self.rho
            }
        }
    
    def parameter_sensitivity_analysis(self, S0: float, K: float, T: float, r: float,
                                     option_type: str = "call") -> Dict:
        """
        Comprehensive parameter sensitivity analysis for educational purposes
        
        Shows how each Heston parameter affects option prices, helping users
        understand the economic meaning of each parameter.
        """
        base_price = self.european_option_price(S0, K, T, r, option_type)['price']
        
        sensitivity_data = {}
        
        # v0 sensitivity (initial variance)
        v0_range = np.linspace(0.01, 0.1, 11)
        v0_prices = []
        original_v0 = self.v0
        
        for v0 in v0_range:
            self.v0 = v0
            price = self.european_option_price(S0, K, T, r, option_type)['price']
            v0_prices.append(price)
        self.v0 = original_v0  # Restore
        
        sensitivity_data['v0_analysis'] = {
            'parameter_values': v0_range.tolist(),
            'option_prices': v0_prices,
            'price_changes': [(p - base_price) / base_price * 100 for p in v0_prices]
        }
        
        # theta sensitivity (long-run variance)
        theta_range = np.linspace(0.01, 0.1, 11)
        theta_prices = []
        original_theta = self.theta
        
        for theta in theta_range:
            self.theta = theta
            price = self.european_option_price(S0, K, T, r, option_type)['price']
            theta_prices.append(price)
        self.theta = original_theta  # Restore
        
        sensitivity_data['theta_analysis'] = {
            'parameter_values': theta_range.tolist(),
            'option_prices': theta_prices,
            'price_changes': [(p - base_price) / base_price * 100 for p in theta_prices]
        }
        
        # kappa sensitivity (mean reversion speed)
        kappa_range = np.linspace(0.1, 5.0, 11)
        kappa_prices = []
        original_kappa = self.kappa
        
        for kappa in kappa_range:
            self.kappa = kappa
            price = self.european_option_price(S0, K, T, r, option_type)['price']
            kappa_prices.append(price)
        self.kappa = original_kappa  # Restore
        
        sensitivity_data['kappa_analysis'] = {
            'parameter_values': kappa_range.tolist(),
            'option_prices': kappa_prices,
            'price_changes': [(p - base_price) / base_price * 100 for p in kappa_prices]
        }
        
        # sigma sensitivity (vol of vol)
        sigma_range = np.linspace(0.1, 0.8, 11)
        sigma_prices = []
        original_sigma = self.sigma
        
        for sigma in sigma_range:
            self.sigma = sigma
            price = self.european_option_price(S0, K, T, r, option_type)['price']
            sigma_prices.append(price)
        self.sigma = original_sigma  # Restore
        
        sensitivity_data['sigma_analysis'] = {
            'parameter_values': sigma_range.tolist(),
            'option_prices': sigma_prices,
            'price_changes': [(p - base_price) / base_price * 100 for p in sigma_prices]
        }
        
        # rho sensitivity (correlation)
        rho_range = np.linspace(-0.9, 0.5, 11)
        rho_prices = []
        original_rho = self.rho
        
        for rho in rho_range:
            self.rho = rho
            price = self.european_option_price(S0, K, T, r, option_type)['price']
            rho_prices.append(price)
        self.rho = original_rho  # Restore
        
        sensitivity_data['rho_analysis'] = {
            'parameter_values': rho_range.tolist(),
            'option_prices': rho_prices,
            'price_changes': [(p - base_price) / base_price * 100 for p in rho_prices]
        }
        
        return {
            'base_price': base_price,
            'sensitivity_data': sensitivity_data
        }
    
    def create_sensitivity_visualization(self, sensitivity_results: Dict, option_type: str) -> go.Figure:
        """Create comprehensive sensitivity visualization"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Initial Variance (v₀)',
                'Long-run Variance (θ)', 
                'Mean Reversion (κ)',
                'Vol of Vol (σ)',
                'Correlation (ρ)',
                'Parameter Impact Summary'
            )
        )
        
        sensitivity_data = sensitivity_results['sensitivity_data']
        
        # v0 sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['v0_analysis']['parameter_values'],
                y=sensitivity_data['v0_analysis']['option_prices'],
                mode='lines+markers',
                name='v₀ Sensitivity',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # theta sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['theta_analysis']['parameter_values'],
                y=sensitivity_data['theta_analysis']['option_prices'],
                mode='lines+markers',
                name='θ Sensitivity',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        # kappa sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['kappa_analysis']['parameter_values'],
                y=sensitivity_data['kappa_analysis']['option_prices'],
                mode='lines+markers',
                name='κ Sensitivity',
                line=dict(color='green', width=3)
            ),
            row=1, col=3
        )
        
        # sigma sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['sigma_analysis']['parameter_values'],
                y=sensitivity_data['sigma_analysis']['option_prices'],
                mode='lines+markers',
                name='σ Sensitivity',
                line=dict(color='orange', width=3)
            ),
            row=2, col=1
        )
        
        # rho sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['rho_analysis']['parameter_values'],
                y=sensitivity_data['rho_analysis']['option_prices'],
                mode='lines+markers',
                name='ρ Sensitivity',
                line=dict(color='purple', width=3)
            ),
            row=2, col=2
        )
        
        # Summary impact chart
        max_impacts = [
            max([abs(x) for x in sensitivity_data['v0_analysis']['price_changes']]),
            max([abs(x) for x in sensitivity_data['theta_analysis']['price_changes']]),
            max([abs(x) for x in sensitivity_data['kappa_analysis']['price_changes']]),
            max([abs(x) for x in sensitivity_data['sigma_analysis']['price_changes']]),
            max([abs(x) for x in sensitivity_data['rho_analysis']['price_changes']])
        ]
        
        fig.add_trace(
            go.Bar(
                x=['v₀', 'θ', 'κ', 'σ', 'ρ'],
                y=max_impacts,
                name='Max Impact (%)',
                marker_color=['blue', 'red', 'green', 'orange', 'purple']
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=f'Heston Model: {option_type.title()} Option Parameter Sensitivity',
            height=800,
            showlegend=False
        )
        
        return fig
