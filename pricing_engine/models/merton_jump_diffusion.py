import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.stats import norm
import time
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Model for Option Pricing
    
    Extends Black-Scholes by incorporating sudden price jumps via Poisson process.
    Captures market crashes, earnings announcements, and other discrete events.
    """
    
    def __init__(self, jump_intensity: float = 0.1, jump_mean: float = -0.05, 
                 jump_std: float = 0.15, max_jumps: int = 50):
        """
        Initialize Merton Jump Diffusion Model
        
        Parameters:
        -----------
        jump_intensity : float
            λ - Average number of jumps per year (default: 0.1 = 1 jump per 10 years)
        jump_mean : float  
            μj - Mean of log-normal jump size distribution (default: -0.05 = -5% average jump)
        jump_std : float
            σj - Standard deviation of jump size distribution (default: 0.15)
        max_jumps : int
            Maximum number of jumps to consider in series expansion (default: 50)
        """
        self.lambda_jump = jump_intensity
        self.mu_jump = jump_mean  
        self.sigma_jump = jump_std
        self.max_jumps = max_jumps
        
        # Precompute jump statistics
        self.k_bar = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        self.model_name = "Merton Jump Diffusion"
    
    def european_option_price(self, S0: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = "call") -> Dict:
        """
        Calculate European option price using Merton's series expansion
        
        The Merton model uses an infinite series where each term represents
        the Black-Scholes price conditional on exactly n jumps occurring.
        """
        start_time = time.time()
        
        # Adjust drift for jump component
        r_adjusted = r - self.lambda_jump * self.k_bar
        
        total_price = 0.0
        individual_terms = []
        
        # Series expansion: sum over possible number of jumps
        for n in range(self.max_jumps):
            # Probability of exactly n jumps
            jump_prob = self._poisson_probability(n, self.lambda_jump * T)
            
            if jump_prob < 1e-10:  # Skip negligible terms
                break
            
            # Adjusted parameters for n jumps
            sigma_n = np.sqrt(sigma**2 + n * self.sigma_jump**2 / T)
            r_n = r_adjusted + n * (self.mu_jump + 0.5 * self.sigma_jump**2) / T
            
            # Black-Scholes price conditional on n jumps
            if option_type.lower() == "call":
                bs_price = self._black_scholes_call(S0, K, T, r_n, sigma_n)
            else:
                bs_price = self._black_scholes_put(S0, K, T, r_n, sigma_n)
            
            term_contribution = jump_prob * bs_price
            total_price += term_contribution
            
            individual_terms.append({
                'n_jumps': n,
                'probability': jump_prob,
                'bs_price': bs_price,
                'contribution': term_contribution,
                'cumulative_contribution': total_price
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'price': round(total_price, 6),
            'execution_time_ms': round(execution_time, 4),
            'method': f'Series expansion ({len(individual_terms)} terms)',
            'jump_parameters': {
                'lambda': self.lambda_jump,
                'mu_jump': self.mu_jump,
                'sigma_jump': self.sigma_jump,
                'k_bar': self.k_bar
            },
            'series_breakdown': individual_terms,
            'convergence_info': {
                'terms_used': len(individual_terms),
                'final_term_contribution': individual_terms[-1]['contribution'] if individual_terms else 0,
                'relative_final_contribution': individual_terms[-1]['contribution'] / total_price if individual_terms and total_price > 0 else 0
            }
        }
    
    def monte_carlo_simulation(self, S0: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = "call", 
                             n_simulations: int = 100000, n_steps: int = 252) -> Dict:
        """
        Monte Carlo simulation with explicit jump modeling
        
        Simulates stock price paths with both continuous diffusion and discrete jumps.
        More intuitive than series expansion but computationally intensive.
        """
        start_time = time.time()
        
        dt = T / n_steps
        r_adjusted = r - self.lambda_jump * self.k_bar
        
        # Initialize arrays
        final_prices = np.zeros(n_simulations)
        jump_counts = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            S = S0
            n_jumps = 0
            
            for step in range(n_steps):
                # Continuous diffusion component
                dW = np.random.normal(0, np.sqrt(dt))
                S *= np.exp((r_adjusted - 0.5 * sigma**2) * dt + sigma * dW)
                
                # Jump component
                if np.random.random() < self.lambda_jump * dt:
                    jump_size = np.random.normal(self.mu_jump, self.sigma_jump)
                    S *= np.exp(jump_size)
                    n_jumps += 1
            
            final_prices[i] = S
            jump_counts[i] = n_jumps
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
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
                'avg_final_price': round(np.mean(final_prices), 2),
                'avg_jumps_per_path': round(np.mean(jump_counts), 2),
                'max_jumps_observed': int(np.max(jump_counts)),
                'paths_with_jumps': int(np.sum(jump_counts > 0)),
                'jump_frequency': round(np.sum(jump_counts > 0) / n_simulations, 4)
            }
        }
    
    def _poisson_probability(self, n: int, lambda_t: float) -> float:
        """Calculate P(N(t) = n) for Poisson process"""
        if lambda_t == 0:
            return 1.0 if n == 0 else 0.0
        return (lambda_t**n * np.exp(-lambda_t)) / factorial(n)
    
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call option formula"""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def _black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put option formula"""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def parameter_sensitivity_analysis(self, S0: float, K: float, T: float, r: float,
                                     sigma: float, option_type: str = "call") -> Dict:
        """
        Analyze how option price changes with jump parameters
        
        Educational tool to understand jump risk impact on option pricing.
        """
        base_price = self.european_option_price(S0, K, T, r, sigma, option_type)['price']
        
        # Test different jump intensities
        lambda_range = np.linspace(0.0, 0.5, 11)
        lambda_prices = []
        
        original_lambda = self.lambda_jump
        for lam in lambda_range:
            self.lambda_jump = lam
            self.k_bar = np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1
            price = self.european_option_price(S0, K, T, r, sigma, option_type)['price']
            lambda_prices.append(price)
        self.lambda_jump = original_lambda  # Restore
        
        # Test different jump means
        mu_range = np.linspace(-0.2, 0.1, 11)
        mu_prices = []
        
        original_mu = self.mu_jump
        for mu in mu_range:
            self.mu_jump = mu
            self.k_bar = np.exp(mu + 0.5 * self.sigma_jump**2) - 1
            price = self.european_option_price(S0, K, T, r, sigma, option_type)['price']
            mu_prices.append(price)
        self.mu_jump = original_mu  # Restore
        
        # Test different jump volatilities
        sigma_j_range = np.linspace(0.05, 0.3, 11)
        sigma_j_prices = []
        
        original_sigma_j = self.sigma_jump
        for sig_j in sigma_j_range:
            self.sigma_jump = sig_j
            self.k_bar = np.exp(self.mu_jump + 0.5 * sig_j**2) - 1
            price = self.european_option_price(S0, K, T, r, sigma, option_type)['price']
            sigma_j_prices.append(price)
        self.sigma_jump = original_sigma_j  # Restore
        self.k_bar = np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1  # Restore
        
        return {
            'base_price': base_price,
            'jump_intensity_analysis': {
                'lambda_values': lambda_range.tolist(),
                'prices': lambda_prices,
                'price_change': [(p - base_price) / base_price * 100 for p in lambda_prices]
            },
            'jump_mean_analysis': {
                'mu_values': mu_range.tolist(),
                'prices': mu_prices,
                'price_change': [(p - base_price) / base_price * 100 for p in mu_prices]
            },
            'jump_volatility_analysis': {
                'sigma_j_values': sigma_j_range.tolist(),
                'prices': sigma_j_prices,
                'price_change': [(p - base_price) / base_price * 100 for p in sigma_j_prices]
            }
        }
    
    def create_visualization(self, sensitivity_data: Dict, option_type: str) -> go.Figure:
        """Create interactive visualization of parameter sensitivity"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Jump Intensity (λ) Sensitivity',
                'Jump Mean (μⱼ) Sensitivity', 
                'Jump Volatility (σⱼ) Sensitivity',
                'Price Impact Summary'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Jump intensity sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['jump_intensity_analysis']['lambda_values'],
                y=sensitivity_data['jump_intensity_analysis']['prices'],
                mode='lines+markers',
                name='Price vs λ',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Jump mean sensitivity  
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['jump_mean_analysis']['mu_values'],
                y=sensitivity_data['jump_mean_analysis']['prices'],
                mode='lines+markers',
                name='Price vs μⱼ',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Jump volatility sensitivity
        fig.add_trace(
            go.Scatter(
                x=sensitivity_data['jump_volatility_analysis']['sigma_j_values'],
                y=sensitivity_data['jump_volatility_analysis']['prices'],
                mode='lines+markers',
                name='Price vs σⱼ',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Summary bar chart
        max_changes = [
            max(sensitivity_data['jump_intensity_analysis']['price_change']),
            max(sensitivity_data['jump_mean_analysis']['price_change']),
            max(sensitivity_data['jump_volatility_analysis']['price_change'])
        ]
        
        fig.add_trace(
            go.Bar(
                x=['Jump Intensity', 'Jump Mean', 'Jump Volatility'],
                y=max_changes,
                name='Max Price Impact (%)',
                marker_color=['blue', 'red', 'green']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Merton Jump Diffusion: {option_type.title()} Option Parameter Sensitivity',
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Jump Intensity (λ)", row=1, col=1)
        fig.update_xaxes(title_text="Jump Mean (μⱼ)", row=1, col=2)
        fig.update_xaxes(title_text="Jump Volatility (σⱼ)", row=2, col=1)
        fig.update_xaxes(title_text="Parameter", row=2, col=2)
        
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Option Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Price Change (%)", row=2, col=2)
        
        return fig
