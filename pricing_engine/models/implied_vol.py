import numpy as np
from scipy.optimize import brentq, newton
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from .black_scholes import BlackScholesModel

class ImpliedVolatilitySolver:
    """Enhanced implied volatility calculation with better error handling"""
    
    def __init__(self):
        self.bs_model = BlackScholesModel()
        self.solver_name = "Implied Volatility Solver"
    
    def calculate_iv(self, market_price: float, S: float, K: float, T: float, r: float,
                    option_type: str = "call", method: str = "brent") -> Dict[str, float]:
        """Enhanced IV calculation with better bounds checking"""
        
        if T <= 0:
            return {"iv": np.nan, "error": "Option expired", "iterations": 0}
        
        if market_price <= 0:
            return {"iv": np.nan, "error": "Invalid market price", "iterations": 0}
        
        # Enhanced bounds checking
        if option_type.lower() == "call":
            intrinsic_value = max(0, S - K * np.exp(-r * T))
            max_value = S  # Call can't be worth more than stock
        else:
            intrinsic_value = max(0, K * np.exp(-r * T) - S)
            max_value = K * np.exp(-r * T)  # Put can't be worth more than discounted strike
        
        if market_price < intrinsic_value * 0.99:  # Allow small tolerance
            return {"iv": np.nan, "error": "Market price below intrinsic value", "iterations": 0}
        
        if market_price > max_value * 1.01:  # Allow small tolerance
            return {"iv": np.nan, "error": "Market price above maximum possible value", "iterations": 0}
        
        # Define objective function with better error handling
        def objective(vol):
            try:
                if vol <= 0.001 or vol > 5.0:
                    return float('inf')
                
                if option_type.lower() == "call":
                    theoretical_price = self.bs_model.call_price(S, K, T, r, vol)
                else:
                    theoretical_price = self.bs_model.put_price(S, K, T, r, vol)
                
                return theoretical_price - market_price
            except:
                return float('inf')
        
        try:
            # Enhanced initial guess based on intrinsic value
            if market_price > intrinsic_value:
                # Better initial bounds
                vol_lower = 0.01
                vol_upper = 3.0
                
                # Check bounds
                if objective(vol_lower) * objective(vol_upper) > 0:
                    # Try to find better bounds
                    test_vols = np.linspace(0.01, 2.0, 10)
                    for v in test_vols:
                        if objective(v) * objective(vol_lower) < 0:
                            vol_upper = v
                            break
                        if objective(v) * objective(vol_upper) < 0:
                            vol_lower = v
                            break
                
                if method == "brent":
                    iv = brentq(objective, vol_lower, vol_upper, maxiter=100, xtol=1e-6)
                    iterations = 100
                elif method == "newton":
                    iv = self._newton_raphson_iv(market_price, S, K, T, r, option_type)
                    iterations = 50
                else:
                    raise ValueError(f"Unknown method: {method}")
            else:
                # Very close to intrinsic, use very low volatility
                iv = 0.01
                iterations = 1
            
            # Calculate final metrics
            if option_type.lower() == "call":
                theoretical_price = self.bs_model.call_price(S, K, T, r, iv)
            else:
                theoretical_price = self.bs_model.put_price(S, K, T, r, iv)
            
            error = abs(theoretical_price - market_price)
            
            return {
                "iv": round(iv, 6),
                "theoretical_price": round(theoretical_price, 4),
                "market_price": round(market_price, 4),
                "pricing_error": round(error, 4),
                "method": method,
                "iterations": iterations,
                "moneyness": round(S/K, 4) if option_type.lower() == "call" else round(K/S, 4),
                "intrinsic_value": round(intrinsic_value, 4)
            }
            
        except Exception as e:
            return {
                "iv": np.nan,
                "error": f"Solver failed: {str(e)}",
                "method": method,
                "iterations": 0
            }
    
    def _newton_raphson_iv(self, market_price: float, S: float, K: float, T: float, r: float,
                          option_type: str, initial_guess: float = 0.3, tolerance: float = 1e-6) -> float:
        """Enhanced Newton-Raphson method"""
        vol = initial_guess
        
        for i in range(50):
            if option_type.lower() == "call":
                price = self.bs_model.call_price(S, K, T, r, vol)
            else:
                price = self.bs_model.put_price(S, K, T, r, vol)
            
            # Calculate vega numerically if needed
            dvol = 0.001
            if option_type.lower() == "call":
                price_up = self.bs_model.call_price(S, K, T, r, vol + dvol)
            else:
                price_up = self.bs_model.put_price(S, K, T, r, vol + dvol)
            
            vega = (price_up - price) / dvol
            
            if abs(vega) < 1e-10:
                break
            
            price_diff = price - market_price
            vol_new = vol - price_diff / vega
            
            if abs(vol_new - vol) < tolerance:
                return max(vol_new, 0.001)
            
            vol = max(vol_new, 0.001)  # Ensure positive volatility
        
        return vol
    
    def analyze_option_chain(self, options_data: List[Dict], S: float, r: float) -> pd.DataFrame:
        """Enhanced option chain analysis with better filtering"""
        results = []
        
        for option in options_data:
            if 'market_price' in option and option['market_price'] > 0:
                iv_result = self.calculate_iv(
                    option['market_price'],
                    S,
                    option['strike'],
                    option.get('time_to_expiry', 0.25),
                    r,
                    option['option_type']
                )
                
                # Only include valid results
                if not np.isnan(iv_result.get('iv', np.nan)):
                    results.append({
                        'strike': option['strike'],
                        'option_type': option['option_type'],
                        'market_price': option['market_price'],
                        'implied_volatility': iv_result.get('iv', np.nan),
                        'pricing_error': iv_result.get('pricing_error', np.nan),
                        'moneyness': iv_result.get('moneyness', np.nan),
                        'time_to_expiry': option.get('time_to_expiry', 0.25),
                        'intrinsic_value': iv_result.get('intrinsic_value', 0)
                    })
        
        return pd.DataFrame(results)
    
    def create_volatility_smile(self, iv_analysis: pd.DataFrame, option_type: str = "call") -> go.Figure:
        """Enhanced volatility smile with better formatting"""
        
        # Filter and clean data
        data = iv_analysis[iv_analysis['option_type'] == option_type].copy()
        data = data.dropna(subset=['implied_volatility'])
        data = data[data['implied_volatility'] > 0]  # Remove invalid values
        data = data.sort_values('strike')
        
        fig = go.Figure()
        
        if data.empty:
            fig.update_layout(
                title=f'No Valid Data for {option_type.title()} Options',
                xaxis_title='Strike Price ($)',
                yaxis_title='Implied Volatility (%)',
                height=400,
                annotations=[
                    dict(
                        text="No valid implied volatility data available",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=16)
                    )
                ]
            )
            return fig
        
        # Create hover text with detailed info
        hover_text = []
        for _, row in data.iterrows():
            hover_text.append(
                f"Strike: ${row['strike']:.2f}<br>" +
                f"IV: {row['implied_volatility']*100:.2f}%<br>" +
                f"Market Price: ${row['market_price']:.2f}<br>" +
                f"Moneyness: {row['moneyness']:.3f}"
            )
        
        fig.add_trace(go.Scatter(
            x=data['strike'],
            y=data['implied_volatility'] * 100,
            mode='markers+lines',
            name=f'{option_type.title()} IV',
            line=dict(color='blue', width=3),
            marker=dict(size=10, color='blue'),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Implied Volatility Smile - {option_type.title()} Options',
            xaxis_title='Strike Price ($)',
            yaxis_title='Implied Volatility (%)',
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            hovermode='closest'
        )
        
        return fig
