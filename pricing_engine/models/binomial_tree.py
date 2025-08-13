import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import time

class BinomialTreeModel:
    """Cox-Ross-Rubinstein Binomial Tree for European and American Options"""
    
    def __init__(self, n_steps: int = 100):
        self.n_steps = n_steps
        self.model_name = "Binomial Tree"
    
    def calculate_parameters(self, T: float, r: float, sigma: float, n_steps: int = None) -> Dict[str, float]:
        """Calculate binomial tree parameters"""
        if n_steps is None:
            n_steps = self.n_steps
            
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1/u  # Down factor (ensures recombining tree)
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        
        return {
            "dt": dt,
            "u": u,
            "d": d,
            "p": p,
            "n_steps": n_steps
        }
    
    def build_stock_tree(self, S0: float, params: Dict[str, float]) -> np.ndarray:
        """Build stock price tree"""
        n_steps = params["n_steps"]
        u, d = params["u"], params["d"]
        
        # Initialize tree matrix
        tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Fill the tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        
        return tree
    
    def price_european_option(self, S: float, K: float, T: float, r: float, sigma: float, 
                            option_type: str = "call", n_steps: int = None) -> Dict[str, float]:
        """Price European option using binomial tree"""
        start_time = time.time()
        
        if n_steps is None:
            n_steps = self.n_steps
        
        params = self.calculate_parameters(T, r, sigma, n_steps)
        stock_tree = self.build_stock_tree(S, params)
        
        u, d, p, dt = params["u"], params["d"], params["p"], params["dt"]
        discount = np.exp(-r * dt)
        
        # Initialize option value tree
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        
        # Calculate terminal payoffs
        for j in range(n_steps + 1):
            if option_type.lower() == "call":
                option_tree[j, n_steps] = max(0, stock_tree[j, n_steps] - K)
            else:
                option_tree[j, n_steps] = max(0, K - stock_tree[j, n_steps])
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = discount * (p * option_tree[j, i + 1] + 
                                              (1 - p) * option_tree[j + 1, i + 1])
        
        execution_time = time.time() - start_time
        
        return {
            "price": round(option_tree[0, 0], 6),
            "execution_time_ms": round(execution_time * 1000, 2),
            "n_steps": n_steps,
            "stock_tree": stock_tree,
            "option_tree": option_tree,
            "params": params
        }
    
    def price_american_option(self, S: float, K: float, T: float, r: float, sigma: float, 
                            option_type: str = "call", n_steps: int = None) -> Dict[str, float]:
        """Price American option with early exercise"""
        start_time = time.time()
        
        if n_steps is None:
            n_steps = self.n_steps
        
        params = self.calculate_parameters(T, r, sigma, n_steps)
        stock_tree = self.build_stock_tree(S, params)
        
        u, d, p, dt = params["u"], params["d"], params["p"], params["dt"]
        discount = np.exp(-r * dt)
        
        # Initialize option value and exercise decision trees
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        exercise_tree = np.zeros((n_steps + 1, n_steps + 1), dtype=bool)
        
        # Calculate terminal payoffs
        for j in range(n_steps + 1):
            if option_type.lower() == "call":
                option_tree[j, n_steps] = max(0, stock_tree[j, n_steps] - K)
            else:
                option_tree[j, n_steps] = max(0, K - stock_tree[j, n_steps])
        
        # Backward induction with early exercise check
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                continuation_value = discount * (p * option_tree[j, i + 1] + 
                                              (1 - p) * option_tree[j + 1, i + 1])
                
                # Intrinsic value (immediate exercise)
                if option_type.lower() == "call":
                    intrinsic_value = max(0, stock_tree[j, i] - K)
                else:
                    intrinsic_value = max(0, K - stock_tree[j, i])
                
                # American option: take maximum of continuation and exercise
                if intrinsic_value > continuation_value:
                    option_tree[j, i] = intrinsic_value
                    exercise_tree[j, i] = True
                else:
                    option_tree[j, i] = continuation_value
                    exercise_tree[j, i] = False
        
        execution_time = time.time() - start_time
        
        # Calculate early exercise premium
        european_price = self.price_european_option(S, K, T, r, sigma, option_type, n_steps)["price"]
        early_exercise_premium = option_tree[0, 0] - european_price
        
        return {
            "price": round(option_tree[0, 0], 6),
            "execution_time_ms": round(execution_time * 1000, 2),
            "n_steps": n_steps,
            "stock_tree": stock_tree,
            "option_tree": option_tree,
            "exercise_tree": exercise_tree,
            "params": params,
            "early_exercise_premium": round(early_exercise_premium, 6)
        }
    
    def convergence_analysis(self, S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = "call", step_sizes: List[int] = None) -> pd.DataFrame:
        """Analyze convergence as number of steps increases"""
        if step_sizes is None:
            step_sizes = [10, 25, 50, 100, 200, 500]
        
        results = []
        for n in step_sizes:
            european_result = self.price_european_option(S, K, T, r, sigma, option_type, n)
            american_result = self.price_american_option(S, K, T, r, sigma, option_type, n)
            
            results.append({
                "n_steps": n,
                "european_price": european_result["price"],
                "american_price": american_result["price"],
                "early_exercise_premium": american_result["early_exercise_premium"],
                "execution_time_ms": european_result["execution_time_ms"]
            })
        
        return pd.DataFrame(results)
    
    def create_tree_visualization(self, stock_tree: np.ndarray, option_tree: np.ndarray, 
                                params: Dict[str, float], max_display_steps: int = 4) -> go.Figure:
        """Create Plotly visualization of binomial tree"""
        n_steps = min(params["n_steps"], max_display_steps)
        
        # Create tree layout positions
        x_positions = []
        y_positions = []
        stock_prices = []
        option_values = []
        
        for i in range(n_steps + 1):
            for j in range(i + 1):
                x_positions.append(i)
                y_positions.append(i/2 - j)
                stock_prices.append(stock_tree[j, i])
                option_values.append(option_tree[j, i])
        
        # Create the plot
        fig = go.Figure()
        
        # Add connecting lines first (so they appear behind nodes)
        for i in range(n_steps):
            for j in range(i + 1):
                # Up connection
                fig.add_trace(go.Scatter(
                    x=[i, i + 1],
                    y=[i/2 - j, (i + 1)/2 - j],
                    mode='lines',
                    line=dict(color='lightgreen', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Down connection
                fig.add_trace(go.Scatter(
                    x=[i, i + 1],
                    y=[i/2 - j, (i + 1)/2 - (j + 1)],
                    mode='lines',
                    line=dict(color='lightcoral', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add nodes on top
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            marker=dict(size=25, color='lightblue', line=dict(width=3, color='navy')),
            text=[f'${s:.1f}<br>${v:.2f}' for s, v in zip(stock_prices, option_values)],
            textposition="middle center",
            textfont=dict(size=9, color='black'),
            name="Tree Nodes",
            hovertemplate='<b>Stock Price:</b> $%{customdata[0]:.2f}<br>' +
                         '<b>Option Value:</b> $%{customdata[1]:.2f}<extra></extra>',
            customdata=list(zip(stock_prices, option_values))
        ))
        
        fig.update_layout(
            title=f"Binomial Tree Visualization ({n_steps} steps)",
            xaxis_title="Time Steps",
            yaxis_title="Price Levels",
            showlegend=False,
            height=500,
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            plot_bgcolor='white'
        )
        
        return fig
