import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

class ModelValidator:
    """Comprehensive model validation and comparison tools"""
    
    def __init__(self, bs_model, mc_model, binomial_model, greeks_calc):
        self.bs_model = bs_model
        self.mc_model = mc_model
        self.binomial_model = binomial_model
        self.greeks_calc = greeks_calc
    
    def comprehensive_model_comparison(self, S: float, K: float, T: float, 
                                     r: float, sigma: float, option_type: str = "call") -> Dict:
        """Run comprehensive comparison across all models"""
        
        results = {}
        
        # Black-Scholes (reference)
        start_time = time.time()
        if option_type.lower() == "call":
            bs_price = self.bs_model.call_price(S, K, T, r, sigma)
        else:
            bs_price = self.bs_model.put_price(S, K, T, r, sigma)
        bs_time = (time.time() - start_time) * 1000
        
        results["black_scholes"] = {
            "price": round(bs_price, 6),
            "execution_time_ms": round(bs_time, 4),
            "method": "analytical",
            "accuracy": "exact"
        }
        
        # Monte Carlo with multiple simulation sizes
        mc_results = []
        simulation_sizes = [1000, 5000, 10000, 50000]
        
        for n_sims in simulation_sizes:
            self.mc_model.n_simulations = n_sims
            mc_result = self.mc_model.european_option_price(S, K, T, r, sigma, option_type.lower())
            
            # Calculate error vs Black-Scholes
            error = abs(mc_result["price"] - bs_price)
            error_pct = (error / bs_price) * 100 if bs_price > 0 else 0
            
            mc_results.append({
                "simulations": n_sims,
                "price": mc_result["price"],
                "execution_time_ms": mc_result["execution_time_ms"],
                "std_error": mc_result["std_error"],
                "confidence_interval": mc_result["confidence_interval"],
                "error_vs_bs": round(error, 6),
                "error_pct": round(error_pct, 4)
            })
        
        results["monte_carlo"] = {
            "convergence_analysis": mc_results,
            "best_result": min(mc_results, key=lambda x: x["error_pct"])
        }
        
        # Binomial Tree with multiple step sizes
        binomial_results = []
        step_sizes = [50, 100, 200, 500, 1000]
        
        for n_steps in step_sizes:
            binomial_result = self.binomial_model.price_european_option(
                S, K, T, r, sigma, option_type.lower(), n_steps
            )
            
            error = abs(binomial_result["price"] - bs_price)
            error_pct = (error / bs_price) * 100 if bs_price > 0 else 0
            
            binomial_results.append({
                "steps": n_steps,
                "price": binomial_result["price"],
                "execution_time_ms": binomial_result["execution_time_ms"],
                "error_vs_bs": round(error, 6),
                "error_pct": round(error_pct, 4)
            })
        
        results["binomial_tree"] = {
            "convergence_analysis": binomial_results,
            "best_result": min(binomial_results, key=lambda x: x["error_pct"])
        }
        
        # Greeks comparison
        greeks = self.greeks_calc.calculate_all_greeks(S, K, T, r, sigma, option_type.lower())
        results["greeks"] = greeks
        
        # Overall summary
        results["summary"] = {
            "reference_price": bs_price,
            "best_mc_error_pct": results["monte_carlo"]["best_result"]["error_pct"],
            "best_binomial_error_pct": results["binomial_tree"]["best_result"]["error_pct"],
            "fastest_method": min(
                [("black_scholes", bs_time),
                 ("monte_carlo", results["monte_carlo"]["best_result"]["execution_time_ms"]),
                 ("binomial", results["binomial_tree"]["best_result"]["execution_time_ms"])],
                key=lambda x: x[1]
            )[0]
        }
        
        return results
    
    def create_convergence_visualization(self, comparison_results: Dict, option_type: str) -> go.Figure:
        """Create comprehensive convergence visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Monte Carlo Convergence", "Binomial Convergence", 
                          "Execution Time Comparison", "Error Analysis"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Monte Carlo convergence
        mc_data = comparison_results["monte_carlo"]["convergence_analysis"]
        mc_sims = [x["simulations"] for x in mc_data]
        mc_prices = [x["price"] for x in mc_data]
        mc_errors = [x["error_pct"] for x in mc_data]
        
        fig.add_trace(
            go.Scatter(x=mc_sims, y=mc_prices, mode='lines+markers', 
                      name='MC Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add Black-Scholes reference line
        bs_price = comparison_results["black_scholes"]["price"]
        fig.add_hline(y=bs_price, line_dash="dash", line_color="red", 
                     annotation_text="BS Reference", row=1, col=1)
        
        # Binomial convergence
        bin_data = comparison_results["binomial_tree"]["convergence_analysis"]
        bin_steps = [x["steps"] for x in bin_data]
        bin_prices = [x["price"] for x in bin_data]
        bin_errors = [x["error_pct"] for x in bin_data]
        
        fig.add_trace(
            go.Scatter(x=bin_steps, y=bin_prices, mode='lines+markers',
                      name='Binomial Price', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_hline(y=bs_price, line_dash="dash", line_color="red",
                     row=1, col=2)
        
        # Execution time comparison
        mc_times = [x["execution_time_ms"] for x in mc_data]
        bin_times = [x["execution_time_ms"] for x in bin_data]
        
        fig.add_trace(
            go.Scatter(x=mc_sims, y=mc_times, mode='lines+markers',
                      name='MC Execution Time', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=bin_steps, y=bin_times, mode='lines+markers',
                      name='Binomial Time', line=dict(color='purple')),
            row=2, col=1, secondary_y=True
        )
        
        # Error analysis
        fig.add_trace(
            go.Bar(x=['MC 1K', 'MC 5K', 'MC 10K', 'MC 50K'], y=mc_errors,
                  name='MC Error %', marker_color='lightblue'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=['Bin 50', 'Bin 100', 'Bin 200', 'Bin 500', 'Bin 1K'], y=bin_errors,
                  name='Binomial Error %', marker_color='lightgreen'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Model Validation: {option_type.title()} Option Pricing",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Simulations", row=1, col=1)
        fig.update_xaxes(title_text="Steps", row=1, col=2)
        fig.update_xaxes(title_text="Parameters", row=2, col=1)
        fig.update_xaxes(title_text="Model Configuration", row=2, col=2)
        
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Option Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Error (%)", row=2, col=2)
        
        return fig
    
    def generate_validation_report(self, comparison_results: Dict) -> str:
        """Generate a text report of validation results"""
        
        report = "# Model Validation Report\n\n"
        
        # Summary
        summary = comparison_results["summary"]
        report += f"**Reference Price (Black-Scholes)**: ${summary['reference_price']:.6f}\n\n"
        
        # Monte Carlo Analysis
        mc_best = comparison_results["monte_carlo"]["best_result"]
        report += f"## Monte Carlo Analysis\n"
        report += f"- **Best Configuration**: {mc_best['simulations']:,} simulations\n"
        report += f"- **Price**: ${mc_best['price']:.6f}\n"
        report += f"- **Error vs BS**: {mc_best['error_pct']:.4f}%\n"
        report += f"- **Execution Time**: {mc_best['execution_time_ms']:.2f}ms\n"
        report += f"- **Confidence Interval**: ±${mc_best['confidence_interval']:.6f}\n\n"
        
        # Binomial Analysis
        bin_best = comparison_results["binomial_tree"]["best_result"]
        report += f"## Binomial Tree Analysis\n"
        report += f"- **Best Configuration**: {bin_best['steps']} steps\n"
        report += f"- **Price**: ${bin_best['price']:.6f}\n"
        report += f"- **Error vs BS**: {bin_best['error_pct']:.4f}%\n"
        report += f"- **Execution Time**: {bin_best['execution_time_ms']:.2f}ms\n\n"
        
        # Performance Summary
        report += f"## Performance Summary\n"
        report += f"- **Fastest Method**: {summary['fastest_method']}\n"
        report += f"- **Most Accurate Monte Carlo**: {mc_best['error_pct']:.4f}% error\n"
        report += f"- **Most Accurate Binomial**: {bin_best['error_pct']:.4f}% error\n\n"
        
        # Greeks
        greeks = comparison_results["greeks"]
        report += f"## Greeks (Black-Scholes)\n"
        for greek, value in greeks.items():
            report += f"- **{greek.title()}**: {value:.6f}\n"
        
        return report
