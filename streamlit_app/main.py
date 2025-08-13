import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.models.monte_carlo import MonteCarloModel
from pricing_engine.models.greeks import GreeksCalculator
from pricing_engine.models.binomial_tree import BinomialTreeModel
from pricing_engine.models.implied_vol import ImpliedVolatilitySolver
from pricing_engine.data_service.market_data import MarketDataService
from pricing_engine.data_service.database import DatabaseService
from pricing_engine.analytics.model_validator import ModelValidator
from pricing_engine.models.merton_jump_diffusion import MertonJumpDiffusion
from pricing_engine.models.heston_stochastic_volatility import HestonStochasticVolatility
from config.settings import settings
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

# Page config
st.set_page_config(
    page_title="Advanced Options Pricing Engine v2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize models and services
def load_models():
    """Load models without caching to avoid recursion errors"""
    
    # Check if models already exist in session state to avoid reloading
    if 'models_loaded' not in st.session_state:
        models = {
            "bs_model": BlackScholesModel(),
            "mc_model": MonteCarloModel(),
            "binomial_model": BinomialTreeModel(),
            "merton_model": MertonJumpDiffusion(),
            "heston_model": HestonStochasticVolatility(),
            "implied_vol": ImpliedVolatilitySolver(),
            "greeks_calc": GreeksCalculator(),
            "market_data": MarketDataService(),
            "database": DatabaseService()
        }
        
        # Add model validator
        models["model_validator"] = ModelValidator(
            models["bs_model"], models["mc_model"], 
            models["binomial_model"], models["greeks_calc"]
        )
        
        # Store in session state instead of caching
        st.session_state['models'] = models
        st.session_state['models_loaded'] = True
        
    return st.session_state['models']

@st.cache_data(ttl=1800)
def get_cached_option_chain(symbol):
    """Cache option chain data to reduce API calls"""
    models = load_models()
    return models["market_data"].get_option_chain(symbol)

def save_calculation_to_db(models, calculation_data):
    """Save calculation to database"""
    try:
        calc_id = models["database"].save_calculation(calculation_data)
        if calc_id:
            st.success(f"✅ Calculation saved successfully (ID: {calc_id[:8]}...)")
        return calc_id
    except Exception as e:
        st.error(f"❌ Could not save to database: {e}")
        return None

def safe_clear_session_state():
    """Safely clear session state without causing recursion errors"""
    try:
        # List of keys to preserve (don't clear these)
        preserve_keys = ['models', 'models_loaded']
        
        # Get all keys to clear
        keys_to_clear = [key for key in st.session_state.keys() 
                        if key not in preserve_keys]
        
        # Clear keys one by one
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Session data cleared successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error clearing session state: {e}")
        return False

def main():
    st.title("🚀 Advanced Options Pricing Engine")
    st.markdown("*Multi-Model Pricing with Enhanced Analytics*")
    
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Main tabs
    main_tabs = st.tabs([
        "💰 Pricing Engine", 
        "🌲 Model Analysis", 
        "📈 Implied Volatility",
        "🔬 Model Validation",
        "⚡ Merton Jump Diffusion",
        "🌊 Heston Stochastic Volatility"
    ])
    
    try:
        with main_tabs[0]:
            pricing_tab(models)
        
        with main_tabs[1]:
            model_analysis_tab(models)
        
        with main_tabs[2]:
            implied_vol_tab(models)
        
        with main_tabs[3]:
            model_validation_tab(models)
            
        with main_tabs[4]:
            merton_tab(models)
            
        with main_tabs[5]:
            heston_tab(models)
            
    except Exception as e:
        st.error(f"Error in tab execution: {e}")
        st.info("Try clearing all data using the button in the sidebar")

def pricing_tab(models):
    """Enhanced pricing tab with manual save"""
    st.header("💰 Multi-Model Option Pricing")
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("---")
        if st.button("🔄 Clear All Data", help="Reset all calculations and session data"):
            if safe_clear_session_state():
                st.rerun()
        
        st.header("📊 Parameters")
        
        # Symbol selection
        symbol = st.selectbox("Select Stock", settings.SUPPORTED_STOCKS, index=0)
        
        # Get market data
        try:
            with st.spinner("Fetching market data..."):
                stock_data = models["market_data"].get_stock_data(symbol)
                options_chain = get_cached_option_chain(symbol)
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            return
        
        # Data source info
        data_source = stock_data.get('data_source', 'unknown')
        if data_source == 'dummy':
            st.warning(f"⚠️ Using fallback data for {symbol}")
        else:
            st.success(f"✅ Current Price: ${stock_data['current_price']} ({data_source})")
        
        # Option parameters
        option_type = st.selectbox("Option Type", ["Call", "Put"], 
                                 help="💡 **Call**: Right to buy at strike price | **Put**: Right to sell at strike price")
        
        # Enhanced expiry/strike selection
        available_expiries = []
        if options_chain and 'options' in options_chain:
            expiry_set = set()
            for opt in options_chain['options']:
                expiry_set.add(opt['expiry_date'])
            available_expiries = sorted(list(expiry_set))
        
        if available_expiries:
            selected_expiry = st.selectbox("Expiration Date", available_expiries, index=0,
                                         help="💡 **Expiry**: When the option contract expires")
            expiry_dt = datetime.strptime(selected_expiry, '%Y-%m-%d').date()
            time_to_maturity = max((expiry_dt - datetime.now().date()).days / 365.0, 1/365)
            
            # Available strikes
            available_strikes = []
            for opt in options_chain['options']:
                if (opt['expiry_date'] == selected_expiry and 
                    opt['option_type'] == option_type.lower()):
                    available_strikes.append(opt['strike'])
            
            available_strikes = sorted(list(set(available_strikes)))
            
            if available_strikes:
                current_price = stock_data['current_price']
                closest_idx = min(range(len(available_strikes)), 
                                key=lambda i: abs(available_strikes[i] - current_price))
                strike_price = st.selectbox("Strike Price", available_strikes, index=closest_idx,
                                          help="💡 **Strike**: Price at which you can exercise the option")
            else:
                strike_price = st.number_input("Strike Price ($)", 
                                             value=float(stock_data['current_price']) * 1.05,
                                             min_value=0.01, step=0.01,
                                             help="💡 **Strike**: Price at which you can exercise the option")
        else:
            # Fallback to manual inputs
            expiry_date = st.date_input("Expiration Date", 
                                      value=datetime.now().date() + timedelta(days=30),
                                      min_value=datetime.now().date(),
                                      help="💡 **Expiry**: When the option contract expires")
            time_to_maturity = (expiry_date - datetime.now().date()).days / 365.0
            selected_expiry = expiry_date.strftime('%Y-%m-%d')
            strike_price = st.number_input("Strike Price ($)",
                                         value=float(stock_data['current_price']) * 1.05,
                                         min_value=0.01, step=0.01,
                                         help="💡 **Strike**: Price at which you can exercise the option")
        
        spot_price = st.number_input("Spot Price ($)",
                                   value=float(stock_data['current_price']),
                                   min_value=0.01, step=0.01,
                                   help="💡 **Spot**: Current market price of the underlying stock")
        
        st.write(f"⏰ **Time to Maturity**: {time_to_maturity:.4f} years ({int(time_to_maturity*365)} days)")
        
        # Market parameters with info tooltips
        volatility = st.slider("Volatility (σ)", 0.05, 1.0,
                             models["market_data"].calculate_historical_volatility(symbol),
                             0.01, format="%.2f",
                             help="💡 **Volatility**: Measure of price fluctuation. Higher = more uncertain price movements")
        
        risk_free_rate = st.slider("Risk-Free Rate (r)", 0.01, 0.10,
                                 models["market_data"].get_risk_free_rate(),
                                 0.001, format="%.3f",
                                 help="💡 **Risk-Free Rate**: Return on safest investments (e.g., Treasury bonds)")
        
        # Model settings
        st.subheader("🔧 Model Settings")
        selected_models = st.multiselect("Select Pricing Models",
                                       ["Black-Scholes", "Monte Carlo", "Binomial Tree", "Merton Jump Diffusion", "Heston Stochastic Volatility Model"],
                                       default=["Black-Scholes"],
                                       help="💡 Choose which mathematical models to use for pricing", key="main_selected_models")
        
        mc_simulations = st.selectbox("MC Simulations", [1000, 5000, 10000, 50000], index=2,
                                    help="💡 **Monte Carlo**: More simulations = higher accuracy but slower computation")
        binomial_steps = st.slider("Binomial Steps", 10, 500, 100,
                                 help="💡 **Binomial Steps**: More steps = higher accuracy but slower computation")

        if "Merton Jump Diffusion" in selected_models:
            with st.expander("🔧 Merton Jump Parameters"):
                jump_intensity = st.slider("Jump Intensity (λ)", 0.0, 0.5, 0.1, 0.01,
                                         help="Average jumps per year")
                jump_mean = st.slider("Jump Mean (μⱼ)", -0.2, 0.1, -0.05, 0.01,
                                    help="Average jump size (negative = crash bias)")
                jump_std = st.slider("Jump Volatility (σⱼ)", 0.05, 0.3, 0.15, 0.01,
                                   help="Jump size variability")
        else:
            # Default values when not selected
            jump_intensity, jump_mean, jump_std = 0.1, -0.05, 0.15

        if "Heston Stochastic Volatility" in selected_models:
            with st.expander("🔧 Heston Parameters"):
                heston_v0 = st.slider("Initial Variance (v₀)", 0.01, 0.2, 0.04, 0.01, key="main_heston_v0")
                heston_theta = st.slider("Long-run Variance (θ)", 0.01, 0.2, 0.04, 0.01, key="main_heston_theta")
                heston_kappa = st.slider("Mean Reversion (κ)", 0.1, 5.0, 1.5, 0.1, key="main_heston_kappa")
                heston_sigma = st.slider("Vol of Vol (σ)", 0.1, 1.0, 0.3, 0.01, key="main_heston_sigma")
                heston_rho = st.slider("Correlation (ρ)", -0.99, 0.99, -0.7, 0.01, key="main_heston_rho")
        else:
            # Default values when not expanded
            heston_v0, heston_theta, heston_kappa, heston_sigma, heston_rho = 0.04, 0.04, 1.5, 0.3, -0.7
        
    
    # Main content
    if time_to_maturity > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calculate results
            results_data = []
            calculation_results = {}
            
            if "Black-Scholes" in selected_models:
                with st.spinner("Calculating Black-Scholes..."):
                    bs_prices = models["bs_model"].calculate_all_prices(
                        spot_price, strike_price, time_to_maturity, risk_free_rate, volatility
                    )
                    bs_price = bs_prices["call_price"] if option_type.lower() == "call" else bs_prices["put_price"]
                    results_data.append({
                        "Model": "Black-Scholes",
                        "Price": f"${bs_price:.4f}",
                        "Execution Time": "<1ms",
                        "Method": "Analytical"
                    })
                    calculation_results["black_scholes"] = {"price": bs_price, "execution_time_ms": 0.1}
            
            if "Monte Carlo" in selected_models:
                with st.spinner(f"Running {mc_simulations:,} Monte Carlo simulations..."):
                    models["mc_model"].n_simulations = mc_simulations
                    mc_result = models["mc_model"].european_option_price(
                        spot_price, strike_price, time_to_maturity, risk_free_rate, 
                        volatility, option_type.lower()
                    )
                    results_data.append({
                        "Model": "Monte Carlo",
                        "Price": f"${mc_result['price']:.4f}",
                        "Execution Time": f"{mc_result['execution_time_ms']:.1f}ms",
                        "Method": f"{mc_simulations:,} simulations"
                    })
                    calculation_results["monte_carlo"] = mc_result
            
            if "Binomial Tree" in selected_models:
                with st.spinner(f"Building {binomial_steps}-step binomial tree..."):
                    binomial_result = models["binomial_model"].price_european_option(
                        spot_price, strike_price, time_to_maturity, risk_free_rate, 
                        volatility, option_type.lower(), binomial_steps
                    )
                    results_data.append({
                        "Model": "Binomial Tree",
                        "Price": f"${binomial_result['price']:.4f}",
                        "Execution Time": f"{binomial_result['execution_time_ms']:.1f}ms",
                        "Method": f"{binomial_steps} steps"
                    })
                    calculation_results["binomial"] = binomial_result

            if "Merton Jump Diffusion" in selected_models:
                with st.spinner("Calculating Merton Jump Diffusion..."):
                    # Configure Merton model
                    models["merton_model"].lambda_jump = jump_intensity
                    models["merton_model"].mu_jump = jump_mean
                    models["merton_model"].sigma_jump = jump_std
                    models["merton_model"].k_bar = np.exp(jump_mean + 0.5 * jump_std**2) - 1
                
                    merton_result = models["merton_model"].european_option_price(
                        spot_price, strike_price, time_to_maturity, 
                        risk_free_rate, volatility, option_type.lower()
                    )
                
                    results_data.append({
                        "Model": "Merton Jump Diffusion",
                        "Price": f"${merton_result['price']:.4f}",
                        "Execution Time": f"{merton_result['execution_time_ms']:.1f}ms",
                        "Method": merton_result['method']
                    })
                
                    calculation_results["merton"] = merton_result    

            if "Heston Stochastic Volatility" in selected_models:
                with st.spinner("Calculating Heston Stochastic Volatility..."):
                    # Configure Heston model
                    models["heston_model"].v0 = heston_v0
                    models["heston_model"].theta = heston_theta
                    models["heston_model"].kappa = heston_kappa
                    models["heston_model"].sigma = heston_sigma
                    models["heston_model"].rho = heston_rho
                    
                    heston_result = models["heston_model"].european_option_price(
                        spot_price, strike_price, time_to_maturity, 
                        risk_free_rate, option_type
                    )
                    
                    results_data.append({
                        "Model": "Heston Stochastic Vol",
                        "Price": f"${heston_result['price']:.4f}",
                        "Execution Time": f"{heston_result['execution_time_ms']:.1f}ms",
                        "Method": heston_result['method']
                    })
                    
                    calculation_results["heston"] = heston_result
            
            # Greeks calculation
            with st.spinner("Calculating Greeks..."):
                greeks = models["greeks_calc"].calculate_all_greeks(
                    spot_price, strike_price, time_to_maturity, risk_free_rate, 
                    volatility, option_type.lower()
                )
            if results_data:
                st.subheader("🎯 Pricing Results")
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            else:
                st.warning("No pricing results available. Please check your model selections and parameters.")
            # Market comparison
            st.subheader("🎯 Market vs Model Comparison")
            
            matching_option = None
            if options_chain and 'options' in options_chain:
                for opt in options_chain['options']:
                    if (abs(opt['strike'] - strike_price) < 0.01 and 
                        opt['option_type'] == option_type.lower() and
                        opt['expiry_date'] == selected_expiry):
                        matching_option = opt
                        break
            
            if matching_option:
                comp_cols = st.columns(4)
                with comp_cols[0]:
                    st.metric("Market Price", f"${matching_option['market_price']:.2f}",
                            help="💡 Current trading price in the market")
                with comp_cols[1]:
                    model_price = float(results_data[0]["Price"].replace("$", ""))
                    st.metric("Model Price", f"${model_price:.2f}",
                            help="💡 Theoretical fair value from mathematical model")
                with comp_cols[2]:
                    diff = matching_option['market_price'] - model_price
                    st.metric("Difference", f"${diff:.2f}",
                            help="💡 Market price minus model price")
                with comp_cols[3]:
                    if model_price > 0:
                        pct_diff = (diff / model_price) * 100
                        st.metric("% Difference", f"{pct_diff:.1f}%",
                                help="💡 Percentage difference between market and model")
                
                # Data source info
                data_source_chain = options_chain.get('data_source', 'unknown')
                if data_source_chain in ['synthetic', 'enhanced_synthetic']:
                    st.info("📝 **Note**: Market prices are synthetic (generated for demonstration)")
                else:
                    st.success(f"📡 **Live Data**: Market prices from {data_source_chain}")
            else:
                st.info("🔍 No matching option found - using synthetic market data")
            
            # MANUAL SAVE BUTTON
            st.subheader("💾 Save Results")
            
            save_cols = st.columns([1, 1, 2])
            with save_cols[0]:
                if st.button("💾 Save to Database", type="primary"):
                    if results_data:
                        calculation_data = {
                            "session_id": st.session_state.session_id,
                            "symbol": symbol,
                            "option_type": option_type.lower(),
                            "parameters": {
                                "spot_price": spot_price,
                                "strike_price": strike_price,
                                "time_to_maturity": time_to_maturity,
                                "volatility": volatility,
                                "risk_free_rate": risk_free_rate,
                                "expiry_date": selected_expiry
                            },
                            "results": calculation_results,
                            "greeks": greeks,
                            "data_source": data_source,
                            "models_used": selected_models
                        }
                        
                        save_calculation_to_db(models, calculation_data)
                    else:
                        st.warning("No results to save. Please run calculations first.")
            
            with save_cols[1]:
                if st.button("📥 Export CSV"):
                    if results_data:
                        # Create export dataframe
                        export_data = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "option_type": option_type,
                            "spot_price": spot_price,
                            "strike_price": strike_price,
                            "time_to_maturity": time_to_maturity,
                            "volatility": volatility,
                            "risk_free_rate": risk_free_rate,
                        }
                        
                        # Add model results
                        for model_name, model_data in calculation_results.items():
                            if isinstance(model_data, dict) and 'price' in model_data:
                                export_data[f"{model_name}_price"] = model_data['price']
                        
                        # Add greeks
                        for greek_name, greek_value in greeks.items():
                            export_data[f"greek_{greek_name}"] = greek_value
                        
                        export_df = pd.DataFrame([export_data])
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"option_pricing_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No results to export.")
        
        with col2:
            # Greeks display with info tooltips
            st.subheader("📈 The Greeks")
            
            greeks_metrics = [
                ("Delta (Δ)", greeks['delta'], "Price sensitivity to $1 change in underlying stock"),
                ("Gamma (Γ)", greeks['gamma'], "Rate of change of Delta with respect to stock price"),
                ("Theta (Θ)", greeks['theta'], "Daily time decay - how much value lost per day"),
                ("Vega (ν)", greeks['vega'], "Sensitivity to 1% change in implied volatility"),
                ("Rho (ρ)", greeks['rho'], "Sensitivity to 1% change in risk-free interest rate")
            ]
            
            for greek_name, greek_value, help_text in greeks_metrics:
                st.metric(greek_name, f"{greek_value:.4f}", help=f"💡 **{help_text}**")
    else:
        st.error("⚠️ Option has expired!")

def model_analysis_tab(models):
    """Enhanced binomial tree analysis tab with info tooltips"""
    st.header("🌲 Advanced Model Analysis")
    st.info("💡 **Binomial Trees**: Discrete-time models that approximate continuous option pricing by creating a tree of possible stock price movements")
    
    # Get parameters
    symbol = "AAPL"
    stock_data = models["market_data"].get_stock_data(symbol)
    
    # Analysis parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Parameters")
        
        spot_price = st.number_input("Analysis Spot Price", value=float(stock_data['current_price']), min_value=0.01,
                                   help="💡 Current stock price for analysis")
        strike_price = st.number_input("Analysis Strike Price", value=float(stock_data['current_price']) * 1.05, min_value=0.01,
                                     help="💡 Exercise price of the option")
        time_to_maturity = st.slider("Time to Maturity (years)", 0.01, 1.0, 0.25, 0.01,
                                   help="💡 Time until option expires")
        volatility = st.slider("Analysis Volatility", 0.05, 1.0, 0.25, 0.01,
                             help="💡 Expected price volatility")
        risk_free_rate = st.slider("Analysis Risk-Free Rate", 0.01, 0.10, 0.05, 0.001,
                                 help="💡 Risk-free interest rate")
        option_type = st.selectbox("Analysis Option Type", ["call", "put"],
                                 help="💡 Call = right to buy, Put = right to sell")
        
        tree_steps = st.slider("Tree Steps", 5, 200, 50,
                             help="💡 More steps = higher accuracy but slower computation")
        show_american = st.checkbox("Show American Option", False,
                                  help="💡 American options can be exercised early, European only at expiry")
        
    with col2:
        if time_to_maturity > 0:
            # European option calculation
            with st.spinner("Building binomial tree..."):
                european_result = models["binomial_model"].price_european_option(
                    spot_price, strike_price, time_to_maturity, risk_free_rate, 
                    volatility, option_type, tree_steps
                )
            
            st.subheader("Analysis Results")
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("European Price", f"${european_result['price']:.4f}",
                        help="💡 Price assuming exercise only at expiration")
                st.metric("Execution Time", f"{european_result['execution_time_ms']:.1f}ms",
                        help="💡 Time taken to compute the result")
            
            if show_american:
                with st.spinner("Calculating American option..."):
                    american_result = models["binomial_model"].price_american_option(
                        spot_price, strike_price, time_to_maturity, risk_free_rate, 
                        volatility, option_type, tree_steps
                    )
                
                with col2_2:
                    st.metric("American Price", f"${american_result['price']:.4f}",
                            help="💡 Price allowing early exercise")
                    st.metric("Early Exercise Premium", f"${american_result['early_exercise_premium']:.4f}",
                            help="💡 Extra value from early exercise flexibility")
            
            # Tree visualization for small trees
            if tree_steps <= 6:
                st.subheader("Tree Visualization")
                st.info("💡 **Tree Structure**: Each node shows stock price (top) and option value (bottom)")
                tree_fig = models["binomial_model"].create_tree_visualization(
                    european_result['stock_tree'], 
                    european_result['option_tree'], 
                    european_result['params'],
                    max_display_steps=min(tree_steps, 4)
                )
                st.plotly_chart(tree_fig, use_container_width=True)
            
            # Convergence analysis
            st.subheader("Convergence Analysis")
            st.info("💡 **Convergence**: As we increase steps, binomial prices approach the exact Black-Scholes solution")
            
            if st.button("Run Convergence Analysis"):
                with st.spinner("Running convergence analysis..."):
                    convergence_df = models["binomial_model"].convergence_analysis(
                        spot_price, strike_price, time_to_maturity, risk_free_rate, 
                        volatility, option_type
                    )
                    
                    fig_conv = go.Figure()
                    
                    fig_conv.add_trace(go.Scatter(
                        x=convergence_df['n_steps'],
                        y=convergence_df['european_price'],
                        mode='lines+markers',
                        name='European Price',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Add Black-Scholes reference
                    bs_price = models["bs_model"].call_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility) if option_type == "call" else models["bs_model"].put_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                    fig_conv.add_hline(y=bs_price, line_dash="dash", line_color="red", 
                                     annotation_text="Black-Scholes Reference")
                    
                    if show_american:
                        fig_conv.add_trace(go.Scatter(
                            x=convergence_df['n_steps'],
                            y=convergence_df['american_price'],
                            mode='lines+markers',
                            name='American Price',
                            line=dict(color='green', width=3)
                        ))
                    
                    fig_conv.update_layout(
                        title="Binomial Tree Convergence Analysis",
                        xaxis_title="Number of Steps",
                        yaxis_title="Option Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_conv, use_container_width=True)
                    
                    # Show convergence error
                    final_price = convergence_df['european_price'].iloc[-1]
                    convergence_error = abs(final_price - bs_price) / bs_price * 100
                    st.success(f"🎯 **Convergence Error**: {convergence_error:.3f}% vs Black-Scholes")

def implied_vol_tab(models):
    """FIXED IMPLIED VOLATILITY TAB"""
    st.header("📈 Implied Volatility Analysis")
    st.info("💡 **Implied Volatility**: The volatility level that makes the Black-Scholes model price match the market price")
    
    iv_cols = st.columns([1, 2])
    
    with iv_cols[0]:
        st.subheader("IV Calculator")
        
        # IV calculation parameters
        symbol = st.selectbox("IV Symbol", settings.SUPPORTED_STOCKS, index=0)
        stock_data = models["market_data"].get_stock_data(symbol)
        
        spot_price = st.number_input("IV Spot Price", value=float(stock_data['current_price']), min_value=0.01,
                                   help="💡 Current stock price")
        strike_price = st.number_input("IV Strike Price", value=float(stock_data['current_price']) * 1.05, min_value=0.01,
                                     help="💡 Option strike price")
        time_to_maturity = st.slider("IV Time to Maturity", 0.01, 1.0, 0.25, 0.01,
                                   help="💡 Time until expiration")
        risk_free_rate = st.slider("IV Risk-Free Rate", 0.01, 0.10, 0.05, 0.001,
                                 help="💡 Risk-free interest rate")
        option_type = st.selectbox("IV Option Type", ["call", "put"],
                                 help="💡 Type of option contract")
        
        market_price_input = st.number_input("Market Option Price", value=5.0, min_value=0.01, step=0.01,
                                           help="💡 Current market trading price")
        iv_method = st.selectbox("Solver Method", ["brent", "newton"], index=0,
                               help="💡 Mathematical algorithm for finding implied volatility")
        
        if st.button("Calculate Implied Volatility", type="primary"):
            with st.spinner("Calculating IV..."):
                iv_result = models["implied_vol"].calculate_iv(
                    market_price_input, spot_price, strike_price, 
                    time_to_maturity, risk_free_rate, option_type, iv_method
                )
                
                if not np.isnan(iv_result.get('iv', np.nan)):
                    st.success(f"🎯 **Implied Volatility: {iv_result['iv']*100:.2f}%**")
                    
                    iv_metrics = st.columns(2)
                    with iv_metrics[0]:
                        st.metric("Theoretical Price", f"${iv_result['theoretical_price']:.4f}",
                                help="💡 Black-Scholes price using calculated IV")
                        st.metric("Pricing Error", f"${iv_result['pricing_error']:.4f}",
                                help="💡 Difference between theoretical and market price")
                    with iv_metrics[1]:
                        st.metric("Method Used", iv_result['method'],
                                help="💡 Algorithm used for calculation")
                        st.metric("Moneyness", f"{iv_result['moneyness']:.4f}",
                                help="💡 Ratio of spot to strike price")
                else:
                    st.error(f"❌ Could not calculate IV: {iv_result.get('error', 'Unknown error')}")
    
    with iv_cols[1]:
        st.subheader("Volatility Smile Analysis")
        st.info("💡 **Volatility Smile**: IV typically varies across strikes, creating a 'smile' or 'skew' pattern")
        
        # Generate synthetic option chain for IV analysis
        symbol_smile = st.selectbox("Smile Analysis Symbol", settings.SUPPORTED_STOCKS, index=0, key="smile_symbol")
        
        # FIXED: Add a unique key and state management for plot updates
        plot_key = f"iv_plot_{symbol_smile}"
        
        if st.button("Generate Volatility Smile", key="generate_smile"):
            with st.spinner("Generating volatility smile..."):
                # Get stock data for smile analysis
                stock_data_smile = models["market_data"].get_stock_data(symbol_smile)
                
                # Generate synthetic options chain using basic logic (since we don't have the enhanced generator)
                spot = stock_data_smile['current_price']
                options_data = []
                
                # Create strikes and calculate synthetic IV
                strikes = np.linspace(spot * 0.8, spot * 1.2, 20)
                expiry = datetime.now() + timedelta(days=30)
                T = 30/365.0
                
                for strike in strikes:
                    for opt_type in ['call', 'put']:
                        # Generate synthetic market price with smile effect
                        moneyness = np.log(spot/strike) if opt_type == 'call' else np.log(strike/spot)
                        base_iv = 0.25 + 0.1 * moneyness**2  # Simple smile curve
                        
                        if opt_type == 'call':
                            market_price = models["bs_model"].call_price(spot, strike, T, 0.05, base_iv)
                        else:
                            market_price = models["bs_model"].put_price(spot, strike, T, 0.05, base_iv)
                        
                        # Add some noise
                        market_price *= np.random.uniform(0.98, 1.02)
                        
                        options_data.append({
                            'strike': strike,
                            'option_type': opt_type,
                            'market_price': max(0.01, market_price),
                            'time_to_expiry': T
                        })
                
                # Store in session state
                st.session_state[f'options_data_{symbol_smile}'] = options_data
                st.session_state[f'spot_price_{symbol_smile}'] = spot
        
        # Check if we have data to plot
        if f'options_data_{symbol_smile}' in st.session_state:
            options_data = st.session_state[f'options_data_{symbol_smile}']
            spot = st.session_state[f'spot_price_{symbol_smile}']
            
            # Option type selection for smile
            option_type_smile = st.radio("Option Type for Smile", ["call", "put", "both"], key="smile_type")
            
            # Calculate IV for all options
            iv_results = []
            for opt in options_data:
                iv_result = models["implied_vol"].calculate_iv(
                    opt['market_price'], spot, opt['strike'], 
                    opt['time_to_expiry'], 0.05, opt['option_type']
                )
                
                if not np.isnan(iv_result.get('iv', np.nan)):
                    iv_results.append({
                        'strike': opt['strike'],
                        'option_type': opt['option_type'],
                        'implied_volatility': iv_result['iv'],
                        'market_price': opt['market_price']
                    })
            
            if iv_results:
                iv_df = pd.DataFrame(iv_results)
                
                # Create smile plot
                fig_smile = go.Figure()
                
                if option_type_smile in ["call", "both"]:
                    call_data = iv_df[iv_df['option_type'] == 'call'].sort_values('strike')
                    if not call_data.empty:
                        fig_smile.add_trace(go.Scatter(
                            x=call_data['strike'],
                            y=call_data['implied_volatility'] * 100,
                            mode='markers+lines',
                            name='Call IV',
                            line=dict(color='blue', width=3),
                            marker=dict(size=8)
                        ))
                
                if option_type_smile in ["put", "both"]:
                    put_data = iv_df[iv_df['option_type'] == 'put'].sort_values('strike')
                    if not put_data.empty:
                        fig_smile.add_trace(go.Scatter(
                            x=put_data['strike'],
                            y=put_data['implied_volatility'] * 100,
                            mode='markers+lines',
                            name='Put IV',
                            line=dict(color='red', width=3),
                            marker=dict(size=8)
                        ))
                
                # Add current spot line
                fig_smile.add_vline(x=spot, line_dash="dash", line_color="green", 
                                  annotation_text="Current Spot")
                
                fig_smile.update_layout(
                    title=f'Volatility Smile - {symbol_smile}',
                    xaxis_title='Strike Price ($)',
                    yaxis_title='Implied Volatility (%)',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_smile, use_container_width=True, key=plot_key)
                
                # Show IV statistics
                if option_type_smile != "both":
                    filtered_data = iv_df[iv_df['option_type'] == option_type_smile]
                else:
                    filtered_data = iv_df
                
                if len(filtered_data) > 0:
                    st.subheader("IV Statistics")
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Average IV", f"{filtered_data['implied_volatility'].mean()*100:.1f}%",
                                help="💡 Mean implied volatility across strikes")
                    with stats_cols[1]:
                        st.metric("Min IV", f"{filtered_data['implied_volatility'].min()*100:.1f}%",
                                help="💡 Lowest implied volatility")
                    with stats_cols[2]:
                        st.metric("Max IV", f"{filtered_data['implied_volatility'].max()*100:.1f}%",
                                help="💡 Highest implied volatility")
                    with stats_cols[3]:
                        iv_range = (filtered_data['implied_volatility'].max() - filtered_data['implied_volatility'].min()) * 100
                        st.metric("IV Range", f"{iv_range:.1f}%",
                                help="💡 Difference between highest and lowest IV")
            else:
                st.warning("Could not calculate implied volatilities for the generated data")
        else:
            st.info("👆 Click 'Generate Volatility Smile' to create analysis")

def model_validation_tab(models):
    """Model validation and comparison tab for educational purposes"""
    st.header("🔬 Advanced Model Validation")
    
    st.markdown("""
    💡 **Educational Purpose**: This tab provides comprehensive validation and comparison of all pricing models.
    Use it to understand model accuracy, convergence, and performance characteristics without relying on market data.
    """)
    
    # Validation parameters
    val_cols = st.columns(2)
    
    with val_cols[0]:
        st.subheader("Model Parameters")
        val_symbol = st.selectbox("Validation Symbol", settings.SUPPORTED_STOCKS, index=0,
                                help="💡 Symbol used for reference (doesn't affect calculations)")
        val_option_type = st.selectbox("Validation Option Type", ["call", "put"],
                                     help="💡 Type of option to analyze")
        val_spot = st.number_input("Spot Price", value=150.0, min_value=1.0,
                                 help="💡 Current stock price for analysis")
        val_strike = st.number_input("Strike Price", value=155.0, min_value=1.0,
                                   help="💡 Exercise price of the option")
    
    with val_cols[1]:
        st.subheader("Market Parameters")
        val_time = st.slider("Time to Maturity (years)", 0.01, 2.0, 0.25, 0.01,
                            help="💡 Time until option expires")
        val_vol = st.slider("Volatility", 0.05, 1.0, 0.25, 0.01,
                           help="💡 Expected price volatility")
        val_rate = st.slider("Risk-Free Rate", 0.01, 0.15, 0.05, 0.001,
                            help="💡 Risk-free interest rate")
        run_validation = st.button("🚀 Run Comprehensive Validation", type="primary")
    
    if run_validation:
        with st.spinner("Running comprehensive model validation..."):
            try:
                # Run validation
                validation_results = models["model_validator"].comprehensive_model_comparison(
                    val_spot, val_strike, val_time, val_rate, val_vol, val_option_type
                )
                
                # Display results
                st.subheader("📊 Validation Results")
                
                # Summary metrics
                summary = validation_results["summary"]
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.metric("Reference Price (BS)", f"${summary['reference_price']:.6f}",
                            help="💡 Black-Scholes analytical solution (exact)")
                with summary_cols[1]:
                    st.metric("Best MC Error", f"{summary['best_mc_error_pct']:.4f}%",
                            help="💡 Lowest Monte Carlo error vs Black-Scholes")
                with summary_cols[2]:
                    st.metric("Best Binomial Error", f"{summary['best_binomial_error_pct']:.4f}%",
                            help="💡 Lowest Binomial Tree error vs Black-Scholes")
                with summary_cols[3]:
                    st.metric("Fastest Method", summary['fastest_method'],
                            help="💡 Model with shortest execution time")
                
                # Convergence visualization
                st.subheader("📈 Convergence Analysis")
                st.info("💡 **Convergence**: Shows how numerical methods approach the analytical solution as precision increases")
                
                convergence_fig = models["model_validator"].create_convergence_visualization(
                    validation_results, val_option_type
                )
                st.plotly_chart(convergence_fig, use_container_width=True)
                
                # Model comparison table
                st.subheader("📋 Detailed Model Comparison")
                
                comparison_cols = st.columns(2)
                
                with comparison_cols[0]:
                    st.write("**Monte Carlo Analysis**")
                    mc_df = pd.DataFrame(validation_results["monte_carlo"]["convergence_analysis"])
                    mc_display = mc_df[['simulations', 'price', 'error_pct', 'execution_time_ms']].copy()
                    mc_display.columns = ['Simulations', 'Price ($)', 'Error (%)', 'Time (ms)']
                    st.dataframe(mc_display, use_container_width=True)
                
                with comparison_cols[1]:
                    st.write("**Binomial Tree Analysis**")
                    bin_df = pd.DataFrame(validation_results["binomial_tree"]["convergence_analysis"])
                    bin_display = bin_df[['steps', 'price', 'error_pct', 'execution_time_ms']].copy()
                    bin_display.columns = ['Steps', 'Price ($)', 'Error (%)', 'Time (ms)']
                    st.dataframe(bin_display, use_container_width=True)
                
                # Educational insights
                st.subheader("🎓 Educational Insights")
                
                insights_cols = st.columns(2)
                
                with insights_cols[0]:
                    st.write("**📊 Accuracy Analysis**")
                    mc_best = validation_results["monte_carlo"]["best_result"]
                    bin_best = validation_results["binomial_tree"]["best_result"]
                    
                    st.write(f"• **Monte Carlo**: Achieves {mc_best['error_pct']:.4f}% error with {mc_best['simulations']:,} simulations")
                    st.write(f"• **Binomial Tree**: Achieves {bin_best['error_pct']:.4f}% error with {bin_best['steps']} steps")
                    st.write(f"• **Error decreases** as computational effort increases")
                
                with insights_cols[1]:
                    st.write("**⏱️ Performance Analysis**")
                    st.write(f"• **Black-Scholes**: Instantaneous (analytical)")
                    st.write(f"• **Best Monte Carlo**: {mc_best['execution_time_ms']:.1f}ms")
                    st.write(f"• **Best Binomial**: {bin_best['execution_time_ms']:.1f}ms")
                    st.write(f"• **Trade-off**: Accuracy vs Speed")
                
                # Generate and display validation report
                validation_report = models["model_validator"].generate_validation_report(validation_results)
                
                with st.expander("📄 Detailed Validation Report"):
                    st.markdown(validation_report)
                
                # Save validation results
                st.subheader("💾 Save Validation Results")
                if st.button("💾 Save Validation to Database"):
                    try:
                        validation_data = {
                            "session_id": st.session_state.session_id,
                            "validation_type": "comprehensive_model_comparison",
                            "symbol": val_symbol,
                            "option_type": val_option_type,
                            "parameters": {
                                "spot_price": val_spot,
                                "strike_price": val_strike,
                                "time_to_maturity": val_time,
                                "volatility": val_vol,
                                "risk_free_rate": val_rate
                            },
                            "validation_results": validation_results,
                            "report": validation_report
                        }
                        
                        save_calculation_to_db(models, validation_data)
                    except Exception as e:
                        st.warning(f"Could not save validation results: {e}")
            
            except Exception as e:
                st.error(f"Error during validation: {e}")
                st.info("Please check that all models are properly initialized")
    else:
        # Educational content when not running validation
        st.subheader("📚 Understanding Model Validation")
        
        ed_cols = st.columns(3)
        
        with ed_cols[0]:
            st.write("**🎯 Black-Scholes Model**")
            st.write("• Analytical (exact) solution")
            st.write("• Instantaneous calculation")
            st.write("• Used as reference standard")
            st.write("• Assumes constant volatility")
        
        with ed_cols[1]:
            st.write("**🎲 Monte Carlo Method**")
            st.write("• Statistical simulation")
            st.write("• Error ∝ 1/√N (N = simulations)")
            st.write("• Flexible for complex payoffs")
            st.write("• Computationally intensive")
        
        with ed_cols[2]:
            st.write("**🌲 Binomial Tree**")
            st.write("• Discrete-time approximation")
            st.write("• Error ∝ 1/n (n = steps)")
            st.write("• Can handle American options")
            st.write("• Good balance of speed/accuracy")
        
        st.info("👆 Configure parameters above and click 'Run Comprehensive Validation' to begin educational analysis")

def merton_tab(models):
    """Merton Jump Diffusion analysis and education tab"""
    st.header("⚡ Merton Jump Diffusion Model")
    
    st.markdown("""
    💡 **Jump Diffusion Modeling**: Extends Black-Scholes by adding sudden price jumps to capture 
    market crashes, earnings announcements, and other discrete events that cause immediate price movements.
    """)
    
    # Educational overview
    with st.expander("📚 Understanding Jump Diffusion"):
        st.markdown("""
        **Why Jumps Matter:**
        - Stock prices don't always move smoothly - they can "gap" up or down
        - News events, earnings, geopolitical events cause sudden price changes
        - Black-Scholes assumes continuous movement, missing this reality
        
        **Mathematical Framework:**
        ```
        dS(t) = (r - λk̄)S(t)dt + σS(t)dW(t) + S(t-)dJ(t)
        ```
        
        **Key Parameters:**
        - **λ (Lambda)**: Jump intensity - how often jumps occur
        - **μⱼ**: Average jump size (often negative for crash bias)
        - **σⱼ**: Jump size volatility - how much jump sizes vary
        """)
    
    # Parameter configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        
        # Basic option parameters
        spot_price = st.number_input("Spot Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=105.0, min_value=0.01)
        time_to_maturity = st.slider("Time to Maturity (years)", 0.01, 2.0, 0.25, 0.01)
        risk_free_rate = st.slider("Risk-Free Rate", 0.01, 0.15, 0.05, 0.001)
        volatility = st.slider("Continuous Volatility", 0.05, 1.0, 0.25, 0.01)
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        st.subheader("Jump Parameters")
        
        jump_intensity = st.slider(
            "Jump Intensity (λ)", 0.0, 1.0, 0.1, 0.01,
            help="💡 Average number of jumps per year. 0.1 = 1 jump every 10 years"
        )
        
        jump_mean = st.slider(
            "Jump Mean (μⱼ)", -0.3, 0.2, -0.05, 0.01,
            help="💡 Average jump size. Negative values model market crashes"
        )
        
        jump_std = st.slider(
            "Jump Volatility (σⱼ)", 0.05, 0.5, 0.15, 0.01,
            help="💡 Variability of jump sizes. Higher = more uncertain jump impact"
        )
        
        # Configure Merton model
        models["merton_model"].lambda_jump = jump_intensity
        models["merton_model"].mu_jump = jump_mean
        models["merton_model"].sigma_jump = jump_std
        models["merton_model"].k_bar = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        
        calculation_method = st.radio(
            "Calculation Method",
            ["Series Expansion", "Monte Carlo", "Both"],
            help="💡 Series expansion is analytical, Monte Carlo is simulation-based"
        )
    
    with col2:
        if st.button("🚀 Calculate Option Price", type="primary"):
            
            results = {}
            
            # Series expansion calculation
            if calculation_method in ["Series Expansion", "Both"]:
                with st.spinner("Calculating via series expansion..."):
                    series_result = models["merton_model"].european_option_price(
                        spot_price, strike_price, time_to_maturity, 
                        risk_free_rate, volatility, option_type
                    )
                    results["series"] = series_result
                    
                    # STORE IN SESSION STATE
                    st.session_state["merton_results"] = results
            
            # Monte Carlo calculation  
            if calculation_method in ["Monte Carlo", "Both"]:
                with st.spinner("Running Monte Carlo simulation..."):
                    mc_result = models["merton_model"].monte_carlo_simulation(
                        spot_price, strike_price, time_to_maturity,
                        risk_free_rate, volatility, option_type, n_simulations=50000
                    )
                    results["monte_carlo"] = mc_result
                    
                    # UPDATE SESSION STATE
                    st.session_state["merton_results"] = results
            
           # Display results
            st.subheader("🎯 Pricing Results")
            
            # Comparison with Black-Scholes
            bs_price = models["bs_model"].call_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility) if option_type == "call" else models["bs_model"].put_price(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
            
            comparison_data = [
                {"Model": "Black-Scholes", "Price": f"${bs_price:.4f}", "Method": "Analytical", "Time": "<1ms"}
            ]
            
            if "series" in results:
                comparison_data.append({
                    "Model": "Merton (Series)", 
                    "Price": f"${results['series']['price']:.4f}",
                    "Method": results['series']['method'],
                    "Time": f"{results['series']['execution_time_ms']:.1f}ms"
                })
            
            if "monte_carlo" in results:
                comparison_data.append({
                    "Model": "Merton (MC)",
                    "Price": f"${results['monte_carlo']['price']:.4f}",
                    "Method": results['monte_carlo']['method'], 
                    "Time": f"{results['monte_carlo']['execution_time_ms']:.1f}ms"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # FIXED: Check session state for persistent data
        if "merton_results" in st.session_state:
            results = st.session_state["merton_results"]
            
            # Show series breakdown with persistent checkbox state
            if "series" in results:
                # Initialize checkbox state in session
                if "show_series_breakdown" not in st.session_state:
                    st.session_state["show_series_breakdown"] = False
                
                # Use session state for checkbox
                show_breakdown = st.checkbox(
                    "Show Series Breakdown", 
                    value=st.session_state["show_series_breakdown"],
                    key="series_breakdown_checkbox"
                )
                
                # Update session state
                st.session_state["show_series_breakdown"] = show_breakdown
                
                # Display breakdown if checked
                if show_breakdown:
                    series_df = pd.DataFrame(results["series"]["series_breakdown"][:10])
                    series_df.columns = ['Jumps', 'Probability', 'BS Price', 'Contribution', 'Cumulative']
                    st.subheader("Series Expansion Details")
                    st.dataframe(series_df, use_container_width=True)
            
            # Monte Carlo statistics
            if "monte_carlo" in results:
                st.subheader("📊 Monte Carlo Simulation Statistics")
                mc_stats = results["monte_carlo"]["simulation_stats"]
                
                mc_cols = st.columns(4)
                with mc_cols[0]:
                    st.metric("Avg Final Price", f"${mc_stats['avg_final_price']:.2f}")
                with mc_cols[1]:
                    st.metric("Avg Jumps/Path", f"{mc_stats['avg_jumps_per_path']:.2f}")
                with mc_cols[2]:
                    st.metric("Max Jumps", mc_stats['max_jumps_observed'])
                with mc_cols[3]:
                    st.metric("Paths with Jumps", f"{mc_stats['jump_frequency']*100:.1f}%")
        
        # Parameter sensitivity analysis
        st.subheader("🔬 Parameter Sensitivity Analysis")
        
        if st.button("Analyze Jump Parameter Sensitivity"):
            with st.spinner("Running sensitivity analysis..."):
                sensitivity_data = models["merton_model"].parameter_sensitivity_analysis(
                    spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type
                )
                
                # Create visualization
                sensitivity_fig = models["merton_model"].create_visualization(sensitivity_data, option_type)
                st.plotly_chart(sensitivity_fig, use_container_width=True)
                
                # Key insights
                st.subheader("🔍 Key Insights")
                lambda_impact = max(sensitivity_data['jump_intensity_analysis']['price_change'])
                mu_impact = max(sensitivity_data['jump_mean_analysis']['price_change'])
                sigma_impact = max(sensitivity_data['jump_volatility_analysis']['price_change'])
                
                st.write(f"**Jump Intensity Impact**: Up to {lambda_impact:.1f}% price change")
                st.write(f"**Jump Mean Impact**: Up to {mu_impact:.1f}% price change") 
                st.write(f"**Jump Volatility Impact**: Up to {sigma_impact:.1f}% price change")
                
                most_sensitive = max([
                    ("Jump Intensity", lambda_impact),
                    ("Jump Mean", abs(mu_impact)), 
                    ("Jump Volatility", sigma_impact)
                ], key=lambda x: x[1])
                
                st.success(f"🎯 **Most Sensitive Parameter**: {most_sensitive[0]} ({most_sensitive[1]:.1f}% max impact)")

def heston_tab(models):
    """Heston Stochastic Volatility analysis and education tab"""
    st.header("🌊 Heston Stochastic Volatility Model")
    
    st.markdown("""
    💡 **Stochastic Volatility Modeling**: The Heston model addresses Black-Scholes' constant volatility 
    assumption by modeling volatility as a mean-reverting stochastic process, capturing the volatility smile 
    observed in real markets.
    """)
    
    # Educational overview
    with st.expander("📚 Understanding Stochastic Volatility"):
        st.markdown("""
        **Why Stochastic Volatility Matters:**
        - Market volatility changes over time - it's not constant as Black-Scholes assumes
        - Volatility tends to cluster: high volatility periods followed by high volatility
        - There's a "leverage effect": stock prices and volatility are negatively correlated
        
        **Mathematical Framework:**
        ```
        dS(t) = rS(t)dt + √V(t)S(t)dW₁(t)  (Stock price)
        dV(t) = κ(θ - V(t))dt + σ√V(t)dW₂(t)  (Variance process)
        ```
        
        **Five Heston Parameters:**
        - **v₀**: Initial variance (current market volatility²)
        - **θ**: Long-run variance (where volatility reverts to)
        - **κ**: Mean reversion speed (how fast volatility returns to θ)
        - **σ**: Volatility of volatility (how much volatility fluctuates)
        - **ρ**: Correlation between price and volatility (usually negative)
        """)
    
    # Parameter configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Option Parameters")
        
        # ADD UNIQUE KEYS TO ALL WIDGETS
        spot_price = st.number_input("Spot Price ($)", value=100.0, min_value=0.01, key="heston_spot_price")
        strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, key="heston_strike_price")
        time_to_maturity = st.slider("Time to Maturity (years)", 0.01, 2.0, 0.25, 0.01, key="heston_time_to_maturity")
        risk_free_rate = st.slider("Risk-Free Rate", 0.01, 0.15, 0.05, 0.001, key="heston_risk_free_rate")
        option_type = st.selectbox("Option Type", ["call", "put"], key="heston_option_type")
        
        st.subheader("Heston Parameters")
        
        v0 = st.slider(
            "Initial Variance (v₀)", 0.01, 0.2, 0.04, 0.01,
            help="💡 Current variance level. 0.04 = 20% volatility",
            key="heston_v0"
        )
        
        theta = st.slider(
            "Long-run Variance (θ)", 0.01, 0.2, 0.04, 0.01,
            help="💡 Long-term variance target. Volatility reverts to √θ",
            key="heston_theta"
        )
        
        kappa = st.slider(
            "Mean Reversion Speed (κ)", 0.1, 5.0, 1.5, 0.1,
            help="💡 How fast variance returns to θ. Higher = faster reversion",
            key="heston_kappa"
        )
        
        sigma = st.slider(
            "Vol of Vol (σ)", 0.1, 1.0, 0.3, 0.01,
            help="💡 How much volatility fluctuates. Higher = more volatile volatility",
            key="heston_sigma"
        )
        
        rho = st.slider(
            "Correlation (ρ)", -0.99, 0.99, -0.7, 0.01,
            help="💡 Correlation between price and volatility. Negative = leverage effect",
            key="heston_rho"
        )
        
        calculation_method = st.radio(
            "Pricing Method",
            ["Characteristic Function", "Monte Carlo", "Both"],
            help="💡 CF is analytical (fast), MC is simulation-based (flexible)",
            key="heston_calculation_method"
        )
    
    with col2:
        if st.button("🚀 Calculate Heston Price", type="primary", key="heston_calculate_button"):
            
            results = {}
            
            # Characteristic function calculation
            if calculation_method in ["Characteristic Function", "Both"]:
                with st.spinner("Calculating via characteristic function..."):
                    cf_result = models["heston_model"].european_option_price(
                        spot_price, strike_price, time_to_maturity, 
                        risk_free_rate, option_type
                    )
                    results["characteristic_function"] = cf_result
            
            # Monte Carlo calculation
            if calculation_method in ["Monte Carlo", "Both"]:
                with st.spinner("Running Monte Carlo simulation..."):
                    mc_result = models["heston_model"].monte_carlo_simulation(
                        spot_price, strike_price, time_to_maturity,
                        risk_free_rate, option_type, n_simulations=100000
                    )
                    results["monte_carlo"] = mc_result
            
            # Display results
            st.subheader("🎯 Heston Pricing Results")
            
            # Comparison with Black-Scholes
            bs_price = models["bs_model"].call_price(spot_price, strike_price, time_to_maturity, risk_free_rate, np.sqrt(theta)) if option_type == "call" else models["bs_model"].put_price(spot_price, strike_price, time_to_maturity, risk_free_rate, np.sqrt(theta))
            
            comparison_data = [
                {"Model": "Black-Scholes", "Price": f"${bs_price:.4f}", "Method": "Constant Vol", "Time": "<1ms"}
            ]
            
            if "characteristic_function" in results:
                comparison_data.append({
                    "Model": "Heston (CF)", 
                    "Price": f"${results['characteristic_function']['price']:.4f}",
                    "Method": results['characteristic_function']['method'],
                    "Time": f"{results['characteristic_function']['execution_time_ms']:.1f}ms"
                })
            
            if "monte_carlo" in results:
                comparison_data.append({
                    "Model": "Heston (MC)",
                    "Price": f"${results['monte_carlo']['price']:.4f}",
                    "Method": results['monte_carlo']['method'],
                    "Time": f"{results['monte_carlo']['execution_time_ms']:.1f}ms"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # Stochastic volatility impact
            if "characteristic_function" in results:
                heston_price = results["characteristic_function"]["price"]
                stoch_vol_premium = heston_price - bs_price
                stoch_vol_premium_pct = (stoch_vol_premium / bs_price) * 100
                
                impact_cols = st.columns(3)
                with impact_cols[0]:
                    st.metric("Stochastic Vol Premium", f"${stoch_vol_premium:.4f}")
                with impact_cols[1]:
                    st.metric("Premium %", f"{stoch_vol_premium_pct:.2f}%")
                with impact_cols[2]:
                    current_vol = np.sqrt(v0) * 100
                    long_run_vol = np.sqrt(theta) * 100
                    st.metric("Current vs Long-run Vol", f"{current_vol:.1f}% vs {long_run_vol:.1f}%")
            
            # Store in session state for persistent display
            st.session_state["heston_results"] = results
        
        # Persistent results display
        if "heston_results" in st.session_state:
            results = st.session_state["heston_results"]
            
            # Monte Carlo statistics
            if "monte_carlo" in results:
                st.subheader("📊 Monte Carlo Simulation Details")
                mc_stats = results["monte_carlo"]["simulation_stats"]
                
                mc_cols = st.columns(4)
                with mc_cols[0]:
                    st.metric("Final Stock Price", f"${mc_stats['final_stock_mean']:.2f}")
                with mc_cols[1]:
                    st.metric("Final Variance", f"{mc_stats['final_variance_mean']:.4f}")
                with mc_cols[2]:
                    st.metric("Avg Volatility", f"{mc_stats['avg_volatility']:.1f}%")
                with mc_cols[3]:
                    st.metric("Paths Below Strike", f"{(mc_stats['paths_below_strike']/100000)*100:.1f}%")
        
        # Parameter sensitivity analysis
        st.subheader("🔬 Parameter Sensitivity Analysis")
        
        if st.button("Analyze Heston Parameter Sensitivity", key="heston_sensitivity_button"):
            with st.spinner("Running comprehensive sensitivity analysis..."):
                sensitivity_results = models["heston_model"].parameter_sensitivity_analysis(
                    spot_price, strike_price, time_to_maturity, risk_free_rate, option_type
                )
                
                # Create visualization
                sensitivity_fig = models["heston_model"].create_sensitivity_visualization(
                    sensitivity_results, option_type
                )
                st.plotly_chart(sensitivity_fig, use_container_width=True)
                
                # Key insights
                st.subheader("🔍 Key Parameter Insights")
                sensitivity_data = sensitivity_results['sensitivity_data']
                
                insights_cols = st.columns(2)
                with insights_cols[0]:
                    v0_impact = max([abs(x) for x in sensitivity_data['v0_analysis']['price_changes']])
                    sigma_impact = max([abs(x) for x in sensitivity_data['sigma_analysis']['price_changes']])
                    st.write(f"**Initial Variance (v₀)**: Up to {v0_impact:.1f}% price impact")
                    st.write(f"**Vol of Vol (σ)**: Up to {sigma_impact:.1f}% price impact")
                
                with insights_cols[1]:
                    rho_impact = max([abs(x) for x in sensitivity_data['rho_analysis']['price_changes']])
                    kappa_impact = max([abs(x) for x in sensitivity_data['kappa_analysis']['price_changes']])
                    st.write(f"**Correlation (ρ)**: Up to {rho_impact:.1f}% price impact")
                    st.write(f"**Mean Reversion (κ)**: Up to {kappa_impact:.1f}% price impact")
                
                # Find most sensitive parameter
                all_impacts = [
                    ("Initial Variance", v0_impact),
                    ("Vol of Vol", sigma_impact),
                    ("Correlation", rho_impact),
                    ("Mean Reversion", kappa_impact)
                ]
                most_sensitive = max(all_impacts, key=lambda x: x[1])
                st.success(f"🎯 **Most Sensitive Parameter**: {most_sensitive[0]} ({most_sensitive[1]:.1f}% max impact)")



if __name__ == "__main__":
    main()
