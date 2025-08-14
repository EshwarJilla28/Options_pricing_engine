# Advanced Options Pricing Engine

A professional, learning-focused options pricing platform that combines multiple pricing models, stochastic processes, and an educational UI. Built with Python and Streamlit, it includes Black-Scholes, Binomial Trees, Monte Carlo, Merton Jump Diffusion, and Heston Stochastic Volatility, with database-backed persistence and realistic synthetic chains for reliable demos when live options data isn't available.

## Key Differentiators
- **Multi-model core**: Black-Scholes (analytical), Binomial (numerical), Monte Carlo (simulation), Merton (jumps), Heston (stochastic vol)
- **Educational UX**: Model Validation tab, parameter sensitivity, tooltips, and convergence visualizations
- **Hybrid data strategy**: Real spot via FMP with caching; synthetic options chain for robust comparisons when APIs are limited
- **Persistence**: MongoDB storage for calculations; explicit manual save and CSV export
- **Professional structure**: Clear tabs for Pricing, Model Analysis, Implied Volatility, Model Validation, Merton, and Heston

## Demo Overview
- **Pricing Engine**: Side-by-side pricing from selected models; Greeks; market vs model comparison; manual save and CSV export
- **Model Analysis**: Binomial tree exploration, European vs American, convergence to Black-Scholes
- **Implied Volatility**: IV solver (Brent/Newton), volatility smiles from synthetic chains
- **Model Validation**: Comprehensive comparison across models with convergence/error/time analyses
- **Merton**: Series-expansion and Monte Carlo implementations; jump parameter sensitivity
- **Heston**: Characteristic-function (corrected Carr–Madan) and Monte Carlo; parameter sensitivity and Feller condition check

## Quick Start (5 minutes)

### 1. Clone and enter
```bash
git clone <your-repo-url>
cd advanced_options_engine
```

### 2. Create environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
Create a `.env` file in the repo root:
```
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=options_pricing
FMP_API_KEY=<your_fmp_key_or_empty>
```

### 5. Start MongoDB (optional but recommended)
```bash
brew services start mongodb-community  # macOS/Homebrew
# or run mongod manually
```

### 6. Run the app
```bash
streamlit run streamlit_app/main.py
```

## First-Run Tips
- If Yahoo endpoints fail, the app automatically uses FMP for spot and generates realistic synthetic option chains
- Use "Save to Database" to persist a calculation; use "Export CSV" for quick downloads
- Tooltips (the small "i") explain each input/output

## Features

### Pricing Models
- **Black-Scholes** (analytical)
- **Monte Carlo** (path simulation)
- **Binomial Trees** (European and American)
- **Merton Jump Diffusion** (series expansion + MC)
- **Heston Stochastic Volatility** (characteristic function + MC)

### Implied Volatility
- Brent/Newton solvers
- Smile visualization with synthetic chains

### Model Validation
- Convergence analysis for Monte Carlo and Binomial
- Error vs Black-Scholes reference and timing

### Risk & Greeks
- Delta, Gamma, Theta, Vega, Rho; intuitive tooltips

### Data Layer
- Spot via FMP fallback with caching
- Synthetic options chain for robust demos
- MongoDB persistence for calculation history (manual save)

## Repository Structure
```
streamlit_app/
├── main.py
pricing_engine/
├── models/
│   ├── black_scholes.py
│   ├── binomial_tree.py
│   ├── monte_carlo.py
│   ├── greeks.py
│   ├── implied_vol.py
│   ├── merton_jump_diffusion.py
│   ├── heston_stochastic_volatility.py
│   └── synthetic_data.py
├── analytics/
│   └── model_validator.py
└── data_service/
    ├── market_data.py
    └── database.py
config/
└── settings.py
docs/
├── user_guide.md (to be added)
├── technical/ (Sphinx scaffold to be added)
└── financial/ (LaTeX scaffold to be added)
```

## Usage Guide (High-Level)

### Pricing Engine tab
- Choose symbol, option parameters, and models
- View model prices side-by-side; compare to market/synthetic prices; see Greeks; save/export results

### Model Analysis tab
- Inspect binomial trees; run convergence; compare with Black-Scholes references

### Implied Volatility tab
- Compute IV from market/synthetic prices; visualize volatility smiles

### Model Validation tab
- Comprehensive comparisons; convergence/error/time tradeoffs

### Merton tab
- Price via series expansion or MC; analyze jump parameter sensitivity

### Heston tab
- Price via characteristic function or MC; analyze parameter sensitivity; verify Feller condition

## Configuration

### ENV (.env)
- `MONGODB_URL`: MongoDB connection string (optional if not saving)
- `DATABASE_NAME`: Database name
- `FMP_API_KEY`: Optional; improves spot-data reliability

### Settings (config/settings.py)
- `SUPPORTED_STOCKS`: Tickers offered in the UI
- `DEFAULT_VOLATILITY` / `DEFAULT_RISK_FREE_RATE` (fallbacks)

## Troubleshooting
- **DuplicateWidgetID**: Ensure each widget has a unique key (handled in code)
- **yfinance empty data / "possibly delisted"**: Expected with Yahoo instability; FMP fallback + synthetic chains cover it
- **RecursionError in cache**: load_models() uses session state instead of @st.cache_resource to avoid hashing complex objects
- **Heston price looks off**: Use the corrected Carr–Madan implementation; if parameters cause instability, switch to Monte Carlo; confirm v0/theta are in variance terms (e.g., 0.04 = 20% vol)
- **"Clear All Data" crash**: Use the safe_clear_session_state() flow; it preserves models and resets user inputs

## Roadmap

### Near-term
- Portfolio risk (single-option first; VaR and stress testing)
- Professional risk dashboards

### Mid-term
- Calibration utilities (Heston/Merton) to market IV surfaces
- Additional exotics (Asian, Barrier, Lookback)

### Documentation
- Technical docs (Sphinx): API and architecture
- Financial docs (LaTeX): Theory and derivations
- End-user docs (Markdown): Tutorials and walkthroughs

## Contributing
- Fork, create a feature branch, submit a PR with a clear description and tests for numerical functions
- Follow existing code style and docstring conventions

## License
MIT

## Acknowledgments
- Inspired by classic derivatives texts (Hull, Gatheral) and practitioner implementations
- Thanks to open-source contributors in the Python quant ecosystem

## Documentation Links (incoming)
- **Technical Documentation** (Sphinx): `docs/technical/`
- **Financial/Mathematical Docs** (LaTeX): `docs/financial/`
- **End-User Guide** (Markdown): `docs/user_guide.md`