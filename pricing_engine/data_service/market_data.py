import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import random
from typing import Dict, Optional, List
from pathlib import Path
import certifi
from config.settings import settings

class MarketDataService:
    """Enhanced service with multiple fallback strategies"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 1800  # 30 minutes
        self.last_request_time = 0
        self.min_request_interval = 3
        self.request_count = 0
        self.daily_limit = 100
        self.last_reset = datetime.now().date()
        
        # Setup cache directory
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.disk_cache_file = self.cache_dir / "stock_quotes.json"
        
        # API providers
        self.api_providers = [
            {
                "name": "Alpha Vantage",
                "enabled": True,
                "daily_limit": 25,
                "calls_today": 0,
                "last_reset": datetime.now().date()
            },
            {
                "name": "Polygon",
                "enabled": True,
                "daily_limit": 5,
                "calls_today": 0,
                "last_reset": datetime.now().date()
            },
            {
                "name": "Financial Modeling Prep",
                "enabled": True,
                "daily_limit": 250,
                "calls_today": 0,
                "last_reset": datetime.now().date()
            }
        ]
    
    def _save_disk_cache(self, symbol: str, price: float):
        """Save successful price to disk cache"""
        try:
            data = {}
            if self.disk_cache_file.exists():
                data = json.loads(self.disk_cache_file.read_text())
            
            data[symbol] = {
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().date().isoformat()
            }
            
            self.disk_cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Could not save cache for {symbol}: {e}")
    
    def _load_disk_cache(self, symbol: str) -> Optional[float]:
        """Load cached price from disk if recent"""
        try:
            if not self.disk_cache_file.exists():
                return None
            
            data = json.loads(self.disk_cache_file.read_text())
            entry = data.get(symbol)
            
            if not entry or "price" not in entry:
                return None
            
            # Use cache if less than 2 hours old
            cache_time = datetime.fromisoformat(entry["timestamp"])
            if (datetime.now() - cache_time).total_seconds() < 7200:  # 2 hours
                print(f"Using cached price for {symbol}: ${entry['price']}")
                return float(entry["price"])
                
        except Exception as e:
            print(f"Could not load cache for {symbol}: {e}")
        
        return None
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> Dict:
        """Enhanced stock data fetching - stop after first success"""
        self._reset_daily_counters()
        
        if self.request_count >= self.daily_limit:
            cached_price = self._load_disk_cache(symbol)
            if cached_price:
                return self._build_data_dict(symbol, cached_price, "cached")
            return self._get_dummy_data(symbol)
        
        cache_key = f"{symbol}_{period}"
        current_time = time.time()
        
        # Check memory cache first
        if cache_key in self.cache:
            cache_data, cache_time = self.cache[cache_key]
            if current_time - cache_time < self.cache_timeout:
                return cache_data
        
        # Try strategies in order, STOP after first success
        strategies = [
            ("FMP Quote", self._try_fmp_quote),
            ("yfinance explicit range", self._try_yfinance_explicit_range),
            ("yfinance download", self._try_yfinance_download),
            ("yfinance fast_info", self._try_yfinance_fast_info),
            ("disk cache", lambda s: (self._load_disk_cache(s), "cached")),
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                print(f"Trying {strategy_name} for {symbol}...")
                
                if strategy_name == "disk cache":
                    price = strategy_func(symbol)
                    if price[0] is not None:
                        data = self._build_data_dict(symbol, price[0], price[1])
                        self.cache[cache_key] = (data, current_time)
                        return data
                else:
                    price, data_source = strategy_func(symbol)
                    if price is not None:
                        # SUCCESS - build data and cache it
                        data = self._build_data_dict(symbol, price, data_source)
                        self.cache[cache_key] = (data, current_time)
                        
                        # Save to disk cache if real data
                        if data_source not in ["cached", "dummy"]:
                            self._save_disk_cache(symbol, price)
                        
                        print(f"✅ Successfully got {symbol} price from {strategy_name}")
                        return data
            except Exception as e:
                print(f"{strategy_name} failed for {symbol}: {e}")
                continue
        
        # All strategies failed
        print(f"All strategies failed for {symbol}, using dummy data")
        return self._get_dummy_data(symbol)
    
    def _try_yfinance_explicit_range(self, symbol: str) -> tuple[Optional[float], str]:
        """Try yfinance with explicit date range"""
        try:
            print(f"Trying yfinance explicit range for {symbol}...")
            
            # Rate limiting
            self._apply_rate_limiting()
            
            ticker = yf.Ticker(symbol)
            
            # Use explicit date range (last 10 days to tomorrow)
            today = datetime.now().date()
            start = (today - timedelta(days=10)).isoformat()
            end = (today + timedelta(days=1)).isoformat()
            
            hist = ticker.history(start=start, end=end, auto_adjust=False, timeout=10)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            if not hist.empty and 'Close' in hist.columns:
                price = float(hist['Close'].dropna().iloc[-1])
                print(f"✅ Got {symbol} price via explicit range: ${price}")
                return price, "yfinance_explicit"
                
        except Exception as e:
            print(f"Explicit range failed for {symbol}: {e}")
        
        return None, "failed"
    
    def _try_yfinance_download(self, symbol: str) -> tuple[Optional[float], str]:
        """Try yfinance download method"""
        try:
            print(f"Trying yfinance download for {symbol}...")
            
            self._apply_rate_limiting()
            
            today = datetime.now().date()
            start = (today - timedelta(days=10)).isoformat()
            end = (today + timedelta(days=1)).isoformat()
            
            data = yf.download(symbol, start=start, end=end, progress=False, timeout=10)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            if not data.empty and 'Close' in data.columns:
                price = float(data['Close'].dropna().iloc[-1])
                print(f"✅ Got {symbol} price via download: ${price}")
                return price, "yfinance_download"
                
        except Exception as e:
            print(f"Download method failed for {symbol}: {e}")
        
        return None, "failed"
    
    def _try_yfinance_fast_info(self, symbol: str) -> tuple[Optional[float], str]:
        """Try yfinance fast_info"""
        try:
            print(f"Trying yfinance fast_info for {symbol}...")
            
            self._apply_rate_limiting()
            
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            # Try different price attributes
            for attr in ['last_price', 'regularMarketPrice', 'price']:
                if hasattr(fast_info, attr):
                    price = getattr(fast_info, attr)
                    if price and price > 0:
                        price = float(price)
                        print(f"✅ Got {symbol} price via fast_info.{attr}: ${price}")
                        return price, "yfinance_fast_info"
                        
        except Exception as e:
            print(f"Fast_info failed for {symbol}: {e}")
        
        return None, "failed"
    
    def _try_fmp_quote(self, symbol: str) -> tuple[Optional[float], str]:
        """Try Financial Modeling Prep quote endpoint"""
        try:
            print(f"Trying FMP quote for {symbol}...")
            
            response = requests.get(
                f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}",
                params={"apikey": settings.FMP_API_KEY},
                timeout=10,
                verify=certifi.where()
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0 and 'price' in data[0]:
                    price = float(data[0]['price'])
                    if price > 0:
                        print(f"✅ Got {symbol} price via FMP: ${price}")
                        return price, "fmp"
                        
        except Exception as e:
            print(f"FMP quote failed for {symbol}: {e}")
        
        return None, "failed"
    
    def _apply_rate_limiting(self):
        """Apply rate limiting with jitter"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # Add jitter to avoid synchronized requests
        time.sleep(random.uniform(0.5, 1.5))
    
    def _build_data_dict(self, symbol: str, price: float, data_source: str) -> Dict:
        """Build standardized data dictionary"""
        return {
            "symbol": symbol,
            "current_price": round(float(price), 2),
            "open": round(float(price) * 0.995, 2),
            "high": round(float(price) * 1.015, 2),
            "low": round(float(price) * 0.985, 2),
            "volume": 25000000,
            "timestamp": datetime.now().isoformat(),
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "52_week_high": round(float(price) * 1.3, 2),
            "52_week_low": round(float(price) * 0.8, 2),
            "data_source": data_source
        }
    
    def _reset_daily_counters(self):
        """Reset API call counters daily"""
        today = datetime.now().date()
        if self.last_reset < today:
            self.request_count = 0
            self.last_reset = today
            
        for provider in self.api_providers:
            if provider["last_reset"] < today:
                provider["calls_today"] = 0
                provider["last_reset"] = today
    
    def _exponential_backoff(self, attempt: int, max_delay: int = 60):
        """Exponential backoff with jitter"""
        delay = min(max_delay, (2 ** attempt) + random.uniform(0, 1))
        time.sleep(delay)
    
    # Keep all your existing methods for option chains, risk-free rate, etc.
    # Just replace the get_stock_data method above
    
    def get_risk_free_rate(self) -> float:
        """Enhanced risk-free rate with multiple fallbacks"""
        try:
            cache_key = "risk_free_rate"
            current_time = time.time()
            
            # Check cache
            if cache_key in self.cache:
                cache_data, cache_time = self.cache[cache_key]
                if current_time - cache_time < 3600:  # 1 hour cache
                    return cache_data
            
            # Try yfinance for ^TNX
            rate = self._try_treasury_yfinance()
            
            # Try FMP as fallback
            if rate is None and settings.FMP_API_KEY:
                rate = self._try_treasury_fmp()
            
            # Use default if all fail
            if rate is None:
                rate = 0.05
                print("Using default risk-free rate: 5%")
            
            # Cache the result
            self.cache[cache_key] = (rate, current_time)
            return rate
            
        except Exception as e:
            print(f"Error in risk-free rate calculation: {str(e)}")
            return 0.05
    
    def _try_treasury_yfinance(self) -> Optional[float]:
        """Try to get treasury rate from yfinance"""
        try:
            treasury = yf.Ticker("^TNX")
            
            # Try explicit range first
            today = datetime.now().date()
            start = (today - timedelta(days=10)).isoformat()
            end = (today + timedelta(days=1)).isoformat()
            
            hist = treasury.history(start=start, end=end, timeout=10)
            
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1]) / 100
                rate = max(rate, 0.01)  # Minimum 1%
                print(f"✅ Got treasury rate from yfinance: {rate*100:.2f}%")
                return rate
                
        except Exception as e:
            print(f"Treasury yfinance failed: {e}")
        
        return None
    
    def _try_treasury_fmp(self) -> Optional[float]:
        """Try to get treasury rate from FMP"""
        try:
            # FMP treasury endpoint
            today = datetime.now().date()
            from_date = (today - timedelta(days=5)).isoformat()
            
            response = requests.get(
                "https://financialmodelingprep.com/api/v4/treasury",
                params={
                    "from": from_date,
                    "to": today.isoformat(),
                    "apikey": settings.FMP_API_KEY
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Look for 10-year rate
                    for entry in data:
                        if 'year10' in entry and entry['year10']:
                            rate = float(entry['year10']) / 100
                            print(f"✅ Got treasury rate from FMP: {rate*100:.2f}%")
                            return max(rate, 0.01)
                            
        except Exception as e:
            print(f"Treasury FMP failed: {e}")
        
        return None

    def get_option_chain(self, symbol: str, expiry_date: str = None) -> Dict:
        """Get option chain with provider fallback and consistent spot pricing"""
        print(f"Fetching option chain for {symbol}...")
        
        # Get consistent spot price first
        stock_data = self.get_stock_data(symbol)
        consistent_spot = stock_data['current_price']
        
        # Try each provider in order
        provider = self._get_available_provider()
        
        if provider:
            try:
                if provider["name"] == "Polygon":
                    result = self._get_polygon_options(symbol, consistent_spot)
                    self._increment_provider_usage("Polygon")
                    return result
                elif provider["name"] == "Financial Modeling Prep":
                    result = self._get_fmp_options(symbol, consistent_spot)
                    self._increment_provider_usage("Financial Modeling Prep")
                    return result
                elif provider["name"] == "Alpha Vantage":
                    result = self._get_alphavantage_options(symbol, consistent_spot)
                    self._increment_provider_usage("Alpha Vantage")
                    return result
            except Exception as e:
                print(f"Error with {provider['name']}: {str(e)}")
        
        # All APIs failed, use synthetic data with consistent spot
        print("All API providers failed or exhausted, using synthetic data...")
        return self._get_synthetic_options(symbol, consistent_spot)
    
    def _get_available_provider(self) -> Optional[Dict]:
        """Get next available API provider"""
        self._reset_daily_counters()
        
        for provider in self.api_providers:
            if provider["enabled"] and provider["calls_today"] < provider["daily_limit"]:
                return provider
        return None
    
    def _increment_provider_usage(self, provider_name: str):
        """Increment usage counter for provider"""
        for provider in self.api_providers:
            if provider["name"] == provider_name:
                provider["calls_today"] += 1
                break
    
    def _get_polygon_options(self, symbol: str, consistent_spot: float) -> Dict:
        """Updated Polygon options with consistent spot price"""
        api_key = settings.POLYGON_API_KEY
        if not api_key:
            raise Exception("Polygon API key not set")

        try:
            # Use consistent spot price instead of fetching again
            spot = consistent_spot
            
            # Get contracts with simplified approach
            r_contracts = requests.get(
                f"https://api.polygon.io/v3/reference/options/contracts",
                params={
                    "underlying_ticker": symbol,
                    "limit": 100,
                    "sort": "expiration_date",
                    "order": "asc",
                    "apiKey": api_key,
                },
                timeout=15,
            )
            
            if r_contracts.status_code != 200:
                raise Exception(f"Polygon contracts error {r_contracts.status_code}")
            
            contracts = r_contracts.json().get("results", [])
            if not contracts:
                raise Exception("No contracts found")
            
            # Process contracts into our format
            options_data = []
            now_date = datetime.now().date()
            
            # Group by expiry and limit to prevent too many calls
            expiry_groups = {}
            for contract in contracts[:50]:  # Limit to first 50 contracts
                exp_date = contract.get("expiration_date")
                if exp_date:
                    try:
                        exp_dt = datetime.strptime(exp_date, "%Y-%m-%d").date()
                        if exp_dt >= now_date:
                            if exp_dt not in expiry_groups:
                                expiry_groups[exp_dt] = []
                            expiry_groups[exp_dt].append(contract)
                    except:
                        continue
            
            # Process only the first 2 expiries to stay within limits
            processed_expiries = 0
            for exp_dt in sorted(expiry_groups.keys()):
                if processed_expiries >= 2:
                    break
                    
                time_to_expiry = max((exp_dt - now_date).days / 365.0, 1/365)
                
                for contract in expiry_groups[exp_dt][:10]:  # Limit contracts per expiry
                    strike = contract.get("strike_price")
                    contract_type = contract.get("contract_type", "").lower()
                    
                    if not strike or not contract_type:
                        continue
                    
                    try:
                        strike = float(strike)
                        # Filter strikes within reasonable range
                        if strike < spot * 0.7 or strike > spot * 1.3:
                            continue
                        
                        # Generate synthetic price for demo (since quote endpoints are limited)
                        from ..models.black_scholes import BlackScholesModel
                        bs_model = BlackScholesModel()
                        
                        if contract_type == "call":
                            market_price = bs_model.call_price(spot, strike, time_to_expiry, 0.05, 0.25)
                        else:
                            market_price = bs_model.put_price(spot, strike, time_to_expiry, 0.05, 0.25)
                        
                        # Add some market noise
                        market_price *= np.random.uniform(0.95, 1.05)
                        market_price = max(0.01, market_price)
                        
                        bid_ask_spread = market_price * 0.02
                        
                        options_data.append({
                            "symbol": symbol,
                            "strike": float(strike),
                            "option_type": contract_type,
                            "expiry_date": exp_dt.strftime('%Y-%m-%d'),
                            "time_to_expiry": time_to_expiry,
                            "market_price": round(float(market_price), 2),
                            "bid": round(float(market_price - bid_ask_spread/2), 2),
                            "ask": round(float(market_price + bid_ask_spread/2), 2),
                            "volume": np.random.randint(10, 500),
                            "open_interest": np.random.randint(100, 2000),
                        })
                    except Exception as e:
                        continue
                
                processed_expiries += 1
            
            if not options_data:
                raise Exception("No valid options processed")
            
            return {
                "symbol": symbol,
                "current_price": spot,
                "timestamp": datetime.now().isoformat(),
                "data_source": "polygon",
                "options": options_data,
            }
            
        except Exception as e:
            raise Exception(f"Polygon error: {str(e)}")
    
    def _get_fmp_options(self, symbol: str, consistent_spot: float) -> Dict:
        """Updated FMP options with consistent spot price"""
        api_key = settings.FMP_API_KEY
        if not api_key:
            raise Exception("FMP API key not set")

        try:
            # Use consistent spot price
            spot = consistent_spot
            
            # Try FMP options chain endpoint
            r_chain = requests.get(
                "https://financialmodelingprep.com/api/v3/options/chain",
                params={"symbol": symbol, "apikey": api_key},
                timeout=15,
            )
            
            if r_chain.status_code != 200:
                raise Exception(f"FMP chain error {r_chain.status_code}")
            
            chain_data = r_chain.json()
            options_data = []
            now_date = datetime.now().date()
            
            # Process chain data
            if isinstance(chain_data, list):
                data_to_process = chain_data
            elif isinstance(chain_data, dict) and "options" in chain_data:
                data_to_process = chain_data["options"]
            else:
                data_to_process = [chain_data] if chain_data else []
            
            for item in data_to_process[:50]:  # Limit to prevent overload
                strike = item.get("strike")
                option_type = item.get("type", "").lower()
                expiry_str = item.get("expirationDate") or item.get("expiration")
                
                if not all([strike, option_type, expiry_str]):
                    continue
                
                try:
                    strike = float(strike)
                    if strike < spot * 0.7 or strike > spot * 1.3:
                        continue
                    
                    # Parse expiry date
                    if "T" in expiry_str:
                        exp_dt = datetime.strptime(expiry_str.split("T")[0], "%Y-%m-%d").date()
                    else:
                        exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    
                    if exp_dt < now_date:
                        continue
                    
                    time_to_expiry = max((exp_dt - now_date).days / 365.0, 1/365)
                    
                    # Get market price from API or synthesize
                    market_price = item.get("last") or item.get("price")
                    bid = item.get("bid")
                    ask = item.get("ask")
                    
                    if market_price:
                        market_price = float(market_price)
                    else:
                        # Synthesize price
                        from ..models.black_scholes import BlackScholesModel
                        bs_model = BlackScholesModel()
                        
                        if option_type == "call":
                            market_price = bs_model.call_price(spot, strike, time_to_expiry, 0.05, 0.25)
                        else:
                            market_price = bs_model.put_price(spot, strike, time_to_expiry, 0.05, 0.25)
                        
                        market_price *= np.random.uniform(0.95, 1.05)
                    
                    market_price = max(0.01, market_price)
                    
                    if not bid or not ask:
                        spread = market_price * 0.02
                        bid = market_price - spread/2
                        ask = market_price + spread/2
                    else:
                        bid = float(bid)
                        ask = float(ask)
                    
                    options_data.append({
                        "symbol": symbol,
                        "strike": float(strike),
                        "option_type": option_type,
                        "expiry_date": exp_dt.strftime('%Y-%m-%d'),
                        "time_to_expiry": time_to_expiry,
                        "market_price": round(float(market_price), 2),
                        "bid": round(float(bid), 2),
                        "ask": round(float(ask), 2),
                        "volume": int(item.get("volume", np.random.randint(10, 500))),
                        "open_interest": int(item.get("openInterest", np.random.randint(100, 2000))),
                    })
                    
                except Exception as e:
                    continue
            
            if not options_data:
                raise Exception("No valid options processed from FMP")
            
            return {
                "symbol": symbol,
                "current_price": spot,
                "timestamp": datetime.now().isoformat(),
                "data_source": "fmp",
                "options": options_data,
            }
            
        except Exception as e:
            raise Exception(f"FMP error: {str(e)}")
    
    def _get_alphavantage_options(self, symbol: str, consistent_spot: float) -> Dict:
        """Alpha Vantage fallback (limited options support in free tier)"""
        # Alpha Vantage free tier has very limited options support
        # Fall back to synthetic with consistent spot
        return self._get_synthetic_options(symbol, consistent_spot)
    
    def _get_synthetic_options(self, symbol: str, consistent_spot: float = None) -> Dict:
        """Generate synthetic options with consistent spot price"""
        if consistent_spot is None:
            stock_data = self.get_stock_data(symbol)
            consistent_spot = stock_data['current_price']
        
        print(f"Generating synthetic options data for {symbol} at spot ${consistent_spot}")
        
        options_data = []
        
        # Create multiple expiration dates
        expiry_dates = [
            datetime.now() + timedelta(days=30),   # 1 month
            datetime.now() + timedelta(days=60),   # 2 months  
            datetime.now() + timedelta(days=90),   # 3 months
        ]
        
        # Create strikes around current price  
        strike_range = np.arange(
            max(5, consistent_spot * 0.8), 
            consistent_spot * 1.2, 
            max(1, consistent_spot * 0.025)  # 2.5% intervals
        )
        strike_range = np.round(strike_range).astype(int)
        
        from ..models.black_scholes import BlackScholesModel
        bs_model = BlackScholesModel()
        
        for expiry in expiry_dates:
            time_to_expiry = (expiry - datetime.now()).days / 365.0
            
            for strike in strike_range:
                for option_type in ['call', 'put']:
                    # Use consistent risk-free rate
                    risk_free_rate = self.get_risk_free_rate()
                    volatility = 0.25  # Default volatility
                    
                    if option_type == 'call':
                        theoretical_price = bs_model.call_price(
                            consistent_spot, strike, time_to_expiry, risk_free_rate, volatility
                        )
                    else:
                        theoretical_price = bs_model.put_price(
                            consistent_spot, strike, time_to_expiry, risk_free_rate, volatility
                        )
                    
                    # Add realistic market noise
                    noise_factor = np.random.uniform(0.98, 1.02)  # Reduced noise for consistency
                    market_price = max(0.01, theoretical_price * noise_factor)
                    
                    # Realistic bid-ask spread
                    bid_ask_spread = market_price * 0.015  # 1.5% spread
                    
                    options_data.append({
                        'symbol': symbol,
                        'strike': float(strike),
                        'option_type': option_type,
                        'expiry_date': expiry.strftime('%Y-%m-%d'),
                        'time_to_expiry': time_to_expiry,
                        'market_price': round(market_price, 2),
                        'bid': round(market_price - bid_ask_spread/2, 2),
                        'ask': round(market_price + bid_ask_spread/2, 2),
                        'volume': np.random.randint(50, 500),
                        'open_interest': np.random.randint(200, 2000),
                    })
        
        return {
            'symbol': symbol,
            'current_price': consistent_spot,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'synthetic',
            'options': options_data
        }
    
    def calculate_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility with better error handling"""
        try:
            cache_key = f"vol_{symbol}_{days}"
            current_time = time.time()
            
            # Check cache
            if cache_key in self.cache:
                cache_data, cache_time = self.cache[cache_key]
                if current_time - cache_time < self.cache_timeout:
                    return cache_data
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{max(days, 30)}d")
            
            if len(hist) < 5:  # Need minimum data points
                return 0.25  # Default volatility
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) < 3:
                return 0.25
            
            # Annualized volatility
            volatility = float(returns.std() * np.sqrt(252))
            volatility = min(max(volatility, 0.05), 2.0)  # Cap between 5% and 200%
            
            # Cache the result
            self.cache[cache_key] = (volatility, current_time)
            return volatility
            
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.25
    
    def get_risk_free_rate(self) -> float:
        """Get risk-free rate with fallback"""
        try:
            cache_key = "risk_free_rate"
            current_time = time.time()
            
            # Check cache
            if cache_key in self.cache:
                cache_data, cache_time = self.cache[cache_key]
                if current_time - cache_time < 3600:  # 1 hour cache
                    return cache_data
            
            # Try to get 10-year Treasury rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1]) / 100
                rate = max(rate, 0.01)  # Minimum 1%
                
                # Cache the result
                self.cache[cache_key] = (rate, current_time)
                return rate
                
        except Exception as e:
            print(f"Error fetching risk-free rate: {str(e)}")
        
        # Default fallback
        default_rate = 0.05
        self.cache[cache_key] = (default_rate, time.time())
        return default_rate
    
    def _get_dummy_data(self, symbol: str) -> Dict:
        """Improved dummy data with realistic prices"""
        base_prices = {
            "AAPL": 185.00, "MSFT": 420.00, "GOOGL": 2800.00,
            "AMZN": 3200.00, "TSLA": 900.00, "META": 320.00,
            "NVDA": 450.00, "JPM": 155.00, "JNJ": 165.00, "V": 285.00
        }
        
        base_price = base_prices.get(symbol, 150.00)
        
        return {
            "symbol": symbol,
            "current_price": base_price,
            "open": round(base_price * 0.995, 2),
            "high": round(base_price * 1.015, 2),
            "low": round(base_price * 0.985, 2),
            "volume": 25000000,
            "timestamp": datetime.now().isoformat(),
            "market_cap": "2.5T",
            "pe_ratio": 28.5,
            "52_week_high": round(base_price * 1.4, 2),
            "52_week_low": round(base_price * 0.7, 2),
            "data_source": "dummy"
        }
