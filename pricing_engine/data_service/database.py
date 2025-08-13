from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
import json
from config.settings import settings

class DatabaseService:
    """Enhanced database service for options pricing application"""
    
    def __init__(self):
        self.client = MongoClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]
        
        # Collections
        self.calculations = self.db.calculations_history
        self.user_preferences = self.db.user_preferences
        self.market_data_cache = self.db.market_data_cache
        self.model_performance = self.db.model_performance
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Calculations history indexes
            self.calculations.create_index([("timestamp", -1)])
            self.calculations.create_index([("symbol", 1), ("timestamp", -1)])
            self.calculations.create_index([("option_type", 1), ("timestamp", -1)])
            
            # Market data cache indexes
            self.market_data_cache.create_index([("symbol", 1), ("timestamp", -1)])
            self.market_data_cache.create_index([("timestamp", -1)], expireAfterSeconds=86400)  # 24-hour TTL
            
            print("✅ Database indexes created successfully")
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
    
    def save_calculation(self, calculation_data: Dict) -> str:
        """Save option pricing calculation to database"""
        try:
            # Add metadata
            calculation_data.update({
                "timestamp": datetime.utcnow(),
                "session_id": calculation_data.get("session_id", "default"),
                "version": "2.0"
            })
            
            # Insert calculation
            result = self.calculations.insert_one(calculation_data)
            
            # Update model performance tracking
            self._update_model_performance(calculation_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error saving calculation: {e}")
            return None
    
    def get_calculation_history(self, symbol: str = None, days: int = 30, 
                              option_type: str = None) -> List[Dict]:
        """Retrieve calculation history with filters"""
        try:
            # Build query
            query = {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
            }
            
            if symbol:
                query["symbol"] = symbol.upper()
            if option_type:
                query["option_type"] = option_type.lower()
            
            # Get results sorted by timestamp (newest first)
            results = list(self.calculations.find(query).sort("timestamp", -1).limit(1000))
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                result["_id"] = str(result["_id"])
                if "timestamp" in result and hasattr(result["timestamp"], "isoformat"):
                    result["timestamp"] = result["timestamp"].isoformat()
            
            return results
            
        except Exception as e:
            print(f"Error retrieving calculation history: {e}")
            return []
    
    def get_calculation_statistics(self, symbol: str = None, days: int = 30) -> Dict:
        """Get calculation statistics and analytics"""
        try:
            # Build base query
            base_query = {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
            }
            if symbol:
                base_query["symbol"] = symbol.upper()
            
            # Aggregation pipeline for statistics
            pipeline = [
                {"$match": base_query},
                {"$group": {
                    "_id": None,
                    "total_calculations": {"$sum": 1},
                    "unique_symbols": {"$addToSet": "$symbol"},
                    "avg_black_scholes_price": {"$avg": "$results.black_scholes.price"},
                    "avg_monte_carlo_price": {"$avg": "$results.monte_carlo.price"},
                    "avg_binomial_price": {"$avg": "$results.binomial.price"},
                    "avg_execution_time": {"$avg": "$results.monte_carlo.execution_time_ms"},
                    "call_count": {
                        "$sum": {"$cond": [{"$eq": ["$option_type", "call"]}, 1, 0]}
                    },
                    "put_count": {
                        "$sum": {"$cond": [{"$eq": ["$option_type", "put"]}, 1, 0]}
                    }
                }}
            ]
            
            result = list(self.calculations.aggregate(pipeline))
            
            if result:
                stats = result[0]
                stats["unique_symbols_count"] = len(stats.get("unique_symbols", []))
                return stats
            else:
                return {
                    "total_calculations": 0,
                    "unique_symbols_count": 0,
                    "call_count": 0,
                    "put_count": 0
                }
                
        except Exception as e:
            print(f"Error getting calculation statistics: {e}")
            return {}
    
    def export_calculations_to_csv(self, symbol: str = None, days: int = 30) -> pd.DataFrame:
        """Export calculation history to pandas DataFrame for CSV export"""
        try:
            calculations = self.get_calculation_history(symbol, days)
            
            if not calculations:
                return pd.DataFrame()
            
            # Flatten the nested data structure
            flattened_data = []
            for calc in calculations:
                row = {
                    "timestamp": calc.get("timestamp"),
                    "symbol": calc.get("symbol"),
                    "option_type": calc.get("option_type"),
                    "spot_price": calc.get("parameters", {}).get("spot_price"),
                    "strike_price": calc.get("parameters", {}).get("strike_price"),
                    "time_to_maturity": calc.get("parameters", {}).get("time_to_maturity"),
                    "volatility": calc.get("parameters", {}).get("volatility"),
                    "risk_free_rate": calc.get("parameters", {}).get("risk_free_rate"),
                    "data_source": calc.get("data_source"),
                }
                
                # Add model results
                results = calc.get("results", {})
                for model_name, model_data in results.items():
                    if isinstance(model_data, dict):
                        row[f"{model_name}_price"] = model_data.get("price")
                        row[f"{model_name}_execution_time_ms"] = model_data.get("execution_time_ms")
                
                # Add Greeks
                greeks = calc.get("greeks", {})
                for greek_name, greek_value in greeks.items():
                    row[f"greek_{greek_name}"] = greek_value
                
                flattened_data.append(row)
            
            return pd.DataFrame(flattened_data)
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return pd.DataFrame()
    
    def save_user_preferences(self, user_id: str, preferences: Dict):
        """Save user preferences"""
        try:
            self.user_preferences.replace_one(
                {"user_id": user_id},
                {
                    "user_id": user_id,
                    "preferences": preferences,
                    "last_updated": datetime.utcnow()
                },
                upsert=True
            )
        except Exception as e:
            print(f"Error saving user preferences: {e}")
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        try:
            result = self.user_preferences.find_one({"user_id": user_id})
            return result.get("preferences", {}) if result else {}
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {}
    
    def _update_model_performance(self, calculation_data: Dict):
        """Update model performance tracking"""
        try:
            performance_data = {
                "timestamp": datetime.utcnow(),
                "symbol": calculation_data.get("symbol"),
                "option_type": calculation_data.get("option_type"),
                "models_used": list(calculation_data.get("results", {}).keys()),
                "execution_times": {
                    model: data.get("execution_time_ms", 0)
                    for model, data in calculation_data.get("results", {}).items()
                    if isinstance(data, dict)
                }
            }
            
            self.model_performance.insert_one(performance_data)
        except Exception as e:
            print(f"Error updating model performance: {e}")
    
    def get_recent_calculations(self, limit: int = 10) -> List[Dict]:
        """Get most recent calculations for quick access"""
        try:
            results = list(
                self.calculations.find()
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            # Format for display
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": str(result["_id"]),
                    "timestamp": result["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": result.get("symbol", "N/A"),
                    "option_type": result.get("option_type", "N/A").title(),
                    "strike": result.get("parameters", {}).get("strike_price", "N/A"),
                    "price_bs": result.get("results", {}).get("black_scholes", {}).get("price", "N/A")
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error getting recent calculations: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old calculation data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Remove old calculations
            result = self.calculations.delete_many({"timestamp": {"$lt": cutoff_date}})
            print(f"Cleaned up {result.deleted_count} old calculations")
            
            # Remove old performance data
            perf_result = self.model_performance.delete_many({"timestamp": {"$lt": cutoff_date}})
            print(f"Cleaned up {perf_result.deleted_count} old performance records")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _serialize_for_mongo(self, data):
        """Convert numpy arrays and other non-serializable objects for MongoDB"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, dict):
            return {k: self._serialize_for_mongo(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_mongo(item) for item in data]
        elif hasattr(data, 'item'):  # numpy scalars
            return data.item()
        return data
    
    def save_calculation(self, calculation_data: Dict) -> str:
        """Save option pricing calculation to database"""
        try:
            # Serialize numpy arrays before saving
            serialized_data = self._serialize_for_mongo(calculation_data)
            
            # Add metadata
            serialized_data.update({
                "timestamp": datetime.utcnow(),
                "session_id": serialized_data.get("session_id", "default"),
                "version": "2.0"
            })
            
            # Insert calculation
            result = self.calculations.insert_one(serialized_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error saving calculation: {e}")
            return None
