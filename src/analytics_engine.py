"""
Cannabis Analytics Engine
Performs various analytical calculations and trend analysis on cannabis data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class CannabisAnalyticsEngine:
    """Advanced analytics engine for cannabis market data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.stock_data = None
        self.legalization_data = None
        self.pricing_data = None
        
    def load_data(self):
        """Load all available data from CSV files"""
        try:
            # Find the most recent data files
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            
            # Load stock data
            stock_files = [f for f in csv_files if 'cannabis_stocks' in f]
            if stock_files:
                latest_stock = sorted(stock_files)[-1]
                self.stock_data = pd.read_csv(os.path.join(self.data_dir, latest_stock))
                self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
                print(f"Loaded stock data: {latest_stock}")
            
            # Load legalization data
            legal_files = [f for f in csv_files if 'legalization_data' in f]
            if legal_files:
                latest_legal = sorted(legal_files)[-1]
                self.legalization_data = pd.read_csv(os.path.join(self.data_dir, latest_legal))
                print(f"Loaded legalization data: {latest_legal}")
            
            # Load pricing data
            pricing_files = [f for f in csv_files if 'cannabis_pricing' in f]
            if pricing_files:
                latest_pricing = sorted(pricing_files)[-1]
                self.pricing_data = pd.read_csv(os.path.join(self.data_dir, latest_pricing))
                self.pricing_data['date'] = pd.to_datetime(self.pricing_data['date'])
                print(f"Loaded pricing data: {latest_pricing}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def calculate_stock_metrics(self) -> dict:
        """Calculate key stock market metrics"""
        if self.stock_data is None or self.stock_data.empty:
            return {}
        
        metrics = {}
        
        for symbol in self.stock_data['Symbol'].unique():
            stock_subset = self.stock_data[self.stock_data['Symbol'] == symbol].sort_values('Date')
            
            if len(stock_subset) < 2:
                continue
                
            # Calculate returns
            stock_subset = stock_subset.copy()
            stock_subset['Daily_Return'] = stock_subset['Close'].pct_change()
            
            # Calculate volatility (30-day rolling standard deviation)
            stock_subset['Volatility'] = stock_subset['Daily_Return'].rolling(window=30).std()
            
            # Calculate moving averages
            stock_subset['MA_20'] = stock_subset['Close'].rolling(window=20).mean()
            stock_subset['MA_50'] = stock_subset['Close'].rolling(window=50).mean()
            
            # Performance metrics
            total_return = (stock_subset['Close'].iloc[-1] / stock_subset['Close'].iloc[0] - 1) * 100
            avg_volume = stock_subset['Volume'].mean()
            current_price = stock_subset['Close'].iloc[-1]
            
            metrics[symbol] = {
                'total_return_pct': round(total_return, 2),
                'current_price': round(current_price, 2),
                'avg_daily_volume': int(avg_volume),
                'volatility': round(stock_subset['Volatility'].iloc[-1] * 100, 2) if not pd.isna(stock_subset['Volatility'].iloc[-1]) else 0,
                'price_trend': 'Up' if stock_subset['Close'].iloc[-1] > stock_subset['Close'].iloc[-10] else 'Down'
            }
        
        return metrics
    
    def analyze_legalization_impact(self) -> dict:
        """Analyze the impact of legalization on market trends"""
        if self.legalization_data is None:
            return {}
        
        analysis = {}
        
        # Timeline analysis
        medical_timeline = self.legalization_data[self.legalization_data['medical_legal'] == True]['medical_year'].value_counts().sort_index()
        recreational_timeline = self.legalization_data[
            (self.legalization_data['recreational_legal'] == True) & 
            (self.legalization_data['recreational_year'].notna())
        ]['recreational_year'].value_counts().sort_index()
        
        # Geographic analysis
        us_states = self.legalization_data[self.legalization_data['country'] == 'USA']
        international = self.legalization_data[self.legalization_data['country'] != 'USA']
        
        analysis['timeline'] = {
            'medical_by_year': medical_timeline.to_dict(),
            'recreational_by_year': recreational_timeline.to_dict()
        }
        
        analysis['geographic'] = {
            'us_medical_states': len(us_states[us_states['medical_legal'] == True]),
            'us_recreational_states': len(us_states[us_states['recreational_legal'] == True]),
            'total_us_states': len(us_states),
            'international_countries': len(international),
            'international_recreational': len(international[international['recreational_legal'] == True])
        }
        
        # Calculate adoption rates
        current_year = datetime.now().year
        recent_adoptions = self.legalization_data[
            (self.legalization_data['recreational_year'] >= current_year - 3) &
            (self.legalization_data['recreational_year'].notna())
        ]
        
        analysis['trends'] = {
            'recent_recreational_adoptions': len(recent_adoptions),
            'adoption_acceleration': len(recent_adoptions) > 5  # Arbitrary threshold
        }
        
        return analysis
    
    def analyze_pricing_trends(self) -> dict:
        """Analyze cannabis product pricing trends"""
        if self.pricing_data is None:
            return {}
        
        analysis = {}
        
        # Price by product type
        avg_prices = self.pricing_data.groupby('product_type')['price_per_gram'].agg(['mean', 'std', 'count']).round(2)
        analysis['price_by_product'] = avg_prices.to_dict('index')
        
        # Price by state
        state_prices = self.pricing_data.groupby('state')['price_per_gram'].agg(['mean', 'std', 'count']).round(2)
        analysis['price_by_state'] = state_prices.to_dict('index')
        
        # Medical vs Recreational pricing
        dispensary_prices = self.pricing_data.groupby('dispensary_type')['price_per_gram'].agg(['mean', 'std', 'count']).round(2)
        analysis['price_by_dispensary_type'] = dispensary_prices.to_dict('index')
        
        # Time-based trends
        self.pricing_data['month'] = self.pricing_data['date'].dt.to_period('M')
        monthly_trends = self.pricing_data.groupby('month')['price_per_gram'].mean()
        
        # Calculate price trend
        if len(monthly_trends) >= 6:
            recent_avg = monthly_trends.tail(3).mean()
            older_avg = monthly_trends.head(3).mean()
            price_trend = 'Increasing' if recent_avg > older_avg else 'Decreasing'
        else:
            price_trend = 'Insufficient data'
        
        analysis['trends'] = {
            'overall_price_trend': price_trend,
            'monthly_averages': {str(k): float(v) for k, v in monthly_trends.items()}
        }
        
        # Potency analysis
        potency_analysis = self.pricing_data.groupby('product_type').agg({
            'thc_percentage': ['mean', 'std'],
            'cbd_percentage': ['mean', 'std']
        }).round(2)
        
        analysis['potency'] = {}
        for product in potency_analysis.index:
            analysis['potency'][product] = {
                'thc_mean': float(potency_analysis.loc[product, ('thc_percentage', 'mean')]),
                'thc_std': float(potency_analysis.loc[product, ('thc_percentage', 'std')]),
                'cbd_mean': float(potency_analysis.loc[product, ('cbd_percentage', 'mean')]),
                'cbd_std': float(potency_analysis.loc[product, ('cbd_percentage', 'std')])
            }
        
        return analysis
    
    def perform_market_segmentation(self) -> dict:
        """Perform market segmentation analysis using clustering"""
        if self.stock_data is None or self.stock_data.empty:
            return {}
        
        # Prepare data for clustering
        stock_metrics = []
        symbols = []
        
        for symbol in self.stock_data['Symbol'].unique():
            stock_subset = self.stock_data[self.stock_data['Symbol'] == symbol].sort_values('Date')
            
            if len(stock_subset) < 30:  # Need sufficient data
                continue
                
            # Calculate features for clustering
            returns = stock_subset['Close'].pct_change().dropna()
            volume_norm = stock_subset['Volume'] / stock_subset['Volume'].mean()
            
            features = [
                returns.mean(),  # Average return
                returns.std(),   # Volatility
                volume_norm.mean(),  # Relative volume
                stock_subset['Close'].iloc[-1] / stock_subset['Close'].iloc[0],  # Total return ratio
                stock_subset['Close'].mean()  # Average price
            ]
            
            stock_metrics.append(features)
            symbols.append(symbol)
        
        if len(stock_metrics) < 3:
            return {'error': 'Insufficient data for clustering'}
        
        # Perform clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(stock_metrics)
        
        n_clusters = min(3, len(stock_metrics))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Organize results
        segmentation = {}
        for i in range(n_clusters):
            cluster_symbols = [symbols[j] for j in range(len(symbols)) if clusters[j] == i]
            cluster_features = [stock_metrics[j] for j in range(len(symbols)) if clusters[j] == i]
            
            if cluster_features:
                avg_features = np.mean(cluster_features, axis=0)
                segmentation[f'Cluster_{i+1}'] = {
                    'stocks': cluster_symbols,
                    'characteristics': {
                        'avg_return': round(avg_features[0], 4),
                        'volatility': round(avg_features[1], 4),
                        'relative_volume': round(avg_features[2], 2),
                        'total_return_ratio': round(avg_features[3], 2),
                        'avg_price': round(avg_features[4], 2)
                    }
                }
        
        return segmentation
    
    def predict_price_trends(self) -> dict:
        """Simple price trend prediction using linear regression"""
        if self.stock_data is None or self.stock_data.empty:
            return {}
        
        predictions = {}
        
        for symbol in self.stock_data['Symbol'].unique():
            stock_subset = self.stock_data[self.stock_data['Symbol'] == symbol].sort_values('Date')
            
            if len(stock_subset) < 30:
                continue
            
            # Prepare data
            stock_subset = stock_subset.copy()
            stock_subset['days_since_start'] = (stock_subset['Date'] - stock_subset['Date'].iloc[0]).dt.days
            
            X = stock_subset[['days_since_start']].values
            y = stock_subset['Close'].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 30 days
            last_day = stock_subset['days_since_start'].max()
            future_days = np.array([[last_day + i] for i in range(1, 31)])
            future_predictions = model.predict(future_days)
            
            # Calculate trend
            current_price = stock_subset['Close'].iloc[-1]
            predicted_price_30d = future_predictions[-1]
            price_change_pct = ((predicted_price_30d - current_price) / current_price) * 100
            
            predictions[symbol] = {
                'current_price': round(current_price, 2),
                'predicted_30d_price': round(predicted_price_30d, 2),
                'predicted_change_pct': round(price_change_pct, 2),
                'trend_direction': 'Bullish' if price_change_pct > 0 else 'Bearish',
                'model_score': round(model.score(X, y), 3)
            }
        
        return predictions
    
    def generate_comprehensive_report(self) -> dict:
        """Generate a comprehensive analytics report"""
        print("Generating comprehensive analytics report...")
        
        self.load_data()
        
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'stock_records': len(self.stock_data) if self.stock_data is not None else 0,
                'legalization_records': len(self.legalization_data) if self.legalization_data is not None else 0,
                'pricing_records': len(self.pricing_data) if self.pricing_data is not None else 0
            }
        }
        
        # Perform all analyses
        try:
            report['stock_metrics'] = self.calculate_stock_metrics()
            print("✓ Stock metrics calculated")
        except Exception as e:
            print(f"✗ Stock metrics failed: {e}")
            report['stock_metrics'] = {}
        
        try:
            report['legalization_analysis'] = self.analyze_legalization_impact()
            print("✓ Legalization analysis completed")
        except Exception as e:
            print(f"✗ Legalization analysis failed: {e}")
            report['legalization_analysis'] = {}
        
        try:
            report['pricing_analysis'] = self.analyze_pricing_trends()
            print("✓ Pricing analysis completed")
        except Exception as e:
            print(f"✗ Pricing analysis failed: {e}")
            report['pricing_analysis'] = {}
        
        try:
            report['market_segmentation'] = self.perform_market_segmentation()
            print("✓ Market segmentation completed")
        except Exception as e:
            print(f"✗ Market segmentation failed: {e}")
            report['market_segmentation'] = {}
        
        try:
            report['price_predictions'] = self.predict_price_trends()
            print("✓ Price predictions completed")
        except Exception as e:
            print(f"✗ Price predictions failed: {e}")
            report['price_predictions'] = {}
        
        # Save report
        import json
        report_filename = f"{self.data_dir}/analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Comprehensive report saved to: {report_filename}")
        return report

if __name__ == "__main__":
    engine = CannabisAnalyticsEngine()
    report = engine.generate_comprehensive_report()
    
    # Print summary
    print("\n" + "="*50)
    print("CANNABIS ANALYTICS SUMMARY")
    print("="*50)
    
    if 'stock_metrics' in report and report['stock_metrics']:
        print("\n📈 TOP PERFORMING STOCKS:")
        sorted_stocks = sorted(report['stock_metrics'].items(), 
                             key=lambda x: x[1]['total_return_pct'], 
                             reverse=True)[:3]
        for symbol, metrics in sorted_stocks:
            print(f"  {symbol}: {metrics['total_return_pct']}% return, ${metrics['current_price']}")
    
    if 'legalization_analysis' in report and report['legalization_analysis']:
        geo = report['legalization_analysis'].get('geographic', {})
        print(f"\n🗺️  LEGALIZATION STATUS:")
        print(f"  US Recreational States: {geo.get('us_recreational_states', 0)}")
        print(f"  US Medical States: {geo.get('us_medical_states', 0)}")
        print(f"  International Countries: {geo.get('international_countries', 0)}")
    
    if 'pricing_analysis' in report and report['pricing_analysis']:
        print(f"\n💰 PRICING TRENDS:")
        trend = report['pricing_analysis'].get('trends', {}).get('overall_price_trend', 'Unknown')
        print(f"  Overall Price Trend: {trend}")
    
    print("\n" + "="*50)