"""
Cannabis Data Collector
Collects data from various sources for cannabis market analysis
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional
import time

class CannabisDataCollector:
    """Collects cannabis-related market and trend data from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.cannabis_stocks = [
            'CGC',    # Canopy Growth Corporation
            'TLRY',   # Tilray Inc
            'CRON',   # Cronos Group Inc
            'ACB',    # Aurora Cannabis Inc
            'HEXO',   # HEXO Corp
            'OGI',    # Organigram Holdings Inc
            'SNDL',   # Sundial Growers Inc
            'CURLF',  # Curaleaf Holdings Inc
            'GTBIF',  # Green Thumb Industries Inc
            'TCNNF',  # Trulieve Cannabis Corp
        ]
        
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def collect_stock_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Collect stock market data for cannabis companies
        
        Args:
            period: Time period for data collection (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with stock data for all cannabis companies
        """
        print("Collecting cannabis stock market data...")
        
        all_stock_data = []
        
        for symbol in self.cannabis_stocks:
            try:
                print(f"Fetching data for {symbol}...")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    hist['Symbol'] = symbol
                    hist['Date'] = hist.index
                    hist = hist.reset_index(drop=True)
                    all_stock_data.append(hist)
                
                # Add delay to respect API limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        if all_stock_data:
            combined_data = pd.concat(all_stock_data, ignore_index=True)
            
            # Save to CSV
            filename = f"{self.data_dir}/cannabis_stocks_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
            combined_data.to_csv(filename, index=False)
            print(f"Stock data saved to {filename}")
            
            return combined_data
        else:
            print("No stock data collected")
            return pd.DataFrame()
    
    def get_company_info(self) -> Dict:
        """Get detailed information about cannabis companies"""
        print("Collecting company information...")
        
        company_info = {}
        
        for symbol in self.cannabis_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                company_info[symbol] = {
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'country': info.get('country', 'N/A'),
                    'website': info.get('website', 'N/A'),
                    'employees': info.get('fullTimeEmployees', 0),
                    'description': info.get('longBusinessSummary', 'N/A')
                }
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting info for {symbol}: {e}")
                continue
        
        # Save company info
        filename = f"{self.data_dir}/company_info_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(company_info, f, indent=2)
        
        print(f"Company info saved to {filename}")
        return company_info
    
    def collect_legalization_data(self) -> pd.DataFrame:
        """
        Create a dataset of cannabis legalization status by state/country
        """
        print("Creating legalization dataset...")
        
        # US State legalization data (as of 2024)
        us_states_data = [
            {'state': 'California', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 1996, 'recreational_year': 2016, 'country': 'USA'},
            {'state': 'Colorado', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2000, 'recreational_year': 2012, 'country': 'USA'},
            {'state': 'Washington', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 1998, 'recreational_year': 2012, 'country': 'USA'},
            {'state': 'Oregon', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 1998, 'recreational_year': 2014, 'country': 'USA'},
            {'state': 'Alaska', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 1998, 'recreational_year': 2014, 'country': 'USA'},
            {'state': 'Nevada', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2000, 'recreational_year': 2016, 'country': 'USA'},
            {'state': 'Maine', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 1999, 'recreational_year': 2016, 'country': 'USA'},
            {'state': 'Massachusetts', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2012, 'recreational_year': 2016, 'country': 'USA'},
            {'state': 'Vermont', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2004, 'recreational_year': 2018, 'country': 'USA'},
            {'state': 'Michigan', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2008, 'recreational_year': 2018, 'country': 'USA'},
            {'state': 'Illinois', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2013, 'recreational_year': 2019, 'country': 'USA'},
            {'state': 'Arizona', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2010, 'recreational_year': 2020, 'country': 'USA'},
            {'state': 'Montana', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2004, 'recreational_year': 2020, 'country': 'USA'},
            {'state': 'New Jersey', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2010, 'recreational_year': 2020, 'country': 'USA'},
            {'state': 'New York', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2014, 'recreational_year': 2021, 'country': 'USA'},
            {'state': 'Virginia', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2018, 'recreational_year': 2021, 'country': 'USA'},
            {'state': 'Connecticut', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2012, 'recreational_year': 2021, 'country': 'USA'},
            {'state': 'New Mexico', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2007, 'recreational_year': 2021, 'country': 'USA'},
            {'state': 'Rhode Island', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2006, 'recreational_year': 2022, 'country': 'USA'},
            {'state': 'Maryland', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2014, 'recreational_year': 2023, 'country': 'USA'},
            {'state': 'Missouri', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2018, 'recreational_year': 2023, 'country': 'USA'},
            # Medical only states
            {'state': 'Florida', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2016, 'recreational_year': None, 'country': 'USA'},
            {'state': 'Texas', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2015, 'recreational_year': None, 'country': 'USA'},
            {'state': 'Pennsylvania', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2016, 'recreational_year': None, 'country': 'USA'},
            {'state': 'Ohio', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2016, 'recreational_year': None, 'country': 'USA'},
        ]
        
        # International data
        international_data = [
            {'state': 'Canada', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2001, 'recreational_year': 2018, 'country': 'Canada'},
            {'state': 'Netherlands', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2003, 'recreational_year': None, 'country': 'Netherlands'},
            {'state': 'Germany', 'medical_legal': True, 'recreational_legal': False, 'medical_year': 2017, 'recreational_year': None, 'country': 'Germany'},
            {'state': 'Uruguay', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2013, 'recreational_year': 2013, 'country': 'Uruguay'},
            {'state': 'Luxembourg', 'medical_legal': True, 'recreational_legal': True, 'medical_year': 2018, 'recreational_year': 2023, 'country': 'Luxembourg'},
        ]
        
        all_data = us_states_data + international_data
        df = pd.DataFrame(all_data)
        
        # Save legalization data
        filename = f"{self.data_dir}/legalization_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Legalization data saved to {filename}")
        
        return df
    
    def generate_sample_pricing_data(self) -> pd.DataFrame:
        """
        Generate sample cannabis product pricing data
        """
        print("Generating sample pricing data...")
        
        import numpy as np
        
        # Product categories
        products = ['Flower', 'Edibles', 'Concentrates', 'Vape Cartridges', 'Topicals', 'Tinctures']
        states = ['California', 'Colorado', 'Washington', 'Oregon', 'Nevada']
        
        # Generate sample data
        data = []
        for _ in range(1000):
            product = np.random.choice(products)
            state = np.random.choice(states)
            
            # Base prices vary by product type
            base_prices = {
                'Flower': 12,
                'Edibles': 8,
                'Concentrates': 35,
                'Vape Cartridges': 45,
                'Topicals': 25,
                'Tinctures': 30
            }
            
            price = base_prices[product] + np.random.normal(0, 5)
            price = max(price, 5)  # Minimum price
            
            data.append({
                'date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'product_type': product,
                'state': state,
                'price_per_gram': round(price, 2),
                'thc_percentage': round(np.random.uniform(15, 30), 1),
                'cbd_percentage': round(np.random.uniform(0, 15), 1),
                'dispensary_type': np.random.choice(['Medical', 'Recreational', 'Both'])
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date')
        
        # Save pricing data
        filename = f"{self.data_dir}/cannabis_pricing_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Pricing data saved to {filename}")
        
        return df
    
    def collect_all_data(self):
        """Collect all available data"""
        print("Starting comprehensive data collection...")
        
        try:
            # Collect stock data
            stock_data = self.collect_stock_data()
            
            # Get company information
            company_info = self.get_company_info()
            
            # Collect legalization data
            legalization_data = self.collect_legalization_data()
            
            # Generate pricing data
            pricing_data = self.generate_sample_pricing_data()
            
            print("\nData collection completed successfully!")
            print(f"- Stock data: {len(stock_data) if not stock_data.empty else 0} records")
            print(f"- Company info: {len(company_info)} companies")
            print(f"- Legalization data: {len(legalization_data)} jurisdictions")
            print(f"- Pricing data: {len(pricing_data)} records")
            
            return {
                'stocks': stock_data,
                'companies': company_info,
                'legalization': legalization_data,
                'pricing': pricing_data
            }
            
        except Exception as e:
            print(f"Error during data collection: {e}")
            return None

if __name__ == "__main__":
    collector = CannabisDataCollector()
    collector.collect_all_data()