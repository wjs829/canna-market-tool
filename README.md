# Cannabis Data Analytics Tool

A comprehensive data analytics platform for analyzing trends in medical and recreational marijuana/cannabis markets.

## Features

- **Market Analysis**: Track cannabis stock prices, market cap, and trading volumes
- **Geographic Trends**: Analyze legalization patterns across states and countries
- **Medical vs Recreational**: Compare usage patterns and market dynamics
- **Price Analytics**: Monitor cannabis product pricing trends
- **Legislative Tracking**: Track policy changes and their market impact
- **Sentiment Analysis**: Analyze public opinion and social media trends
- **Interactive Dashboards**: Real-time visualization of key metrics

## Components

1. **Data Collection Module** (`data_collector.py`)
   - Stock market data collection
   - News and sentiment data
   - Government and policy data
   - Pricing and market data

2. **Analytics Engine** (`analytics_engine.py`)
   - Trend analysis algorithms
   - Predictive modeling
   - Statistical analysis
   - Machine learning insights

3. **Visualization Dashboard** (`dashboard.py`)
   - Interactive web interface
   - Real-time charts and graphs
   - Customizable analytics views

4. **Data Storage** (`data_manager.py`)
   - Efficient data storage and retrieval
   - Data cleaning and preprocessing
   - Historical data management

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Dashboard
```bash
streamlit run dashboard.py
```

### Run Data Collection
```bash
python data_collector.py
```

### Generate Analytics Report
```bash
python analytics_engine.py
```

## Data Sources

- Financial markets (Yahoo Finance, Alpha Vantage)
- Government databases
- News APIs
- Social media sentiment
- Industry reports
- Legislative databases

## Legal Notice

This tool is for educational and research purposes only. Please ensure compliance with all local laws and regulations regarding cannabis data collection and analysis.