# Cannabis Data Analytics Tool - User Guide

## 🌿 Welcome to Cannabis Market Analytics

This comprehensive tool provides advanced data analytics for analyzing trends in medical and recreational cannabis markets. The tool includes data collection, analytics processing, interactive dashboards, and machine learning predictions.

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
python run.py
```

This will show you a menu with all available options:
1. Run Data Collection
2. Run Analytics Engine  
3. Launch Interactive Dashboard
4. Open Jupyter Notebook
5. Run All (Collect Data + Analytics)
6. Exit

### Option 2: Individual Components

#### Data Collection
```bash
python src/data_collector.py
```

#### Analytics Engine
```bash
python src/analytics_engine.py
```

#### Interactive Dashboard
```bash
streamlit run dashboard.py
```

#### Jupyter Notebook
```bash
jupyter lab notebooks/cannabis_analytics.ipynb
```

## 📊 Features

### Data Collection (`src/data_collector.py`)
- **Stock Market Data**: Real-time cannabis company stock prices from Yahoo Finance
- **Legalization Data**: Comprehensive database of cannabis legalization status by jurisdiction
- **Pricing Data**: Cannabis product pricing across different markets and product types
- **Company Information**: Detailed information about major cannabis companies

### Analytics Engine (`src/analytics_engine.py`)
- **Market Metrics**: Calculate key performance indicators and market statistics
- **Trend Analysis**: Identify patterns in pricing, sales, and market adoption
- **Correlation Analysis**: Find relationships between different market factors
- **Geographic Analysis**: Compare markets across different states and countries
- **Segmentation**: Cluster analysis of companies and markets

### Interactive Dashboard (`dashboard.py`)
- **Real-time Visualizations**: Dynamic charts and graphs
- **Market Explorer**: Interactive bubble charts and geographic maps
- **Performance Tracking**: Monitor stock performance and market trends
- **Filtering and Analysis**: Customizable views and data exploration tools

### Jupyter Notebook (`notebooks/cannabis_analytics.ipynb`)
- **Comprehensive Analysis**: Step-by-step data analysis workflow
- **Machine Learning Models**: Predictive models for price forecasting
- **Visualization Gallery**: Collection of charts and statistical analyses
- **Export Capabilities**: Generate reports and export data

## 📈 Analytics Capabilities

### Market Analysis
- Stock price performance and volatility analysis
- Market capitalization trends
- Trading volume patterns
- Risk-adjusted returns (Sharpe ratios)

### Geographic Trends
- State-by-state legalization tracking
- International market comparison
- Price variation by geography
- Market maturity analysis

### Product Analytics
- Pricing trends by product type (flower, edibles, concentrates, etc.)
- Potency analysis (THC/CBD content)
- Medical vs recreational market comparison
- Seasonal pricing patterns

### Predictive Modeling
- Stock price prediction using machine learning
- Market growth forecasting
- Price trend prediction
- Legalization timeline modeling

## 🗂️ Data Sources

- **Yahoo Finance**: Stock market data
- **Government Sources**: Legalization status and regulatory information
- **Industry Reports**: Pricing and market trend data
- **Public APIs**: Real-time market information

## 📁 Project Structure

```
cannabis-analytics/
├── src/                          # Source code
│   ├── data_collector.py        # Data collection module
│   ├── analytics_engine.py      # Analytics processing
│   └── __init__.py
├── notebooks/                    # Jupyter notebooks
│   └── cannabis_analytics.ipynb # Comprehensive analysis notebook
├── data/                         # Data storage
│   ├── cannabis_stocks_*.csv   # Stock market data
│   ├── legalization_data_*.csv # Legalization information
│   ├── cannabis_pricing_*.csv  # Pricing data
│   └── company_info_*.json     # Company details
├── exports/                      # Generated reports and exports
├── dashboard.py                  # Streamlit dashboard
├── run.py                       # Main launcher script
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection for data collection

### Installation
1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tool:
   ```bash
   python run.py
   ```

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Static visualizations
- **plotly**: Interactive charts
- **streamlit**: Web dashboard framework
- **scikit-learn**: Machine learning models
- **yfinance**: Stock market data
- **jupyter**: Notebook environment

## 📊 Sample Analysis Results

The tool provides insights such as:

- **Stock Performance**: "SNDL showed 29% return over the past year with high volatility"
- **Market Trends**: "21 US states have legalized recreational cannabis"
- **Pricing Analysis**: "Average cannabis price is $23.45/gram with concentrates commanding premium prices"
- **Geographic Insights**: "California shows the most mature market with lowest price volatility"
- **Predictions**: "Linear models suggest continued price stabilization in mature markets"

## 🔧 Customization

### Adding New Data Sources
Edit `src/data_collector.py` to add new APIs or data sources.

### Custom Analytics
Extend `src/analytics_engine.py` with your own analysis functions.

### Dashboard Customization
Modify `dashboard.py` to add new visualizations or change the layout.

### Configuration
Update `config.py` to change default settings, stock lists, or analysis parameters.

## 📱 Dashboard Features

The Streamlit dashboard includes:

### Overview Page
- Key market metrics
- Stock performance summary
- Legalization status overview
- Price trend indicators

### Stock Analysis
- Individual stock performance
- Correlation analysis
- Risk metrics
- Portfolio analysis

### Geographic Analysis
- State-by-state comparison
- International markets
- Legalization timeline
- Price geography

### Pricing Analysis
- Product type comparison
- Medical vs recreational pricing
- Seasonal trends
- Potency analysis

### Predictions
- 30-day price forecasts
- Market growth projections
- Risk assessments
- Model accuracy metrics

## 🚨 Important Notes

### Legal Compliance
- This tool is for educational and research purposes only
- Ensure compliance with local laws regarding cannabis data and analysis
- The tool does not constitute investment advice

### Data Accuracy
- Stock data is sourced from Yahoo Finance (subject to their terms)
- Pricing data includes synthetic samples for demonstration
- Legalization data is updated manually and may not reflect recent changes

### Performance
- Large datasets may take time to process
- Some visualizations require significant memory
- Dashboard performance depends on data size

## 🔄 Updates and Maintenance

### Data Refresh
Run data collection regularly to get the latest market information:
```bash
python src/data_collector.py
```

### Analysis Updates
Regenerate analytics after collecting new data:
```bash
python src/analytics_engine.py
```

### Dashboard Refresh
Restart the dashboard to pick up new data:
```bash
streamlit run dashboard.py
```

## 🆘 Troubleshooting

### Common Issues

**"No module named 'src'"**
- Run commands from the project root directory
- Ensure Python path includes the src directory

**"Yahoo Finance API errors"**
- Check internet connection
- Some stocks may be delisted
- API rate limits may cause delays

**"Dashboard won't load"**
- Ensure all dependencies are installed
- Check if port 8501 is available
- Try running with different port: `streamlit run dashboard.py --server.port 8502`

**"Jupyter notebook won't open"**
- Install Jupyter: `pip install jupyter`
- Try Jupyter Lab: `pip install jupyterlab`

### Getting Help

1. Check the error messages in the terminal
2. Ensure all dependencies are properly installed
3. Verify you're running commands from the correct directory
4. Check that your Python environment has all required packages

## 📧 Support

For issues, suggestions, or contributions:
- Review the code in the `src/` directory
- Check configuration settings in `config.py`
- Examine sample data in the `data/` directory
- Use the Jupyter notebook for detailed analysis examples

---

**Happy Analyzing! 🌿📊**