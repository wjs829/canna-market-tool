"""
Configuration settings for Cannabis Analytics Tool
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPORT_DIR = PROJECT_ROOT / "exports"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# Cannabis stocks to track
CANNABIS_STOCKS = [
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
    'MSOS',   # AdvisorShares Pure US Cannabis ETF
    'MJ',     # ETFMG Alternative Harvest ETF
]

# Data collection settings
DATA_COLLECTION = {
    'stock_period': '1y',  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    'api_delay': 0.1,      # Delay between API calls (seconds)
    'retry_attempts': 3,   # Number of retry attempts for failed requests
    'timeout': 30,         # Request timeout (seconds)
}

# Analysis settings
ANALYSIS = {
    'correlation_threshold': 0.3,    # Minimum correlation to report
    'significance_level': 0.05,      # Statistical significance threshold
    'rolling_window_days': 30,       # Rolling window for moving averages
    'volatility_window': 30,         # Window for volatility calculations
}

# Visualization settings
VISUALIZATION = {
    'default_height': 500,
    'default_width': 800,
    'color_scheme': 'viridis',
    'chart_theme': 'plotly_white',
}

# Dashboard settings
DASHBOARD = {
    'title': 'Cannabis Market Analytics',
    'page_icon': '🌿',
    'layout': 'wide',
    'sidebar_expanded': True,
}

# Export settings
EXPORT = {
    'formats': ['csv', 'json', 'excel'],
    'include_charts': True,
    'chart_format': 'png',
    'chart_dpi': 300,
}

# Prediction model settings
MODELS = {
    'train_test_split': 0.8,
    'random_state': 42,
    'cross_validation_folds': 5,
    'max_features': 10,
}

# US States cannabis data
US_STATES_CANNABIS = {
    'legal_recreational': [
        'Alaska', 'Arizona', 'California', 'Colorado', 'Connecticut',
        'Illinois', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
        'Missouri', 'Montana', 'Nevada', 'New Jersey', 'New Mexico',
        'New York', 'Oregon', 'Rhode Island', 'Vermont', 'Virginia',
        'Washington'
    ],
    'legal_medical': [
        'Alabama', 'Arkansas', 'Delaware', 'Florida', 'Hawaii',
        'Louisiana', 'Minnesota', 'Mississippi', 'New Hampshire',
        'North Dakota', 'Ohio', 'Oklahoma', 'Pennsylvania', 'South Dakota',
        'Texas', 'Utah', 'West Virginia'
    ]
}

# International cannabis data
INTERNATIONAL_CANNABIS = {
    'countries_legal_recreational': [
        'Canada', 'Uruguay', 'Luxembourg', 'Malta'
    ],
    'countries_legal_medical': [
        'Germany', 'Australia', 'Israel', 'Netherlands', 'Czech Republic',
        'Poland', 'United Kingdom', 'France', 'Italy', 'Spain'
    ]
}

# API endpoints and data sources
DATA_SOURCES = {
    'stock_data': 'Yahoo Finance',
    'news_data': 'NewsAPI',
    'government_data': 'Various government sources',
    'pricing_data': 'Industry reports and surveys'
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(PROJECT_ROOT / 'logs' / 'cannabis_analytics.log')
}

# Create logs directory
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)