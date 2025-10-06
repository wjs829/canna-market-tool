"""
Cannabis Analytics Dashboard
Interactive Streamlit dashboard for cannabis market data visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_collector import CannabisDataCollector
    from src.analytics_engine import CannabisAnalyticsEngine
except ImportError:
    # Fallback if running from different directory
    from data_collector import CannabisDataCollector
    from analytics_engine import CannabisAnalyticsEngine

# Page configuration
st.set_page_config(
    page_title="Cannabis Market Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class CannabinDashboard:
    def __init__(self):
        self.data_collector = CannabisDataCollector()
        self.analytics_engine = CannabisAnalyticsEngine()
        
    def load_data(self):
        """Load data with caching"""
        if 'stock_data' not in st.session_state:
            with st.spinner("Loading cannabis market data..."):
                self.analytics_engine.load_data()
                st.session_state.stock_data = self.analytics_engine.stock_data
                st.session_state.legalization_data = self.analytics_engine.legalization_data
                st.session_state.pricing_data = self.analytics_engine.pricing_data
        
        return (st.session_state.stock_data, 
                st.session_state.legalization_data, 
                st.session_state.pricing_data)
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.markdown('<div class="sidebar-header">🌿 Cannabis Analytics</div>', unsafe_allow_html=True)
        
        # Data collection controls
        st.sidebar.subheader("Data Management")
        
        if st.sidebar.button("🔄 Refresh Data", help="Collect fresh market data"):
            with st.spinner("Collecting fresh data..."):
                self.data_collector.collect_all_data()
                # Clear cached data
                for key in ['stock_data', 'legalization_data', 'pricing_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Data refreshed successfully!")
                st.rerun()
        
        # Analysis controls
        st.sidebar.subheader("Analysis Options")
        
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Overview", "Stock Analysis", "Legalization Trends", "Pricing Analysis", "Market Predictions"]
        )
        
        # Date range selector
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.selectbox(
            "Select Time Period",
            ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Last Year", "All Time"]
        )
        
        return analysis_type, date_range
    
    def render_overview(self, stock_data, legalization_data, pricing_data):
        """Render the overview dashboard"""
        st.markdown('<div class="main-header">🌿 Cannabis Market Analytics Dashboard</div>', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏢 Cannabis Stocks Tracked",
                value=len(stock_data['Symbol'].unique()) if stock_data is not None else 0,
                delta=None
            )
        
        with col2:
            rec_legal = len(legalization_data[legalization_data['recreational_legal'] == True]) if legalization_data is not None else 0
            st.metric(
                label="🗺️ Recreational Legal Jurisdictions",
                value=rec_legal,
                delta=None
            )
        
        with col3:
            med_legal = len(legalization_data[legalization_data['medical_legal'] == True]) if legalization_data is not None else 0
            st.metric(
                label="🏥 Medical Legal Jurisdictions", 
                value=med_legal,
                delta=None
            )
        
        with col4:
            avg_price = pricing_data['price_per_gram'].mean() if pricing_data is not None else 0
            st.metric(
                label="💰 Avg Price per Gram",
                value=f"${avg_price:.2f}",
                delta=None
            )
        
        # Charts section
        if stock_data is not None and not stock_data.empty:
            st.subheader("📈 Stock Market Performance")
            
            # Stock performance chart
            fig = go.Figure()
            
            for symbol in stock_data['Symbol'].unique()[:5]:  # Top 5 stocks
                stock_subset = stock_data[stock_data['Symbol'] == symbol].sort_values('Date')
                fig.add_trace(go.Scatter(
                    x=stock_subset['Date'],
                    y=stock_subset['Close'],
                    mode='lines',
                    name=symbol,
                    hovertemplate=f'<b>{symbol}</b><br>Price: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Cannabis Stock Prices Over Time",
                xaxis_title="Date",
                yaxis_title="Price ($USD)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Legalization timeline
        if legalization_data is not None:
            st.subheader("🗺️ Cannabis Legalization Timeline")
            
            # Create timeline chart
            legal_timeline = legalization_data[legalization_data['recreational_legal'] == True].copy()
            legal_timeline = legal_timeline[legal_timeline['recreational_year'].notna()]
            
            if not legal_timeline.empty:
                timeline_counts = legal_timeline['recreational_year'].value_counts().sort_index()
                
                fig = px.bar(
                    x=timeline_counts.index,
                    y=timeline_counts.values,
                    title="Recreational Cannabis Legalization by Year",
                    labels={'x': 'Year', 'y': 'Number of Jurisdictions'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_stock_analysis(self, stock_data):
        """Render detailed stock analysis"""
        st.header("📈 Cannabis Stock Analysis")
        
        if stock_data is None or stock_data.empty:
            st.warning("No stock data available. Please refresh data first.")
            return
        
        # Stock selector
        selected_stocks = st.multiselect(
            "Select stocks to analyze:",
            options=sorted(stock_data['Symbol'].unique()),
            default=sorted(stock_data['Symbol'].unique())[:3]
        )
        
        if not selected_stocks:
            st.warning("Please select at least one stock.")
            return
        
        # Calculate metrics
        metrics = self.analytics_engine.calculate_stock_metrics()
        
        # Display metrics table
        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            metrics_df = metrics_df[metrics_df.index.isin(selected_stocks)]
            
            st.subheader("📊 Stock Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)
        
        # Price comparison chart
        st.subheader("💹 Price Comparison")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stock Prices', 'Trading Volume'),
            vertical_spacing=0.1
        )
        
        for symbol in selected_stocks:
            stock_subset = stock_data[stock_data['Symbol'] == symbol].sort_values('Date')
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=stock_subset['Date'],
                    y=stock_subset['Close'],
                    mode='lines',
                    name=f'{symbol} Price',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
            
            # Volume chart
            fig.add_trace(
                go.Scatter(
                    x=stock_subset['Date'],
                    y=stock_subset['Volume'],
                    mode='lines',
                    name=f'{symbol} Volume',
                    line=dict(width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text="Cannabis Stock Analysis")
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market segmentation
        st.subheader("🎯 Market Segmentation")
        segmentation = self.analytics_engine.perform_market_segmentation()
        
        if segmentation and 'error' not in segmentation:
            for cluster_name, cluster_data in segmentation.items():
                with st.expander(f"📈 {cluster_name}"):
                    st.write(f"**Stocks:** {', '.join(cluster_data['stocks'])}")
                    
                    chars = cluster_data['characteristics']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average Return", f"{chars['avg_return']:.2%}")
                        st.metric("Volatility", f"{chars['volatility']:.2%}")
                    
                    with col2:
                        st.metric("Relative Volume", f"{chars['relative_volume']:.2f}x")
                        st.metric("Average Price", f"${chars['avg_price']:.2f}")
    
    def render_legalization_analysis(self, legalization_data):
        """Render legalization trends analysis"""
        st.header("🗺️ Cannabis Legalization Analysis")
        
        if legalization_data is None:
            st.warning("No legalization data available.")
            return
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_medical = len(legalization_data[legalization_data['medical_legal'] == True])
            st.metric("Medical Legal", total_medical)
        
        with col2:
            total_recreational = len(legalization_data[legalization_data['recreational_legal'] == True])
            st.metric("Recreational Legal", total_recreational)
        
        with col3:
            total_jurisdictions = len(legalization_data)
            st.metric("Total Jurisdictions", total_jurisdictions)
        
        # Geographic breakdown
        st.subheader("🌎 Geographic Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # US States breakdown
            us_data = legalization_data[legalization_data['country'] == 'USA']
            
            us_medical = len(us_data[us_data['medical_legal'] == True])
            us_recreational = len(us_data[us_data['recreational_legal'] == True])
            us_total = len(us_data)
            
            fig = go.Figure(data=[
                go.Bar(name='Medical Only', x=['USA'], y=[us_medical - us_recreational]),
                go.Bar(name='Recreational', x=['USA'], y=[us_recreational]),
                go.Bar(name='Neither', x=['USA'], y=[us_total - us_medical])
            ])
            
            fig.update_layout(
                barmode='stack',
                title="US Cannabis Legalization Status",
                yaxis_title="Number of States"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # International breakdown
            intl_data = legalization_data[legalization_data['country'] != 'USA']
            
            if not intl_data.empty:
                intl_breakdown = intl_data.groupby('country').agg({
                    'medical_legal': 'sum',
                    'recreational_legal': 'sum'
                }).reset_index()
                
                fig = px.bar(
                    intl_breakdown,
                    x='country',
                    y=['medical_legal', 'recreational_legal'],
                    title="International Cannabis Legalization",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Timeline analysis
        st.subheader("📅 Legalization Timeline")
        
        # Recreational legalization timeline
        rec_data = legalization_data[
            (legalization_data['recreational_legal'] == True) & 
            (legalization_data['recreational_year'].notna())
        ]
        
        if not rec_data.empty:
            timeline = rec_data['recreational_year'].value_counts().sort_index()
            
            fig = px.line(
                x=timeline.index,
                y=timeline.values,
                title="Recreational Cannabis Legalization Over Time",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Jurisdictions",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Detailed Legalization Data")
        
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            country_filter = st.selectbox("Filter by Country", ["All"] + list(legalization_data['country'].unique()))
        
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All", "Medical Only", "Recreational Legal", "Neither"])
        
        # Apply filters
        filtered_data = legalization_data.copy()
        
        if country_filter != "All":
            filtered_data = filtered_data[filtered_data['country'] == country_filter]
        
        if status_filter == "Medical Only":
            filtered_data = filtered_data[
                (filtered_data['medical_legal'] == True) & 
                (filtered_data['recreational_legal'] == False)
            ]
        elif status_filter == "Recreational Legal":
            filtered_data = filtered_data[filtered_data['recreational_legal'] == True]
        elif status_filter == "Neither":
            filtered_data = filtered_data[
                (filtered_data['medical_legal'] == False) & 
                (filtered_data['recreational_legal'] == False)
            ]
        
        st.dataframe(filtered_data, use_container_width=True)
    
    def render_pricing_analysis(self, pricing_data):
        """Render pricing trends analysis"""
        st.header("💰 Cannabis Pricing Analysis")
        
        if pricing_data is None:
            st.warning("No pricing data available.")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = pricing_data['price_per_gram'].mean()
            st.metric("Average Price/Gram", f"${avg_price:.2f}")
        
        with col2:
            max_price = pricing_data['price_per_gram'].max()
            st.metric("Highest Price", f"${max_price:.2f}")
        
        with col3:
            min_price = pricing_data['price_per_gram'].min()
            st.metric("Lowest Price", f"${min_price:.2f}")
        
        with col4:
            price_std = pricing_data['price_per_gram'].std()
            st.metric("Price Volatility", f"±${price_std:.2f}")
        
        # Price by product type
        st.subheader("📊 Price by Product Type")
        
        product_prices = pricing_data.groupby('product_type')['price_per_gram'].agg(['mean', 'std']).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=product_prices.index,
            y=product_prices['mean'],
            error_y=dict(type='data', array=product_prices['std']),
            name='Average Price',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title="Average Cannabis Prices by Product Type",
            xaxis_title="Product Type",
            yaxis_title="Price per Gram ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price by state
        st.subheader("🗺️ Price by State")
        
        state_prices = pricing_data.groupby('state')['price_per_gram'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=state_prices.index,
            y=state_prices.values,
            title="Average Cannabis Prices by State",
            color=state_prices.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Price per Gram ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Medical vs Recreational pricing
        st.subheader("🏥 Medical vs Recreational Pricing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dispensary_prices = pricing_data.groupby('dispensary_type')['price_per_gram'].mean()
            
            fig = px.pie(
                values=dispensary_prices.values,
                names=dispensary_prices.index,
                title="Average Prices by Dispensary Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price distribution
            fig = px.box(
                pricing_data,
                x='dispensary_type',
                y='price_per_gram',
                title="Price Distribution by Dispensary Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Potency analysis
        st.subheader("🧪 Potency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # THC vs Price scatter
            fig = px.scatter(
                pricing_data,
                x='thc_percentage',
                y='price_per_gram',
                color='product_type',
                title="THC Percentage vs Price",
                hover_data=['cbd_percentage', 'state']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CBD vs Price scatter
            fig = px.scatter(
                pricing_data,
                x='cbd_percentage',
                y='price_per_gram',
                color='product_type',
                title="CBD Percentage vs Price",
                hover_data=['thc_percentage', 'state']
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_predictions(self, stock_data):
        """Render market predictions"""
        st.header("🔮 Market Predictions")
        
        if stock_data is None or stock_data.empty:
            st.warning("No stock data available for predictions.")
            return
        
        # Generate predictions
        with st.spinner("Generating predictions..."):
            predictions = self.analytics_engine.predict_price_trends()
        
        if not predictions:
            st.warning("Unable to generate predictions with current data.")
            return
        
        st.subheader("📈 30-Day Price Predictions")
        
        # Create predictions table
        pred_df = pd.DataFrame(predictions).T
        pred_df = pred_df.sort_values('predicted_change_pct', ascending=False)
        
        # Color code the predictions
        def color_predictions(val):
            if val > 0:
                return 'background-color: #d4edda; color: #155724'  # Green
            else:
                return 'background-color: #f8d7da; color: #721c24'  # Red
        
        styled_df = pred_df.style.applymap(color_predictions, subset=['predicted_change_pct'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Prediction visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bullish vs Bearish count
            bullish_count = sum(1 for p in predictions.values() if p['trend_direction'] == 'Bullish')
            bearish_count = len(predictions) - bullish_count
            
            fig = px.pie(
                values=[bullish_count, bearish_count],
                names=['Bullish', 'Bearish'],
                title="Market Sentiment Distribution",
                color_discrete_map={'Bullish': '#28a745', 'Bearish': '#dc3545'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Predicted changes bar chart
            symbols = list(predictions.keys())
            changes = [predictions[s]['predicted_change_pct'] for s in symbols]
            colors = ['green' if c > 0 else 'red' for c in changes]
            
            fig = go.Figure(data=[
                go.Bar(x=symbols, y=changes, marker_color=colors)
            ])
            
            fig.update_layout(
                title="Predicted 30-Day Price Changes",
                xaxis_title="Stock Symbol",
                yaxis_title="Predicted Change (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model accuracy information
        st.subheader("🎯 Model Accuracy")
        
        accuracy_data = [(symbol, data['model_score']) for symbol, data in predictions.items()]
        accuracy_df = pd.DataFrame(accuracy_data, columns=['Symbol', 'Model Score'])
        
        avg_accuracy = accuracy_df['Model Score'].mean()
        
        st.info(f"Average Model R² Score: {avg_accuracy:.3f}")
        st.caption("R² Score indicates how well the linear regression model fits the historical data. Higher scores (closer to 1.0) indicate better fit.")
        
        # Display accuracy chart
        fig = px.bar(
            accuracy_df,
            x='Symbol',
            y='Model Score',
            title="Model Accuracy by Stock",
            color='Model Score',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        # Load data
        stock_data, legalization_data, pricing_data = self.load_data()
        
        # Render sidebar
        analysis_type, date_range = self.render_sidebar()
        
        # Render main content based on selection
        if analysis_type == "Overview":
            self.render_overview(stock_data, legalization_data, pricing_data)
        
        elif analysis_type == "Stock Analysis":
            self.render_stock_analysis(stock_data)
        
        elif analysis_type == "Legalization Trends":
            self.render_legalization_analysis(legalization_data)
        
        elif analysis_type == "Pricing Analysis":
            self.render_pricing_analysis(pricing_data)
        
        elif analysis_type == "Market Predictions":
            self.render_predictions(stock_data)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🌿 **Cannabis Market Analytics Dashboard** | "
            "Built with Streamlit, Plotly, and Python | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

# Run the dashboard
if __name__ == "__main__":
    dashboard = CannabinDashboard()
    dashboard.run()