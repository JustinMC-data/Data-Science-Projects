"""
Smart Pricing Engine - Streamlit Application
============================================

A multi-page Streamlit app for demand forecasting and price optimization.
Uses trained ML models to predict demand and optimize pricing.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import joblib

# Get paths
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent

# Add src to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Smart Pricing Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_feature_matrix():
    """Load the processed feature matrix."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'feature_matrix.csv'
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None


@st.cache_data
def load_model_comparison():
    """Load model comparison results."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'model_comparison.csv'
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        return df
    return None


@st.cache_data
def load_elasticity_analysis():
    """Load elasticity analysis results."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'elasticity_analysis.csv'
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None


@st.cache_data
def load_pricing_recommendations():
    """Load pricing recommendations."""
    path = PROJECT_ROOT / 'data' / 'processed' / 'pricing_recommendations.csv'
    if path.exists():
        df = pd.read_csv(path)
        return df
    return None


@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    path = PROJECT_ROOT / 'models' / 'xgboost_demand_model.joblib'
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_feature_columns():
    """Load the feature column names."""
    path = PROJECT_ROOT / 'models' / 'feature_columns.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Explorer", "🔬 Model Performance", "💰 Price Optimizer", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    # Show data status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Status")
    
    feature_matrix = load_feature_matrix()
    model = load_model()
    
    if feature_matrix is not None:
        st.sidebar.success(f"✅ Data loaded ({len(feature_matrix):,} records)")
    else:
        st.sidebar.warning("⚠️ No data found")
    
    if model is not None:
        st.sidebar.success("✅ Model loaded")
    else:
        st.sidebar.warning("⚠️ No model found")
    
    # GitHub link in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/JustinMC-data/smart-pricing-engine)")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🔬 Model Performance":
        show_model_performance()
    elif page == "💰 Price Optimizer":
        show_price_optimizer()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display the home/overview page."""
    
    st.markdown('<p class="main-header">🎯 Smart Pricing Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-Powered Demand Forecasting & Price Optimization</p>', unsafe_allow_html=True)
    
    # Load real metrics
    model_comparison = load_model_comparison()
    feature_matrix = load_feature_matrix()
    elasticity_analysis = load_elasticity_analysis()
    
    # Project overview
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Project Overview")
        st.markdown("""
        This application demonstrates an end-to-end machine learning workflow for 
        **price elasticity analysis** and **demand forecasting**. 
        
        The Smart Pricing Engine helps businesses:
        - 📈 Understand how price changes affect demand
        - 🎯 Identify optimal pricing strategies
        - 💡 Make data-driven pricing decisions
        
        By analyzing over one million past sales transactions, the system learns patterns—like 
        which products sell more when discounted, which ones customers will buy regardless of 
        price, and how seasons affect shopping behavior. The end result is a simple interface 
        where a store owner can select any product, adjust the price, and instantly see the 
        predicted impact on sales and revenue.
        """)
        
        st.markdown("### How It Works")
        st.markdown("""
        1. **Data Processing**: Transaction data is cleaned and aggregated to weekly product-level observations
        2. **Feature Engineering**: Temporal, price context, and lag features capture demand drivers
        3. **Model Training**: Multiple ML models (Linear Regression, Random Forest, XGBoost) predict demand
        4. **Elasticity Analysis**: Price sensitivity is calculated for each product
        5. **Optimization**: Revenue-maximizing prices are identified
        """)
    
    with col2:
        st.markdown("### Key Metrics")
        
        # Use real metrics from loaded data
        if feature_matrix is not None:
            n_products = feature_matrix['StockCode'].nunique()
            st.metric(label="Products Analyzed", value=f"{n_products:,}")
        else:
            st.metric(label="Products Analyzed", value="N/A")
        
        if model_comparison is not None:
            best_r2 = model_comparison['R2'].max()
            best_model = model_comparison['R2'].idxmax()
            best_mae = model_comparison.loc[best_model, 'MAE']
            st.metric(label=f"Best Model R²", value=f"{best_r2:.3f}")
            st.metric(label="Avg. Prediction Error (MAE)", value=f"±{best_mae:.1f} units")
        else:
            st.metric(label="Best Model R²", value="N/A")
            st.metric(label="Avg. Prediction Error", value="N/A")
        
        if elasticity_analysis is not None:
            avg_uplift = elasticity_analysis['RevenueUplift'].mean()
            st.metric(label="Avg. Revenue Uplift Potential", value=f"+{avg_uplift:.1f}%")
        else:
            st.metric(label="Revenue Uplift Potential", value="N/A")
    
    st.markdown("---")
    
    # Workflow visualization
    st.markdown("### ML Pipeline")
    
    cols = st.columns(5)
    
    steps = [
        ("📥", "Data\nIngestion"),
        ("🧹", "Cleaning &\nAggregation"),
        ("⚙️", "Feature\nEngineering"),
        ("🤖", "Model\nTraining"),
        ("📊", "Elasticity\nAnalysis")
    ]
    
    for col, (icon, label) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
                <span style="font-size: 2rem;">{icon}</span>
                <p style="margin-top: 0.5rem; font-weight: 500;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats if data available
    if feature_matrix is not None:
        st.markdown("### Dataset Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(feature_matrix):,}")
        with col2:
            st.metric("Unique Products", f"{feature_matrix['StockCode'].nunique():,}")
        with col3:
            st.metric("Time Periods", f"{feature_matrix['YearWeek'].nunique()} weeks")
        with col4:
            total_revenue = feature_matrix['TotalRevenue'].sum()
            st.metric("Total Revenue", f"£{total_revenue:,.0f}")
    
    st.info("👈 Use the sidebar to navigate to different sections of the application.")


def show_data_explorer():
    """Display the data exploration page."""
    
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the Online Retail II dataset and understand the underlying patterns.")
    
    # Load data
    data = load_feature_matrix()
    
    if data is None:
        st.error("❌ Feature matrix not found. Please run the complete_workflow.py script first.")
        st.code("python notebooks/complete_workflow.py", language="bash")
        return
    
    # Data overview
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Unique Products", f"{data['StockCode'].nunique():,}")
    with col3:
        st.metric("Time Periods", f"{data['YearWeek'].nunique()} weeks")
    with col4:
        st.metric("Total Revenue", f"£{data['TotalRevenue'].sum():,.0f}")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Time Series", "Price-Quantity", "Top Products"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            price_data = data['AvgPrice'].clip(upper=data['AvgPrice'].quantile(0.99))
            fig = px.histogram(price_data, nbins=50, 
                              title='Distribution of Product Prices (99th percentile)')
            fig.update_layout(xaxis_title='Price (£)', yaxis_title='Count', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Quantity Distribution")
            qty_data = data['TotalQuantity'].clip(upper=data['TotalQuantity'].quantile(0.99))
            fig = px.histogram(qty_data, nbins=50,
                              title='Distribution of Weekly Quantities (99th percentile)')
            fig.update_layout(xaxis_title='Quantity', yaxis_title='Count', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Weekly Revenue Trend")
        
        weekly_revenue = data.groupby('YearWeek')['TotalRevenue'].sum().reset_index()
        weekly_revenue = weekly_revenue.sort_values('YearWeek')
        
        fig = px.line(weekly_revenue, x='YearWeek', y='TotalRevenue',
                     title='Total Revenue Over Time')
        fig.update_layout(xaxis_title='Week', yaxis_title='Revenue (£)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly breakdown
        st.markdown("#### Monthly Quantity by Product Category")
        monthly_qty = data.groupby('Month')['TotalQuantity'].sum().reset_index()
        fig = px.bar(monthly_qty, x='Month', y='TotalQuantity',
                    title='Total Quantity Sold by Month')
        fig.update_layout(xaxis_title='Month', yaxis_title='Total Quantity')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Price vs. Quantity Relationship")
        
        # Sample for performance
        sample_size = min(5000, len(data))
        sample = data.sample(sample_size)
        
        fig = px.scatter(sample, 
                        x='AvgPrice', y='TotalQuantity',
                        opacity=0.5,
                        title=f'Price-Quantity Relationship (Sample of {sample_size:,} observations)')
        fig.update_layout(xaxis_title='Price (£)', yaxis_title='Quantity')
        
        # Limit axes to remove extreme outliers from view
        fig.update_xaxes(range=[0, sample['AvgPrice'].quantile(0.95)])
        fig.update_yaxes(range=[0, sample['TotalQuantity'].quantile(0.95)])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Top Products by Revenue")
        
        top_products = data.groupby(['StockCode', 'Description']).agg({
            'TotalRevenue': 'sum',
            'TotalQuantity': 'sum',
            'AvgPrice': 'mean'
        }).reset_index()
        
        top_products = top_products.nlargest(20, 'TotalRevenue')
        
        fig = px.bar(top_products, x='StockCode', y='TotalRevenue',
                    hover_data=['Description', 'TotalQuantity', 'AvgPrice'],
                    title='Top 20 Products by Total Revenue')
        fig.update_layout(xaxis_title='Product Code', yaxis_title='Total Revenue (£)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Table view
        st.markdown("#### Product Details")
        display_df = top_products[['StockCode', 'Description', 'TotalRevenue', 'TotalQuantity', 'AvgPrice']].copy()
        display_df.columns = ['Code', 'Description', 'Revenue (£)', 'Quantity', 'Avg Price (£)']
        display_df['Revenue (£)'] = display_df['Revenue (£)'].apply(lambda x: f"£{x:,.2f}")
        display_df['Quantity'] = display_df['Quantity'].apply(lambda x: f"{x:,.0f}")
        display_df['Avg Price (£)'] = display_df['Avg Price (£)'].apply(lambda x: f"£{x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Data preview
    st.markdown("---")
    with st.expander("🔍 View Raw Data Sample"):
        st.dataframe(data.head(100), use_container_width=True)


def show_model_performance():
    """Display the model performance comparison page."""
    
    st.markdown("## 🔬 Model Performance")
    st.markdown("Compare model performance and understand feature importance.")
    
    # Load data
    model_comparison = load_model_comparison()
    feature_columns = load_feature_columns()
    model = load_model()
    
    if model_comparison is None:
        st.error("❌ Model comparison data not found. Please run the complete_workflow.py script first.")
        return
    
    st.markdown("### Model Comparison")
    
    # Format the comparison table
    display_df = model_comparison.copy()
    display_df = display_df.reset_index()
    display_df.columns = ['Model', 'RMSE', 'MAE', 'R²', 'MAPE (%)']
    
    # Highlight best values
    st.dataframe(
        display_df.style.highlight_max(subset=['R²'], color='lightgreen')
                        .highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='lightgreen')
                        .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'R²': '{:.4f}', 'MAPE (%)': '{:.2f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Best model callout
    best_model = model_comparison['R2'].idxmax()
    best_r2 = model_comparison['R2'].max()
    st.success(f"🏆 **Best Model**: {best_model} with R² = {best_r2:.4f}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### R² Score Comparison")
        fig = px.bar(display_df, x='Model', y='R²',
                    color='R²', color_continuous_scale='Blues',
                    title='Model R² Scores (Higher is Better)')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Error Metrics Comparison")
        
        # Melt for grouped bar chart
        error_df = display_df[['Model', 'RMSE', 'MAE']].melt(
            id_vars='Model', var_name='Metric', value_name='Value'
        )
        
        fig = px.bar(error_df, x='Model', y='Value', color='Metric',
                    barmode='group', title='Error Metrics (Lower is Better)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance
    st.markdown("### Feature Importance (XGBoost)")
    
    if model is not None and feature_columns is not None:
        # Get feature importance from model
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Top 15 features
        top_features = importance_df.tail(15)
        
        fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues',
                    title='Top 15 Most Important Features')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        top_3 = importance_df.nlargest(3, 'Importance')['Feature'].tolist()
        st.info(f"""
        **Key Insights:**
        - The top 3 most important features are: **{', '.join(top_3)}**
        - Historical quantity patterns (lag and rolling features) are strong predictors
        - Price-related features show meaningful contribution to demand prediction
        """)
    else:
        st.warning("Model or feature columns not loaded. Cannot display feature importance.")
    
    # Model interpretation
    st.markdown("---")
    st.markdown("### 📖 Model Interpretation")
    
    st.markdown("""
    **Understanding the Results:**
    
    - **R² (Coefficient of Determination)**: Measures how much variance in demand our model explains. 
      An R² of 0.35 means our model explains about 35% of the variation in weekly product demand.
    
    - **RMSE (Root Mean Square Error)**: Average prediction error in units. Lower is better.
    
    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual demand. 
      More robust to outliers than RMSE.
    
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage error. High values here are often 
      due to products with very low quantities where small errors produce large percentages.
    
    **Why Linear Regression performed well:**
    The Online Retail II dataset may have relatively linear price-demand relationships, 
    and the strong lag features provide a solid foundation for prediction.
    """)


def show_price_optimizer():
    """Display the interactive price optimizer page."""
    
    st.markdown("## 💰 Price Optimizer")
    st.markdown("Select a product and simulate demand at different price points to find the optimal price.")
    
    # Load data
    feature_matrix = load_feature_matrix()
    elasticity_analysis = load_elasticity_analysis()
    model = load_model()
    feature_columns = load_feature_columns()
    
    if feature_matrix is None or model is None:
        st.error("❌ Required data not found. Please run the complete_workflow.py script first.")
        return
    
    st.markdown("---")
    
    # Get unique products with their descriptions
    product_info = feature_matrix.groupby('StockCode').agg({
        'Description': 'first',
        'AvgPrice': 'mean',
        'TotalQuantity': 'mean',
        'TotalRevenue': 'sum'
    }).reset_index()
    
    # Filter to products with decent data
    product_info = product_info[product_info['TotalRevenue'] > 1000]
    product_info = product_info.sort_values('TotalRevenue', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Product Selection")
        
        # Product selector
        selected_product = st.selectbox(
            "Select Product",
            options=product_info['StockCode'].tolist()[:100],  # Top 100 products
            format_func=lambda x: f"{x} - {product_info[product_info['StockCode']==x]['Description'].values[0][:30]}..."
        )
        
        # Get product data
        product_data = product_info[product_info['StockCode'] == selected_product].iloc[0]
        product_features = feature_matrix[feature_matrix['StockCode'] == selected_product]
        
        st.markdown("### Product Details")
        st.markdown(f"**Description**: {product_data['Description']}")
        
        current_price = product_data['AvgPrice']
        avg_quantity = product_data['TotalQuantity']
        
        st.metric("Average Price", f"£{current_price:.2f}")
        st.metric("Avg Weekly Quantity", f"{avg_quantity:.0f} units")
        
        st.markdown("### Price Input")
        
        # Price slider
        min_price = max(0.01, current_price * 0.5)
        max_price = current_price * 2.0
        
        test_price = st.slider(
            "Test Price",
            min_value=float(min_price),
            max_value=float(max_price),
            value=float(current_price),
            step=0.10,
            format="£%.2f"
        )
        
        # Make prediction using the model
        st.markdown("### Model Prediction")
        
        # Get a sample observation for this product to use as base features
        if len(product_features) > 0:
            base_obs = product_features.iloc[-1:].copy()  # Most recent observation
            
            # Update price in the features
            base_obs['AvgPrice'] = test_price
            
            # Update price-relative features
            hist_avg = current_price
            if hist_avg > 0:
                base_obs['PriceRelativeToAvg'] = test_price / hist_avg
            
            # Get prediction
            try:
                X_pred = base_obs[feature_columns]
                predicted_qty = max(0, model.predict(X_pred)[0])
                predicted_revenue = test_price * predicted_qty
                
                current_revenue = current_price * avg_quantity
                
                qty_change = predicted_qty - avg_quantity
                rev_change = predicted_revenue - current_revenue
                
                st.metric("Predicted Demand", f"{predicted_qty:.0f} units",
                         delta=f"{qty_change:+.0f}")
                st.metric("Predicted Revenue", f"£{predicted_revenue:.2f}",
                         delta=f"£{rev_change:+.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                predicted_qty = avg_quantity
                predicted_revenue = test_price * predicted_qty
    
    with col2:
        st.markdown("### Demand Curve Simulation")
        
        # Generate demand curve using the model
        prices = np.linspace(min_price, max_price, 50)
        demands = []
        revenues = []
        
        for p in prices:
            try:
                sim_obs = product_features.iloc[-1:].copy()
                sim_obs['AvgPrice'] = p
                if current_price > 0:
                    sim_obs['PriceRelativeToAvg'] = p / current_price
                
                X_sim = sim_obs[feature_columns]
                pred_qty = max(0, model.predict(X_sim)[0])
                demands.append(pred_qty)
                revenues.append(p * pred_qty)
            except:
                demands.append(0)
                revenues.append(0)
        
        curve_df = pd.DataFrame({
            'Price': prices,
            'Demand': demands,
            'Revenue': revenues
        })
        
        # Find optimal
        optimal_idx = np.argmax(revenues)
        optimal_price = prices[optimal_idx]
        optimal_demand = demands[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Demand curve
        fig.add_trace(go.Scatter(
            x=curve_df['Price'], y=curve_df['Demand'],
            name='Predicted Demand',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Revenue curve
        fig.add_trace(go.Scatter(
            x=curve_df['Price'], y=curve_df['Revenue'],
            name='Predicted Revenue',
            line=dict(color='#2ca02c', width=2),
            yaxis='y2'
        ))
        
        # Optimal price marker
        fig.add_vline(x=optimal_price, line_dash="dash", line_color="red",
                     annotation_text=f"Optimal: £{optimal_price:.2f}")
        
        # Current price marker
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray",
                     annotation_text=f"Current: £{current_price:.2f}")
        
        # Test price marker
        fig.add_vline(x=test_price, line_dash="solid", line_color="orange",
                     annotation_text=f"Test: £{test_price:.2f}")
        
        fig.update_layout(
            title='Price-Demand-Revenue Relationship',
            xaxis_title='Price (£)',
            yaxis=dict(title='Demand (units)', side='left', color='#1f77b4'),
            yaxis2=dict(title='Revenue (£)', side='right', overlaying='y', color='#2ca02c'),
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization Summary
        st.markdown("### Optimization Summary")
        
        col_a, col_b, col_c = st.columns(3)
        
        current_revenue = current_price * avg_quantity
        
        with col_a:
            st.markdown("**📍 Current**")
            st.markdown(f"Price: £{current_price:.2f}")
            st.markdown(f"Demand: {avg_quantity:.0f} units")
            st.markdown(f"Revenue: £{current_revenue:.2f}")
        
        with col_b:
            st.markdown("**🎯 Optimal**")
            st.markdown(f"Price: £{optimal_price:.2f}")
            st.markdown(f"Demand: {optimal_demand:.0f} units")
            st.markdown(f"Revenue: £{optimal_revenue:.2f}")
        
        with col_c:
            st.markdown("**📈 Potential Uplift**")
            price_change = (optimal_price - current_price) / current_price * 100 if current_price > 0 else 0
            revenue_change = (optimal_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
            st.markdown(f"Price change: {price_change:+.1f}%")
            st.markdown(f"**Revenue uplift: {revenue_change:+.1f}%**")
    
    # Show elasticity analysis if available
    st.markdown("---")
    
    if elasticity_analysis is not None and len(elasticity_analysis) > 0:
        st.markdown("### 📋 Top Pricing Recommendations")
        
        # Display top recommendations
        display_cols = ['StockCode', 'Description', 'CurrentAvgPrice', 'OptimalPrice', 
                       'ElasticityClass', 'RevenueUplift']
        
        if all(col in elasticity_analysis.columns for col in display_cols):
            top_recommendations = elasticity_analysis.nlargest(10, 'RevenueUplift')[display_cols].copy()
            top_recommendations.columns = ['Code', 'Description', 'Current Price', 'Optimal Price', 
                                          'Elasticity', 'Revenue Uplift %']
            
            # Format columns
            top_recommendations['Description'] = top_recommendations['Description'].str[:40]
            top_recommendations['Current Price'] = top_recommendations['Current Price'].apply(lambda x: f"£{x:.2f}")
            top_recommendations['Optimal Price'] = top_recommendations['Optimal Price'].apply(lambda x: f"£{x:.2f}")
            top_recommendations['Revenue Uplift %'] = top_recommendations['Revenue Uplift %'].apply(lambda x: f"+{x:.1f}%")
            
            st.dataframe(top_recommendations, use_container_width=True, hide_index=True)


def show_about_page():
    """Display the about/project info page."""
    
    st.markdown("## ℹ️ About This Project")
    
    # Load real metrics for the about page
    model_comparison = load_model_comparison()
    feature_matrix = load_feature_matrix()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Smart Pricing Engine
        
        This application demonstrates an end-to-end machine learning workflow for 
        price elasticity analysis and demand forecasting. The project showcases 
        data processing, feature engineering, model training, and deployment as 
        an interactive web application.
        
        ### Business Problem
        
        Pricing is one of the most critical decisions for retail businesses. Traditional pricing 
        relies on intuition or simple rules, but these approaches fail to capture the complex 
        relationship between price and customer demand. This project uses machine learning to 
        predict how demand changes with price, enabling data-driven pricing decisions.
        
        ### Technical Stack
        
        - **Languages**: Python 3.9+
        - **Data Processing**: pandas, NumPy
        - **Machine Learning**: scikit-learn, XGBoost
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Web Framework**: Streamlit
        - **Version Control**: Git
        
        ### Methodology
        
        1. **Data Source**: UCI Online Retail II Dataset (~1M transactions)
        2. **Feature Engineering**: 28 features including temporal, price context, and lag features
        3. **Models**: Baseline, Linear Regression, Random Forest, XGBoost
        4. **Evaluation**: RMSE, MAE, R², MAPE with time-series train/test split
        5. **Price Elasticity**: Model-based demand curves and revenue optimization
        
        ### Key Findings
        """)
        
        if model_comparison is not None:
            best_model = model_comparison['R2'].idxmax()
            best_r2 = model_comparison['R2'].max()
            st.markdown(f"""
            - **Best performing model**: {best_model} with R² = {best_r2:.3f}
            - All ML models significantly outperformed the baseline
            - Historical demand (lag features) proved to be the strongest predictors
            - Price elasticity varies significantly across product categories
            - 43 of 50 analyzed products showed >5% revenue uplift potential
            """)
        
        st.markdown("""
        ### Lessons Learned
        
        - Feature engineering is often more impactful than model selection
        - Simple models can outperform complex ones—always test multiple approaches
        - Price optimization must consider both demand elasticity and revenue maximization
        - Interactive tools make ML insights accessible to business stakeholders
        """)
    
    with col2:
        st.markdown("### Project Stats")
        
        if feature_matrix is not None:
            st.metric("Records Processed", f"{len(feature_matrix):,}")
            st.metric("Products Analyzed", f"{feature_matrix['StockCode'].nunique():,}")
            st.metric("Features Created", "28")
            st.metric("Models Trained", "4")
        
        st.markdown("---")
        
        st.markdown("### Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/JustinMC-data/smart-pricing-engine)
        - [Dataset Source](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Acknowledgments
    
    - UCI Machine Learning Repository for the Online Retail II dataset
    
    ---
    
    *Built with Python, Streamlit, and scikit-learn*
    """)


if __name__ == "__main__":
    main()
