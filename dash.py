import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing Intelligence Dashboard ",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5d6d7e;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #eaf2f8;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Function for Safe Division ---
def safe_divide(numerator, denominator):
    """A helper function to prevent division by zero errors."""
    if isinstance(numerator, (int, float, np.number)) and isinstance(denominator, (int, float, np.number)):
        return numerator / denominator if denominator != 0 else 0
    return np.divide(numerator, denominator, out=np.zeros_like(numerator.astype(float)), where=(denominator!=0))

# --- Data Loading and Preparation ---
@st.cache_data
def load_and_prepare_data():
    """
    Load, clean, and prepare all datasets from the 'data/' directory.
    Standardizes column names and calculates key derived metrics.
    """
    try:
        # Load datasets
        facebook_df = pd.read_csv('data/Facebook.csv')
        google_df = pd.read_csv('data/Google.csv')
        tiktok_df = pd.read_csv('data/TikTok.csv')
        business_df = pd.read_csv('data/Business.csv')

        # Standardize column names
        for df in [facebook_df, google_df, tiktok_df]:
            df.rename(columns={'attributed revenue': 'attributed_revenue'}, inplace=True)

        business_df.rename(columns={
            '# of orders': 'orders',
            '# of new orders': 'new_orders',
            'new customers': 'new_customers',
            'total revenue': 'total_revenue',
            'gross profit': 'gross_profit',
            'COGS': 'cogs'
        }, inplace=True)

        # Add platform identifiers
        facebook_df['platform'] = 'Facebook'
        google_df['platform'] = 'Google'
        tiktok_df['platform'] = 'TikTok'

        # Combine marketing data
        marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)

        # Convert date columns
        marketing_df['date'] = pd.to_datetime(marketing_df['date'])
        business_df['date'] = pd.to_datetime(business_df['date'])

        # Calculate derived marketing metrics
        marketing_df['ctr'] = safe_divide(marketing_df['clicks'], marketing_df['impression']) * 100
        marketing_df['cpc'] = safe_divide(marketing_df['spend'], marketing_df['clicks'])
        marketing_df['roas'] = safe_divide(marketing_df['attributed_revenue'], marketing_df['spend'])
        marketing_df['cpm'] = safe_divide(marketing_df['spend'], marketing_df['impression']) * 1000

        # Calculate derived business metrics
        business_df['avg_order_value'] = safe_divide(business_df['total_revenue'], business_df['orders'])
        business_df['profit_margin'] = safe_divide(business_df['gross_profit'], business_df['total_revenue']) * 100
        business_df['pct_new_customers'] = safe_divide(business_df['new_customers'], business_df['orders']) * 100

        # Replace potential infinite values
        marketing_df.replace([np.inf, -np.inf], 0, inplace=True)
        business_df.replace([np.inf, -np.inf], 0, inplace=True)

        return marketing_df, business_df

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure CSV files are in a 'data/' directory.")
        return None, None
    except KeyError as e:
        st.error(f"A required column is missing: {e}. Please check your CSV files.")
        return None, None

# --- Main App ---
def main():
    # --- Headers and Introduction ---
    st.markdown('<h1 class="main-header">🚀 Marketing Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An integrated view of marketing performance and its impact on core business outcomes.</p>', unsafe_allow_html=True)

    with st.expander("ℹ️ How to Use This Dashboard"):
        st.write("""
        This dashboard is designed to provide a comprehensive overview of your marketing efforts and their direct influence on business results.
        - **Use the filters on the left** to slice the data by date, marketing platform, or state.
        - **Executive Summary:** Start here for a high-level view of your most critical business health metrics like Marketing Efficiency Ratio (MER) and Customer Acquisition Cost (CAC).
        - **Performance Deep Dives:** Explore the subsequent sections to analyze specific platforms, campaigns, and customer trends.
        - **Key Insights:** Conclude with the dynamically generated insights at the bottom for actionable recommendations.
        """)
    # --- Load Data ---
    marketing_df, business_df = load_and_prepare_data()
    if marketing_df is None or business_df is None:
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.header("📅 Date Range Filter")
    min_date = min(marketing_df['date'].min(), business_df['date'].min()).to_pydatetime()
    max_date = max(marketing_df['date'].max(), business_df['date'].max()).to_pydatetime()

    date_range = st.sidebar.date_input(
        "Select Date Range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if len(date_range) != 2:
        st.warning("Please select a valid start and end date.")
        st.stop()
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    st.sidebar.header("📊 Data Filters")
    unique_platforms = sorted(marketing_df['platform'].unique())
    platforms = st.sidebar.multiselect("Select Platforms", options=unique_platforms, default=unique_platforms)

    unique_states = sorted(marketing_df['state'].unique())
    default_states = list(unique_states)[:min(5, len(unique_states))]
    states = st.sidebar.multiselect("Select States", options=unique_states, default=default_states)

    if not platforms or not states:
        st.warning("Please select at least one platform and one state to display data.")
        st.stop()

    # --- Filter Data Based on Selections ---
    marketing_filtered = marketing_df[
        (marketing_df['date'] >= start_date) & (marketing_df['date'] <= end_date) &
        (marketing_df['platform'].isin(platforms)) &
        (marketing_df['state'].isin(states))
    ]
    business_filtered = business_df[
        (business_df['date'] >= start_date) & (business_df['date'] <= end_date)
    ]

    if marketing_filtered.empty or business_filtered.empty:
        st.warning("No data available for the selected filters. Please expand your selection.")
        st.stop()

    # --- Calculate Core Metrics ---
    total_spend = marketing_filtered['spend'].sum()
    total_revenue = business_filtered['total_revenue'].sum()
    total_profit = business_filtered['gross_profit'].sum()
    total_orders = business_filtered['orders'].sum()
    total_new_customers = business_filtered['new_customers'].sum()

    mer = safe_divide(total_revenue, total_spend)
    cac = safe_divide(total_spend, total_new_customers)
    direct_roas = safe_divide(marketing_filtered['attributed_revenue'].sum(), total_spend)

    # --- Executive Summary ---
    st.header("📈 Executive Summary: The Bottom Line")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Gross Profit", f"${total_profit:,.0f}", f"{safe_divide(total_profit, total_revenue)*100:.1f}% Margin")
    col3.metric("Marketing Efficiency Ratio (MER)", f"{mer:.2f}x", "Total Revenue / Total Spend")
    col4.metric("Customer Acquisition Cost (CAC)", f"${cac:,.2f}", "Total Spend / New Customers")

    st.markdown("---")

    # --- Marketing Performance Deep Dive ---
    st.header("🎯 Marketing Performance Deep Dive")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Marketing Spend", f"${total_spend:,.0f}")
    m_col2.metric("Attributed ROAS", f"{direct_roas:.2f}x", "Attributed Revenue / Spend")
    m_col3.metric("Total Clicks", f"{marketing_filtered['clicks'].sum():,.0f}", f"Avg CPC: ${safe_divide(total_spend, marketing_filtered['clicks'].sum()):.2f}")
    m_col4.metric("Total Impressions", f"{marketing_filtered['impression'].sum():,.0f}", f"Avg CPM: ${safe_divide(total_spend, marketing_filtered['impression'].sum())*1000:.2f}")

    # --- Performance Over Time ---
    st.subheader("Performance Trends Over Time")

    weekly_marketing = marketing_filtered.set_index('date').resample('W-MON').sum().reset_index()
    weekly_business = business_filtered.set_index('date').resample('W-MON').sum().reset_index()
    weekly_merged = pd.merge(weekly_marketing, weekly_business, on='date', how='inner')

    if not weekly_merged.empty:
        weekly_merged['mer'] = safe_divide(weekly_merged['total_revenue'], weekly_merged['spend'])
        weekly_merged['cac'] = safe_divide(weekly_merged['spend'], weekly_merged['new_customers'])

        fig_trends = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=('Weekly Spend, Revenue & Profit', 'Weekly MER & CAC'),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        fig_trends.add_trace(go.Bar(x=weekly_merged['date'], y=weekly_merged['spend'], name='Spend', marker_color='#F39C12'), row=1, col=1)
        fig_trends.add_trace(go.Scatter(x=weekly_merged['date'], y=weekly_merged['total_revenue'], name='Total Revenue', mode='lines+markers', line=dict(color='#2E86C1')), row=1, col=1)
        fig_trends.add_trace(go.Scatter(x=weekly_merged['date'], y=weekly_merged['gross_profit'], name='Gross Profit', mode='lines+markers', line=dict(color='#28B463')), row=1, col=1)

        fig_trends.add_trace(go.Scatter(x=weekly_merged['date'], y=weekly_merged['mer'], name='MER', line=dict(color='#8E44AD')), row=2, col=1, secondary_y=False)
        fig_trends.add_trace(go.Scatter(x=weekly_merged['date'], y=weekly_merged['cac'], name='CAC', line=dict(color='#E74C3C')), row=2, col=1, secondary_y=True)

        fig_trends.update_layout(height=600, title_text="Weekly Business & Marketing Cadence")
        fig_trends.update_yaxes(title_text="<b>Revenue & Spend ($)</b>", row=1, col=1)
        fig_trends.update_yaxes(title_text="<b>MER (Ratio)</b>", row=2, col=1, secondary_y=False)
        fig_trends.update_yaxes(title_text="<b>CAC ($)</b>", row=2, col=1, secondary_y=True)

        st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.info("Not enough weekly data to show trends.")

    st.markdown("---")
    # --- Customer Insights ---
    st.header("👥 Customer Acquisition Insights")
    cust_col1, cust_col2 = st.columns(2)

    returning_customers_orders = total_orders - total_new_customers

    with cust_col1:
        new_cust_val = max(0, total_new_customers)
        ret_cust_val = max(0, returning_customers_orders)

        fig_cust_pie = px.pie(
            names=['New Customers', 'Returning Customers'],
            values=[new_cust_val, ret_cust_val],
            title='New vs. Returning Customer Orders',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig_cust_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_cust_pie, use_container_width=True)

    with cust_col2:
        st.metric("Total New Customers Acquired", f"{total_new_customers:,.0f}")
        st.metric("Avg. Order Value (AOV)", f"${business_filtered['avg_order_value'].mean():,.2f}")
        st.metric("% of Orders from New Customers", f"{safe_divide(total_new_customers, total_orders)*100:.1f}%")

    st.markdown("---")

    # --- Channel & Campaign Analysis ---
    st.header("📊 Channel & Campaign Performance")

    # Platform Analysis
    st.subheader("Platform Contribution & Efficiency")
    platform_fig, platform_metrics = create_platform_analysis(marketing_filtered)
    if platform_fig:
        st.plotly_chart(platform_fig, use_container_width=True)
        st.dataframe(platform_metrics)
    else:
        st.info("No platform data to display.")
    # Campaign Analysis
    st.subheader("Top & Bottom Performing Campaigns")
    campaign_metrics = marketing_filtered.groupby(['campaign', 'platform']).agg(
        spend=('spend', 'sum'),
        attributed_revenue=('attributed_revenue', 'sum'),
        clicks=('clicks', 'sum')
    ).reset_index()
    campaign_metrics['roas'] = safe_divide(campaign_metrics['attributed_revenue'], campaign_metrics['spend'])

    top_campaigns = campaign_metrics.nlargest(10, 'roas')

    fig_camp = px.scatter(
        campaign_metrics,
        x='spend',
        y='roas',
        size='clicks',
        color='platform',
        hover_name='campaign',
        hover_data={'spend': ':.2f', 'roas': ':.2f', 'clicks': ':,.0f'},
        log_x=True,
        size_max=60, 
        height=600,
        title='Campaign Efficiency: Spend vs. ROAS (Bubble size = Clicks)',
        labels={'spend': 'Total Spend ($) - Log Scale', 'roas': 'Attributed ROAS'},
        template='plotly_white'
    )
    fig_camp.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig_camp, use_container_width=True)
    
    st.write("Top 10 Campaigns by ROAS:")
    st.dataframe(top_campaigns.round(2))

    # --- Insights and Recommendations ---
    st.markdown("---")
    st.header("💡 Key Insights & Recommendations")

    if not platform_metrics.empty:
        # --- Calculations for Insights ---
        best_roas_platform = platform_metrics.loc[platform_metrics['roas'].idxmax()]
        
        # Check if there is more than one platform to find the worst one
        if len(platform_metrics) > 1:
            worst_roas_platform = platform_metrics.loc[platform_metrics['roas'].idxmin()]
        else:
            worst_roas_platform = None # No "worst" platform if only one is selected

        avg_ctr = safe_divide(marketing_filtered['clicks'].sum(), marketing_filtered['impression'].sum()) * 100

        # --- Insight 1: Overall Performance ---
        insight_overall = f"""
        📈 **Overall Performance**: The business achieved a blended **{direct_roas:.2f}x ROAS** (Return On Ad Spend) and a **{mer:.2f}x MER** (Marketing Efficiency Ratio), indicating a healthy overall return.
        """
        st.markdown(f'<div class="insight-box">{insight_overall}</div>', unsafe_allow_html=True)

        # --- Insight 2: Best Performing Platform ---
        insight_best_platform = f"""
        🏆 **Best Performing Platform**: **{best_roas_platform['platform']}** is the most efficient channel with a **{best_roas_platform['roas']:.2f}x ROAS**. Campaigns here are driving the highest returns.
        """
        st.markdown(f'<div class="insight-box">{insight_best_platform}</div>', unsafe_allow_html=True)

        # --- Insight 3: Underperforming Platform (if applicable) ---
        if worst_roas_platform is not None and worst_roas_platform['platform'] != best_roas_platform['platform']:
            insight_worst_platform = f"""
            📉 **Underperforming Platform**: **{worst_roas_platform['platform']}** shows the lowest efficiency at **{worst_roas_platform['roas']:.2f}x ROAS**. Consider reallocating budget from this channel to higher performers.
            """
            st.markdown(f'<div class="insight-box">{insight_worst_platform}</div>', unsafe_allow_html=True)

        # --- Insight 4: Customer Acquisition ---
        insight_customer = f"""
        👥 **Customer Acquisition**: New customers were acquired at an average cost of **${cac:,.2f} (CAC)**. With **{safe_divide(total_new_customers, total_orders)*100:.1f}%** of orders coming from new customers, the acquisition funnel is healthy.
        """
        st.markdown(f'<div class="insight-box">{insight_customer}</div>', unsafe_allow_html=True)

        # --- Insight 5: Creative Engagement ---
        insight_ctr = f"""
        🎯 **Good Click-Through Rate**: The average Click-Through Rate (CTR) is **{avg_ctr:.2f}%**, suggesting that ad creatives are effectively capturing audience attention.
        """
        st.markdown(f'<div class="insight-box">{insight_ctr}</div>', unsafe_allow_html=True)

    else:
        st.info("Not enough data to generate insights for the selected filters.")

def create_platform_analysis(marketing_df):
    """Create platform comparison analysis."""
    if marketing_df.empty:
        return None, pd.DataFrame()

    platform_metrics = marketing_df.groupby('platform').agg(
        spend=('spend', 'sum'),
        attributed_revenue=('attributed_revenue', 'sum'),
        clicks=('clicks', 'sum'),
        impression=('impression', 'sum')
    ).reset_index()

    platform_metrics['roas'] = safe_divide(platform_metrics['attributed_revenue'], platform_metrics['spend'])
    platform_metrics['cpc'] = safe_divide(platform_metrics['spend'], platform_metrics['clicks'])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('Spend vs. Attributed Revenue', 'Attributed ROAS', 'Revenue Contribution'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'pie'}]]
    )
    fig.add_trace(go.Bar(name='Spend', x=platform_metrics['platform'], y=platform_metrics['spend'], marker_color='#F39C12'), row=1, col=1)
    fig.add_trace(go.Bar(name='Attributed Revenue', x=platform_metrics['platform'], y=platform_metrics['attributed_revenue'], marker_color='#2E86C1'), row=1, col=1)
    fig.add_trace(go.Bar(x=platform_metrics['platform'], y=platform_metrics['roas'], name='ROAS', marker_color='#28B463'), row=1, col=2)
    fig.add_trace(go.Pie(labels=platform_metrics['platform'], values=platform_metrics['attributed_revenue'], name="Revenue", hole=0.4), row=1, col=3)

    fig.update_layout(height=400, showlegend=True, title_text="Platform Performance Breakdown")
    return fig, platform_metrics.round(2)

if __name__ == "__main__":
    main()