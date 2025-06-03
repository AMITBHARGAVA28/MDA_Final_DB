# -*- coding: utf-8 -*-
"""Horizon Europe Dashboard - Complete Version (Fixed)"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import warnings
import duckdb
import zipfile
import io

# Configuration
warnings.filterwarnings('ignore')
st.set_page_config(
    layout="wide",
    page_title="Horizon Europe Dashboard",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ====================
# DATA LOADING & PREPROCESSING
# ====================

@st.cache_data(ttl=1800, show_spinner="Loading and processing data...")
def load_data():
    """Load and preprocess all data with DuckDB optimizations"""
    con = duckdb.connect()
    try:
        with zipfile.ZipFile("data.zip") as z:
            with z.open("horizon_organizations.csv") as org_file:
                org_df = pd.read_csv(org_file, encoding='latin1')
            with z.open("horizon_projects.csv") as proj_file:
                proj_df = pd.read_csv(proj_file, encoding='latin1')

        con.register('org_df', org_df)
        con.register('proj_df', proj_df)

        proj_df = con.execute("""
            SELECT 
                *,
                TRY_CAST(strptime(startDate, '%d-%m-%Y') AS DATE) AS startDate,
                TRY_CAST(strptime(endDate, '%d-%m-%Y') AS DATE) AS endDate,
                YEAR(TRY_CAST(strptime(startDate, '%d-%m-%Y') AS DATE)) AS start_year,
                CASE 
                    WHEN fundingCategory IN ('ERC', 'MSCA') THEN 'Excellent Science'
                    WHEN fundingCategory LIKE 'EIC%' THEN 'Innovative Europe'
                    WHEN fundingCategory IN ('RIA / IA') THEN 'Global Challenges'
                    WHEN fundingCategory = 'CSA' THEN 'Cross-cutting'
                    ELSE 'Other'
                END AS pillar
            FROM proj_df
            WHERE CountryName IS NOT NULL 
              AND topic_label IS NOT NULL
              AND ecMaxContribution IS NOT NULL
        """).df()

        org_df = con.execute("""
            SELECT 
                *,
                SPLIT_PART(geolocation, ',', 1)::FLOAT AS lat,
                SPLIT_PART(geolocation, ',', 2)::FLOAT AS lon,
                CASE 
                    WHEN activityType = 'HES' THEN 'University'
                    WHEN activityType = 'REC' THEN 'Research Organization'
                    WHEN activityType = 'PRC' THEN 'Private Company'
                    WHEN activityType = 'PUB' THEN 'Public Institution'
                    ELSE 'Other'
                END AS org_type,
                TRY_CAST(ecContribution AS FLOAT) AS ecContribution
            FROM org_df
            WHERE Country IS NOT NULL
              AND organisationID IS NOT NULL
        """).df()

    finally:
        con.close()

    proj_df['SME'] = proj_df['SME'].fillna(False)
    org_df['SME'] = org_df['SME'].fillna(False)

    return org_df, proj_df

# Load data
org_df, proj_df = load_data()

# ====================
# OPTIMIZED NETWORK ANALYSIS
# ====================

@st.cache_data(ttl=3600, show_spinner="Building collaboration network...")
def build_collaboration_network(_org_df, min_projects=3, max_nodes=150):
    """Optimized network construction with size limits"""
    # Handle empty data case
    if _org_df.empty:
        return nx.Graph()

    # Fast edge counting with DuckDB
    con = duckdb.connect()
    con.register('org_df', _org_df)

    # Get all project-organization pairs
    project_orgs = con.execute("""
        SELECT projectID, ARRAY_AGG(DISTINCT organisationID) AS orgs
        FROM org_df
        GROUP BY projectID
        HAVING COUNT(DISTINCT organisationID) > 1
    """).df()

    # Count collaborations using defaultdict
    edge_counter = defaultdict(int)
    for orgs in project_orgs['orgs']:
        for u, v in combinations(sorted(orgs), 2):
            edge_counter[(u, v)] += 1

    # Limit edges to top 5000 collaborations
    top_edges = sorted(edge_counter.items(), key=lambda x: -x[1])[:5000]

    # Create network
    G = nx.Graph()
    G.add_edges_from((u, v, {'weight': w}) for (u, v), w in top_edges)

    # Filter nodes by degree
    degrees = dict(G.degree())
    filtered_nodes = [n for n, d in degrees.items() if d >= min_projects]
    filtered_nodes = sorted(filtered_nodes, key=lambda x: -degrees[x])[:max_nodes]

    # Add node attributes
    org_stats = con.execute("""
        SELECT 
            organisationID,
            SUM(ecContribution) AS total_funding,
            COUNT(DISTINCT projectID) AS project_count,
            MAX(org_type) AS org_type,
            MAX(Country) AS country,
            MAX(name) AS name
        FROM org_df
        GROUP BY organisationID
    """).df()

    con.close()

    org_stats = org_stats.set_index('organisationID')
    for node in G.nodes():
        if node in org_stats.index:
            G.nodes[node].update(org_stats.loc[node].to_dict())

    return G.subgraph(filtered_nodes)

def create_network_viz(G):
    """Enhanced network visualization with Plotly"""
    if G.number_of_nodes() == 0:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No collaboration data available with current filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title="Top Organization Collaboration Network",
            title_x=0.5,
            height=720
        )
        return fig

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, text, colors, sizes = [], [], [], [], []

    color_map = {
        'Research Organization': '#1f77b4',
        'University': '#ff7f0e',
        'Private Company': '#2ca02c',
        'Public Institution': '#9467bd',
        'Other': '#7f7f7f'
    }

    for node in G.nodes():
        node_data = G.nodes[node]
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node size by number of connections (collaborations)
        collab_count = len(list(G.neighbors(node)))
        sizes.append(10 + collab_count * 1.5)

        # Tooltip with detailed info
        text.append(
            f"<b>{node_data.get('name', node)}</b><br>"
            f"Type: {node_data.get('org_type', 'N/A')}<br>"
            f"Country: {node_data.get('country', 'N/A')}<br>"
            f"Projects: {node_data.get('project_count', 0)}<br>"
            f"Funding: €{node_data.get('total_funding', 0)/1e6:.2f}M<br>"
            f"Collaborations: {collab_count}"
        )

        org_type = node_data.get('org_type', 'Other')
        colors.append(color_map.get(org_type, '#7f7f7f'))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.4, color='rgba(150,150,150,0.3)'),
                hoverinfo='none'
            ),
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=1.5, color='white'),
                    opacity=0.85,
                    showscale=False
                ),
                text=text,
                hoverinfo='text'
            )
        ],
        layout=go.Layout(
            title="Top Organization Collaboration Network",
            title_x=0.5,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            height=720,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
    )

    for org_type, hexcolor in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=hexcolor),
            legendgroup=org_type,
            showlegend=True,
            name=org_type
        ))

    return fig

# ====================
# FORECASTING FUNCTIONS (FIXED TO PREVENT NEGATIVE VALUES)
# ====================

@st.cache_data(show_spinner=False)
def generate_prophet_forecast(_data, years=5):
    """Generate Prophet forecast with error handling and log‐safe transform."""
    try:
        # Group and sum by year, then floor at 1 to avoid log(0)
        ts = (
            _data.groupby('start_year')['ecMaxContribution']
            .sum()
            .clip(lower=1.0)
            .reset_index()
        )
        ts.columns = ['ds', 'y']
        ts['ds'] = pd.to_datetime(ts['ds'], format='%Y')

        # Log‐transform
        ts['y'] = np.log(ts['y'])

        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False
        )
        model.fit(ts)

        future = model.make_future_dataframe(periods=years, freq='Y')
        forecast = model.predict(future)

        # Reverse‐log: ensure any negative predictions (unlikely) are floored before exp
        forecast['yhat'] = np.exp(forecast['yhat'].clip(lower=0))
        forecast['yhat_lower'] = np.exp(forecast['yhat_lower'].clip(lower=0))
        forecast['yhat_upper'] = np.exp(forecast['yhat_upper'].clip(lower=0))

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    except Exception as e:
        st.error(f"Prophet forecast failed: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def generate_arima_forecast(_data, years=5, order=(1, 1, 1)):
    """Generate ARIMA forecast with error handling and log‐safe transform."""
    try:
        # Sum by year, floor to 1 to avoid log(0)
        ts_series = (
            _data.groupby('start_year')['ecMaxContribution']
            .sum()
            .clip(lower=1.0)
        )
        ts_series.index = pd.to_datetime(ts_series.index, format='%Y')

        # Log‐transform
        ts_log = np.log(ts_series)

        model = ARIMA(ts_log, order=order)
        model_fit = model.fit()

        # Forecast log‐series, then exp & clip negative predictions
        forecast_log = model_fit.forecast(steps=years)
        forecast = np.exp(forecast_log.clip(lower=0))

        return forecast

    except Exception as e:
        st.error(f"ARIMA forecast failed: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def generate_linear_forecast(_data, years=5):
    """Generate linear trend forecast with log transform"""
    try:
        ts_data = _data.groupby('start_year')['ecMaxContribution'].sum().reset_index()
        X = ts_data['start_year'].values.reshape(-1, 1)
        y = ts_data['ecMaxContribution'].values

        # Apply log transform to prevent negative forecasts
        log_y = np.log(y)

        model = LinearRegression()
        model.fit(X, log_y)

        future_years = np.arange(
            ts_data['start_year'].max() + 1,
            ts_data['start_year'].max() + years + 1
        )
        log_forecast = model.predict(future_years.reshape(-1, 1))
        # Reverse log transform
        forecast = np.exp(log_forecast)

        return pd.Series(forecast, index=future_years)
    except Exception as e:
        st.error(f"Linear forecast failed: {str(e)}")
        return None

# ====================
# VISUALIZATION HELPERS (WITH EMPTY DATA HANDLING)
# ====================

def create_sunburst_chart(data):
    """Interactive sunburst chart of funding distribution."""
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title='Funding Distribution by Country and Research Area')
        return fig

    # Aggregate data
    agg_data = data.groupby(['CountryName', 'pillar', 'topic_label']).agg(
        total_funding=('ecMaxContribution', 'sum'),
        project_count=('id', 'nunique')
    ).reset_index()

    # Convert total_funding into millions (M) for readability
    agg_data['funding_M'] = agg_data['total_funding'] / 1e6

    fig = px.sunburst(
        agg_data,
        path=['CountryName', 'pillar', 'topic_label'],
        values='funding_M',
        color='project_count',
        color_continuous_scale='Blues',
        title='Funding Distribution by Country and Research Area (in €M)',
        height=700,
        branchvalues='total'
    )
    fig.update_traces(
        texttemplate='<b>%{label}</b><br>%{percentParent:.1%}',
        insidetextorientation='radial',
        hovertemplate=(
            "<b>%{label}</b><br>Total Funding: €%{value:.2f}M<br>Projects: %{color:,}"
        )
    )
    fig.update_layout(
        margin=dict(t=40, b=0, l=0, r=0)
    )

    return fig

def create_funding_bubble_chart(data):
    """Interactive bubble chart of research topics."""
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title='Research Topic Funding Profile')
        return fig

    agg_data = data.groupby('topic_label').agg(
        total_funding=('ecMaxContribution', 'sum'),
        project_count=('id', 'nunique'),
        avg_funding=('ecMaxContribution', 'mean'),
        country_count=('CountryName', 'nunique')
    ).reset_index()

    # Convert to human‐friendly units:
    agg_data['total_funding_M'] = agg_data['total_funding'] / 1e6
    agg_data['avg_funding_M'] = agg_data['avg_funding'] / 1e6

    fig = px.scatter(
        agg_data,
        x='project_count',
        y='avg_funding_M',         # <-- millions on y‐axis
        size='total_funding_M',    # <-- bubble size in millions
        color='country_count',
        hover_name='topic_label',
        log_x=True,
        size_max=60,
        labels={
            'project_count': 'Number of Projects (log scale)',
            'avg_funding_M': 'Avg Funding per Project (€M)',
            'country_count': 'Participating Countries',
            'total_funding_M': 'Total Funding (€M)'
        },
        title='Research Topic Funding Profile (in €M)',
        color_continuous_scale='Viridis'
    )
    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        selector=dict(mode='markers')
    )
    fig.update_layout(hovermode='closest')
    return fig

def create_pillar_distribution(data):
    """Pillar funding distribution visualization"""
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title='Funding Distribution by Horizon Europe Pillars')
        return fig

    pillar_data = data.groupby('pillar').agg(
        total_funding=('ecMaxContribution', 'sum'),
        project_count=('id', 'nunique')
    ).reset_index()

    fig = px.pie(
        pillar_data,
        names='pillar',
        values='total_funding',
        title='Funding Distribution by Horizon Europe Pillars',
        hole=0.4,
        labels={'total_funding': 'Total Funding (€)'}
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Total Funding: €%{value:,.0f}<br>Projects: %{customdata}",
        customdata=pillar_data['project_count']
    )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    return fig

# ====================
# DASHBOARD LAYOUT (FIXED)
# ====================

def main():
    # Sidebar Filters
    st.sidebar.title("🌐 Dashboard Controls")

    with st.sidebar.expander("🔍 Global Filters", expanded=True):
        # Year range slider
        year_range = st.slider(
            "Project Years",
            int(proj_df['start_year'].min()),
            int(proj_df['start_year'].max()),
            (2021, 2027),
            help="Filter projects by their start year"
        )

        # Country multiselect
        countries = st.multiselect(
            "Select Countries",
            options=proj_df['CountryName'].unique(),
            default=['Germany', 'France', 'Spain', 'Italy'],
            help="Filter by participating countries"
        )

        # Funding category multiselect
        funding_cats = st.multiselect(
            "Funding Categories",
            options=proj_df['fundingCategory'].unique(),
            default=['PCP / PPI', 'COFUND / ERA-NET', 'RIA / IA'],
            help="Filter by Horizon Europe funding instruments"
        )

        # Organization type filter
        org_types = st.multiselect(
            "Organization Types",
            options=org_df['org_type'].unique(),
            default=['Research Organization', 'University'],
            help="Filter by organization type in network view"
        )

    # Apply filters
    filtered_proj = proj_df[
        (proj_df['start_year'] >= year_range[0]) &
        (proj_df['start_year'] <= year_range[1])
    ]

    if countries:
        filtered_proj = filtered_proj[filtered_proj['CountryName'].isin(countries)]

    if funding_cats:
        filtered_proj = filtered_proj[filtered_proj['fundingCategory'].isin(funding_cats)]

    filtered_org = org_df[
        (org_df['projectID'].isin(filtered_proj['id'])) &
        (org_df['org_type'].isin(org_types if org_types else org_df['org_type'].unique())) &
        (org_df['Country'].isin(countries if countries else org_df['Country'].unique()))
    ]

    # Main Dashboard
    st.title("🌍 Horizon Europe Funding Dashboard")
    st.markdown("""
    *Analyzing the EU's €95.5B research and innovation program (2021-2027)*  
    """)

    # Key Metrics Ribbon
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Funding",
            f"€{filtered_proj['ecMaxContribution'].sum()/1e9:.2f}B",
            help="Sum of EU contributions to selected projects"
        )
    with col2:
        st.metric(
            "Projects Count",
            f"{len(filtered_proj):,}",
            help="Number of projects matching filters"
        )
    with col3:
        st.metric(
            "Participating Countries",
            filtered_proj['CountryName'].nunique(),
            help="Number of unique countries in selected projects"
        )
    with col4:
        st.metric(
            "Avg. Project Size",
            f"€{filtered_proj['ecMaxContribution'].mean()/1e6:.1f}M",
            help="Average EU contribution per project"
        )

    # Impact Metrics
    st.subheader("🌱 Horizon Impact Metrics")
    icol1, icol2, icol3, icol4 = st.columns(4)
    with icol1:
        sme_funding = filtered_org[filtered_org['SME']]['ecContribution'].sum()
        st.metric(
            "SME Participation",
            f"€{sme_funding/1e9:.2f}B",
            f"{sme_funding/filtered_org['ecContribution'].sum():.1%} of total",
            help="Funding going to small and medium enterprises"
        )
    with icol2:
        intl_counts = (
            filtered_org
            .groupby('projectID')['Country']
            .nunique()
        )
        intl_projects = int((intl_counts > 1).sum())
        st.metric(
            "International Projects",
            f"{intl_projects:,}",
            f"{intl_projects/len(filtered_proj):.1%} of total",
            help="Projects with participants from multiple countries"
        )   
        
    with icol3:
        green_topics = ['climate', 'environment', 'energy', 'sustainab']
        green_mask = filtered_proj['topic_label'].str.contains(
            '|'.join(green_topics), case=False
        )
        st.metric(
            "Green Transition Funding",
            f"€{filtered_proj[green_mask]['ecMaxContribution'].sum()/1e9:.1f}B",
            help="Funding for climate and sustainability projects"
        )
    with icol4:
        widening_countries = [
            'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
            'Estonia', 'Hungary', 'Latvia', 'Lithuania',
            'Malta', 'Poland', 'Portugal', 'Romania',
            'Slovakia', 'Slovenia'
        ]
        widening_funding = filtered_proj[
            filtered_proj['CountryName'].isin(widening_countries)
        ]['ecMaxContribution'].sum()
        st.metric(
            "Widening Participation",
            f"€{widening_funding/1e9:.2f}B",
            f"{widening_funding/filtered_proj['ecMaxContribution'].sum():.1%} of total",
            help="Funding for less research-intensive EU countries"
        )

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Strategic Overview",
        "🔗 Collaboration Network",
        "💰 Funding Analysis",
        "🔮 Predictive Insights"
    ])

    with tab1:
        st.header("Strategic Funding Overview")

        col1, col2 = st.columns([3, 2])
        with col1:
            sunburst = create_sunburst_chart(filtered_proj)
            st.plotly_chart(sunburst, use_container_width=True)
        with col2:
            pillar = create_pillar_distribution(filtered_proj)
            st.plotly_chart(pillar, use_container_width=True)

            # Financial metrics explanation
            with st.expander("💡 Understanding the Metrics"):
                st.markdown("""
                **Key Financial Metrics:**
                - **EU Contribution (ecMaxContribution):** Maximum funding allocated from Horizon Europe
                - **Total Project Cost:** Complete budget including partner contributions
                - **Organization Funding:** Amount received by each participating entity

                **Horizon Europe Pillars:**
                1. **Excellent Science:** Frontier research (ERC, MSCA)
                2. **Global Challenges:** Thematic research missions
                3. **Innovative Europe:** Supporting startups and scale-ups
                4. **Cross-cutting:** Widening participation, research infrastructure
                """)

    with tab2:
        st.header("Research Collaboration Network")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Network Controls")
            min_collab = st.slider(
                "Minimum Collaborations",
                1, 20, 3,
                help="Filter organizations with at least this many joint projects"
            )
            max_nodes = st.slider(
                "Maximum Nodes to Display",
                50, 300, 150,
                help="Limit network complexity for better performance"
            )
            st.markdown("""
            **Network Insights:**
            - Node size = Total funding received
            - Color = Organization type
            - Edge thickness = Collaboration strength
            """)

        with col2:
            with st.spinner("Building collaboration network..."):
                G = build_collaboration_network(filtered_org, min_collab, max_nodes)
                fig = create_network_viz(G)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Funding Distribution Analysis")

        bubble = create_funding_bubble_chart(filtered_proj)
        st.plotly_chart(bubble, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top-Funded Research Topics – Cross-country Comparison")

            if filtered_proj.empty:
                st.info("No data available with the current filters.")
            else:
                # ── 1. pick up to N countries with most projects (keeps chart legible)
                N_COUNTRIES = 6
                sel_countries = (
                    filtered_proj['CountryName']
                    .value_counts()
                    .nlargest(N_COUNTRIES)
                    .index.tolist()
                )

                # ── 2. aggregate funding per (country, topic)
                agg = (
                    filtered_proj
                    .loc[filtered_proj['CountryName'].isin(sel_countries)]
                    .groupby(['CountryName', 'topic_label'], as_index=False)
                    .agg(total_funding=('ecMaxContribution', 'sum'))
                )

                # ── 3. within every country keep the TOP-5 funded topics
                agg = (
                    agg.sort_values(['CountryName', 'total_funding'], ascending=False)
                       .groupby('CountryName')
                       .head(5)
                )

                # ── 4. convert € to millions for easier reading
                agg['total_funding_m'] = agg['total_funding'] / 1e6

                # ── 5. build one grouped bar chart (common Y-axis = easy comparison)
                fig = px.bar(
                    agg,
                    x='topic_label',
                    y='total_funding_m',
                    color='CountryName',
                    barmode='group',
                    text='total_funding_m',
                    title='Top-Funded Topics (Millions €) – Side-by-Side Comparison',
                    labels={
                        'topic_label': 'Research Topic',
                        'total_funding_m': 'Funding (M €)'
                    },
                    height=600
                )

                # prettify
                fig.update_layout(
                    xaxis_tickangle=-45,
                    legend_title_text='Country',
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    margin=dict(l=10, r=10, t=60, b=120)
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Funding Forecasts")

        with st.expander("Forecast Settings", expanded=True):
            col1, col2 = st.columns(2)
            models = col1.multiselect(
                "Select Forecasting Models",
                options=["Prophet", "ARIMA", "Linear Trend"],
                default=["Prophet", "Linear Trend"]
            )
            forecast_years = col2.slider(
                "Forecast Horizon (years)",
                1, 15, 5
            )

            scope = st.radio(
                "Analysis Scope",
                options=["EU-Wide", "By Country", "By Funding Category"],
                horizontal=True
            )

            # Dynamic scope filters
            if scope == "By Country":
                country = st.selectbox(
                    "Select Country for Forecast",
                    options=filtered_proj['CountryName'].unique()
                )
                forecast_data = filtered_proj[filtered_proj['CountryName'] == country]
            elif scope == "By Funding Category":
                category = st.selectbox(
                    "Select Funding Category for Forecast",
                    options=filtered_proj['fundingCategory'].unique()
                )
                forecast_data = filtered_proj[filtered_proj['fundingCategory'] == category]
            else:
                forecast_data = filtered_proj

        # Generate forecast plot
        if len(forecast_data) >= 3:
            fig = go.Figure()

            # Historical data
            hist_data = forecast_data.groupby('start_year')['ecMaxContribution'].sum().reset_index()

            # Draw the historical line first
            fig.add_trace(go.Scatter(
                x=hist_data['start_year'],
                y=hist_data['ecMaxContribution'],
                name='Historical Funding',
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3)
            ))

            # Store last observed point for anchoring forecasts
            last_year = hist_data.iloc[-1, 0]
            last_val = hist_data.iloc[-1, 1]

            # ---------- Prophet ----------
            if "Prophet" in models:
                prophet_fcst = generate_prophet_forecast(forecast_data, forecast_years)
                if prophet_fcst is not None:
                    # keep only genuine future periods
                    future_only = prophet_fcst[prophet_fcst['ds'].dt.year > last_year]

                    # build an anchored series (start at last historical point)
                    x_vals = np.concatenate(([last_year], future_only['ds'].dt.year.values))
                    y_vals = np.concatenate(([last_val], future_only['yhat'].values))

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        name='Prophet Forecast',
                        mode='lines+markers',
                        line=dict(color='#ff7f0e', dash='dot')
                    ))

            # ---------- ARIMA ----------
            if "ARIMA" in models:
                arima_fcst = generate_arima_forecast(forecast_data, forecast_years, order=(1, 1, 1))
                if arima_fcst is not None:
                    fig.add_trace(go.Scatter(
                        x=arima_fcst.index.year,
                        y=arima_fcst.values,
                        name='ARIMA Forecast',
                        line=dict(dash='dash', color='#2ca02c')
                    ))

            # ---------- Linear Trend ----------
            if "Linear Trend" in models:
                linear_fcst = generate_linear_forecast(forecast_data, forecast_years)
                if linear_fcst is not None:
                    future_years = linear_fcst.index
                    x_vals = np.concatenate(([last_year], future_years))
                    y_vals = np.concatenate(([last_val], linear_fcst.values))

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        name='Linear Trend',
                        mode='lines+markers',
                        line=dict(color='#d62728', dash='dashdot')
                    ))

            # ---------- Layout ----------
            fig.update_layout(
                title=f"{forecast_years}-Year Funding Forecast ({scope})",
                xaxis_title="Year",
                yaxis_title="Funding (€)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for forecasting. Please broaden your filters.")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    **Data Source:** [CORDIS EU Research Projects](https://data.europa.eu/data/datasets/cordisref-data)  
    **Last Updated:** {datetime.now().strftime("%Y-%m-%d")}  
    """)

if __name__ == "__main__":
    main()
