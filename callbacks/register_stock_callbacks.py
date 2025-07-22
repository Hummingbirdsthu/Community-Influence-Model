from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from datetime import date
import pytz
import base64
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from itertools import combinations
import requests
from sklearn.preprocessing import StandardScaler
import dash_table
import yfinance as yf


import src.tab1 as tab1
import src.tab2 as tab2
import src.tab3 as tab3


# Đọc dữ liệu
df = pd.read_csv('data/filtered_lambda_output.csv')
df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])

df_lambda = df.nlargest(70, 'Market Cap')

def create_stock_graph(df, selected_sectors=None, threshold=0.8):
    G = nx.Graph()

    filtered_df = df[df['Sector'].isin(selected_sectors)] if selected_sectors else df
    features = ['Volume', 'Market Cap', 'Change', 'Beta (1Y)']
    filtered_df[features] = StandardScaler().fit_transform(filtered_df[features])
    filtered_df = filtered_df.dropna(subset=['Volume', 'Market Cap', 'Change'])

    for _, row in filtered_df.iterrows():
        symbol = row['Id']
        G.add_node(symbol,
                   Lambda=row['Lambda'],
                   sector=row['Sector'],
                    Beta=row['Beta (1Y)'],
                    community=row['Cluster'],
                   name=row['Name'])
    symbols = filtered_df['Id'].tolist()

    for u, v in combinations(symbols, 2):
        u_vector = filtered_df[filtered_df['Id'] == u][['Volume', 'Market Cap', 'Change','Beta (1Y)']].values
        v_vector = filtered_df[filtered_df['Id'] == v][['Volume', 'Market Cap', 'Change','Beta (1Y)']].values
        if u_vector.size > 0 and v_vector.size > 0:
            weight = cosine_similarity(u_vector, v_vector)[0][0]# + 1) / 2
            if df[df['Id']==u]['Sector'].values[0] == df[df['Id']==v]['Sector'].values[0] and weight >= threshold/2:  
                G.add_edge(u, v, weight=weight)
            elif weight >= threshold:
                G.add_edge(u, v, weight=weight)

    return G
def top_lambda_nodes_by_sector(df, selected_sectors=None, total_top=50):
    if selected_sectors:
        df = df[df['Sector'].isin(selected_sectors)]

    sectors = df['Sector'].dropna().unique()
    k = len(sectors)
    per_sector = max(1, total_top // k)

    top_df_list = []
    for sector in sectors:
        top_sector_df = df[df['Sector'] == sector].nlargest(per_sector, 'Lambda')
        top_df_list.append(top_sector_df)

    final_df = pd.concat(top_df_list).sort_values('Lambda', ascending=False).head(total_top)
    return final_df
def clustered_random_layout(G, cluster_attr='sector', cluster_radius=1.0, seed=42):
    np.random.seed(seed)
    clusters = list(set(nx.get_node_attributes(G, cluster_attr).values()))
    
    cluster_centers = {
        cluster: (np.random.uniform(-3, 3), np.random.uniform(-3, 3))
        for cluster in clusters
    }

    pos = {}
    for node in G.nodes():
        cluster = G.nodes[node].get(cluster_attr, 'Unknown')
        cx, cy = cluster_centers[cluster]
        dx = np.random.normal(0, cluster_radius)
        dy = np.random.normal(0, cluster_radius)
        pos[node] = (cx + dx, cy + dy)
    return pos
def create_node_edge(G, sector_colors=None):
    pos = clustered_random_layout(G, cluster_attr='sector', cluster_radius=0.8)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='rgba(150,150,150,0.8)'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_colors, node_size = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        lambda_value = G.nodes[node].get('Lambda', 0.0)
        sector = G.nodes[node].get('sector', 'Unknown')
        name = G.nodes[node].get('name', '')
        symbol = node

        node_colors.append(sector_colors[sector])
        node_size.append(10 + 60 * lambda_value)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_size,
            line=dict(width=1.5, color='black')
        ),
        textfont=dict(size=9, color='black'),
        hoverinfo='text',
        showlegend=False
    )
    
    return edge_trace, node_trace

def change_marketcap(val):
    if val > 1e12:
        return f"${val / 1e12:.2f}T"
    elif val > 1e9:
        return f"${val / 1e9:.2f}B"
    elif val > 1e6:
        return f"${val / 1e6:.2f}M"
    else:
        return f"${val:.2f}"
def change_icon(val):
    if val > 0:
        return f"▲ {val}"
    elif val < 0:
        return f"▼ {val}"
    else:
        return f"▬ {val}"

def image_to_base64(url):
    try:
        response = requests.get(url)
        return "data:image/jpeg;base64," + base64.b64encode(response.content).decode('utf-8')
    except:
        return "data:image/jpeg;base64," + base64.b64encode(requests.get("https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228").content).decode('utf-8')


# 1. Generate summary cards
def generate_stats_card (title, subtitle, value, image_path):
    return html.Div(
        dbc.Card([
            dbc.CardImg(src=image_path, top=True, style={'width': '50px','alignSelf': 'center'}),
            dbc.CardBody([
                html.P(value, className="card-value", style={'margin': '0px','fontSize': '39px','fontWeight': 'bold'}),
                html.H4(title, className="card-title", style={'margin': '0px','fontSize': '18px','fontWeight': 'bold'}),
                html.P(subtitle, className="card-subtitle", style={'margin': '0px', 'fontSize': '14px', 'color': '#333'})
            ], style={'textAlign': 'center'}),
        ], style={'width': '100%', 'height': '100%', 'margin': '0px', "backgroundColor":'#f8f9fa','border':'none','borderRadius':'10px', 
                  'textAlign': 'center', 'paddingTop': '20px', 'paddingBottom': '20px'})
    )


# 2. Create horizontal bar chart
def create_horizontal_bar_chart(df):
        sector_counts = df['Sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Count']
        sector_counts = sector_counts.sort_values(by='Count', ascending=False)

        fig = px.bar(
            sector_counts,
            x='Count',
            y='Sector',
            orientation='h',
            color='Sector',
            title='Number of Stocks by Sector',
            labels={'Count': 'Number of Stocks', 'Sector': 'Sector'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            title=dict(text="Number of Stocks by Sector", x=0.5, font=dict(size=16, color="#FFFFFF")),
            xaxis=dict(title='Number of Stocks'),
            yaxis=dict(title='Sector'),
            height=400,
            margin=dict(l=50, r=50, t=30, b=80),
            # margin=dict(l=50, r=50, t=30, b=80),

        )
        return fig

# 3. Create Sankey diagram
def create_sankey_diagram_blue(df):
    sector_community_counts = df.groupby(['Sector', 'Cluster']).size().reset_index(name='Count')
    sector_community_counts = sector_community_counts.sort_values(by=['Cluster', 'Count'], ascending=False)

    # Create unique labels for nodes
    unique_sectors = sector_community_counts['Sector'].unique().tolist()
    unique_clusters = sector_community_counts['Cluster'].unique().tolist()
    all_labels = unique_sectors + unique_clusters

    # Create source and target indices
    source_indices = sector_community_counts['Sector'].apply(lambda x: unique_sectors.index(x))
    target_indices = sector_community_counts['Cluster'].apply(lambda x: len(unique_sectors) + unique_clusters.index(x))

    fig = px.sankey(
        sector_community_counts,
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=px.colors.sequential.Blues_r * 2 # Use Blues_r color scale
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sector_community_counts['Count']
        )
    )
    fig.update_layout(
        title_text="Sankey Diagram of Sector to Community",
        font=dict(family="Montserrat", size=12, color="#000000"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=50, r=50, t=30, b=80)
    )
    return fig



G = create_stock_graph(df_lambda)



def register_stock_callbacks(app):

    # I. Summary cards
    @app.callback(
        [Output('total-stock-card', 'children'),
        Output('total-sector-card', 'children'),
        Output('total-community-card', 'children'),
        Output('max-change-card', 'children')],
        [Input('sector-filter', 'value'),
        Input('community-filter', 'value')]
    )
    def update_summary_cards(selected_sector, selected_community):
        dff = df.copy()
        if selected_sector:
            dff = dff[dff['Sector'] == selected_sector]
        # elif not selected_sector:
        #     dff = dff[dff['Sector'].isin(dff['Sector'].unique())]
        if selected_community is not None:
            dff = dff[dff['Cluster'] == selected_community]

        total_stock = len(dff)
        total_sector = dff['Sector'].nunique()
        total_community = dff['Cluster'].nunique()
        # get string name community: Community 1, Community 2, Community 1&2
        community_names = []
        if total_community is not None:
            community_names = [f"Community {i + 1}" for i in range(total_community)]
        community_string = ", ".join(community_names) if community_names else "N/A"

        # Tính max_change
        
        max_change = dff['Change'].max() if not dff.empty else None
        min_change = dff['Change'].min() if not dff.empty else None
        if abs(max_change) < abs(min_change):
            max_change = min_change
        change_sign = '▲' if max_change >= 0 else '▼'
        max_change_stock = dff.loc[dff['Change'].idxmax(), 'Id'] if not dff.empty else ''
        formatted_change = html.Span([
            change_sign + f" {abs(max_change):.2f}",
            html.Span(f" ({max_change_stock})", style={'color': '#555', 'fontSize': '13px'})
        ])

        return (
            generate_stats_card(
                title="Total Stock",
                subtitle=".",
                value=total_stock,
                image_path="/assets/img/num_stocks.png"
            ),
            generate_stats_card(
                title="Total Sector",
                subtitle=".",
                value=total_sector,
                image_path="/assets/img/num_sectors.png"
            ),
            generate_stats_card(
                title="Total Community",
                subtitle=community_string,
                value=total_community,
                image_path="/assets/img/num_community.png"
            ),
            generate_stats_card(
                title="Max Change",
                subtitle=formatted_change,
                value=f"$ {max_change:.2f}" if max_change is not None else "N/A",
                image_path="/assets/img/max_change.png"
            ),
        )

    # II. Tabs và nội dung của từng tab
    @app.callback(
        Output('tab-content', 'children'),
        Input('graph-tabs', 'value')
    )
    def display_tab_content(selected_tab):
        if selected_tab == 'network_visualize':
            return tab1.tab1_layout
        
        elif selected_tab == 'community_analysis':
            return tab2.tab2_layout
            
        elif selected_tab == 'stock_analysis':
            return tab3.tab3_layout
        
        return html.Div("Tab content not found.")
    
    

#---------------------------------------PHẦN NÀY LÀ CALLBACK------------------------------------------------#

# ====================================TAB 1======================================

    # III. Callback Tabs
    # 1. Callback Tab 1; Network Visualize
    # Callback cho 'network-graph' - Stock Network Visualization
    @app.callback(
        Output('network-graph', 'children'),
        [Input('sector-filter', 'value'),
         Input('community-filter', 'value')]
    )

    def update_network_graph(selected_sectors, selected_cluster):
        G = create_stock_graph(df_lambda, selected_sectors=selected_sectors)
        if selected_cluster is not None:
            nodes_to_keep = [node for node, data in G.nodes(data=True) if data.get('community') == selected_cluster]
            G.remove_nodes_from([node for node in G.nodes() if node not in nodes_to_keep])
        
        sector_partition = {node: data.get('sector', 'Unknown') for node, data in G.nodes(data=True)}
        sector_colors = {
            sector: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            for i, sector in enumerate(sorted(set(sector_partition.values())))
        }
        edge_trace, node_trace = create_node_edge(G, sector_colors=sector_colors)
        
        node_text = [f"Stock: {node}<br>Community: {G.nodes[node]['community']}<br>Sector: {G.nodes[node]['sector']}<br>Name: {G.nodes[node]['name']}<br>Lambda: {G.nodes[node]['Lambda']:.2f}" for node in G.nodes()]
        node_trace.hoverinfo = 'text'
        node_trace.hovertext = node_text
        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        fig.update_layout(
            title=dict(
                text="Stocks Network",
                x=0.,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            #width=500,
            margin=dict(l=0, r=20, t=50, b=0),
            font=dict(size=13, color='#000')
        )

        for sector in sorted(set(sector_partition.values())):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=5, color=sector_colors[sector]),
                name=f"{sector}",
                hoverinfo='none',
                showlegend=True
            ))

        return dcc.Graph(figure=fig)


    # Callback cho các biểu đồ bổ sung
    @app.callback(
        [Output('lambda-distribution-chart', 'children'),
         Output('sunburst-sector-cluster', 'figure')],
        [Input('sector-filter', 'value'),
         Input('community-filter', 'value')]
    )

    def update_additional_charts(selected_sectors, selected_community):
        # Lọc dữ liệu theo Sector nếu có
        filtered_df = df.copy()
        if selected_community:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_community)]
        if selected_sectors:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
        # Histogram phân phối Lambda
        lambda_hist_fig = go.Figure()
        lambda_hist_fig.add_trace(go.Histogram(
            x=filtered_df['Lambda'],
            nbinsx=10,  # Số bin cố định, có thể điều chỉnh
            marker=dict(
                color="#060189", 
                line=dict(width=1, color='white')
            ),
            opacity=0.7,
            hoverinfo='x+y'
        ))

        lambda_hist_fig.update_layout(
            title={
                'text': "Distribution of Lambda Values",
                'y': 0.95,
                'x': 0.,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': {
                    'size': 16,
                    'color': 'black',
                    'family': "Montserrat"
                }
            },
            xaxis=dict(
                title="<b>Lambda Value</b>",
                title_font=dict(size=12, color='black'),
                tickfont=dict(size=11, color='black'),
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis=dict(
                title="<b>Frequency</b>",
                title_font=dict(size=12, color='black'),
                tickfont=dict(size=11, color='black')
            ),
            margin=dict(l=40, r=40, t=50, b=20),
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(
                bgcolor='rgba(29,185,84,0.8)',
                font_size=13,
                font_family="Montserrat"
            ),
            showlegend=False
        )

        # Vẽ biểu đồ Sunburst
        sector_cluster_summary = filtered_df.groupby(['Sector', 'Cluster'], as_index=False).size()
        fig_sunburst = px.sunburst(sector_cluster_summary, path=['Sector', 'Cluster'], values='size',
                                title='Sector and Community Distribution',
                                color_discrete_sequence=px.colors.sequential.Blues[-1::-1])
        
        fig_sunburst.update_traces(textinfo='label+percent entry')
        fig_sunburst.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', color="#000"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )

        return (
            dcc.Graph(figure=lambda_hist_fig, style={'height': '400px'}),
            fig_sunburst
        )

    # Callback cho các biểu đồ bổ sung
    @app.callback(
        [Output('network-table', 'data'),
         Output('network-table', 'columns'),
         Output('network-table', 'style_data_conditional')],
        [Input('sector-filter', 'value'),
         Input('community-filter', 'value')]
    )

    def update_network_table(selected_sector, selected_community):
        # Lọc dữ liệu theo Sector và Community nếu có
        filtered_df = df.copy()
        if selected_sector:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sector)]# if selected_sector else df
            #filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        if selected_community is not None:
            filtered_df = filtered_df[filtered_df['Cluster'] == selected_community]

        # Chọn các cột cần hiển thị
        cols = ['Id', 'Name', 'Sector', 'Close', 'Change', 'Volume', 'Market Cap', 'EPS', 'Lambda', 'Beta (1Y)']
        dff_table = filtered_df.sort_values('EPS', ascending=False).copy()
        dff_table = dff_table[cols]

        # Tạo cột MarketCap% để tô nền
        max_cap = dff_table['Market Cap'].max()
        dff_table['MarketCap%'] = (dff_table['Market Cap'] / max_cap * 100).round(1)

        # Định nghĩa các cột cho DataTable
        dff_table['Market Cap'] = dff_table['Market Cap'].apply(change_marketcap)

        def round_2(val):
            return round(val, 2)
        dff_table['EPS'] = dff_table['EPS'].apply(round_2)
        dff_table['Lambda'] = dff_table['Lambda'].apply(round_2)
        dff_table['Beta (1Y)'] = dff_table['Beta (1Y)'].apply(round_2)
        dff_table['Change'] = dff_table['Change'].apply(round_2)

        # Thêm icon cho cột Change
        dff_table['Change'] = dff_table['Change'].apply(change_icon)

        columns = [
            {'name': 'Name', 'id': 'Id'},
            {'name': 'Sector', 'id': 'Sector'},
            {'name': 'Change', 'id': 'Change', 'presentation': 'markdown'},
            {'name': 'Lambda', 'id': 'Lambda'},
            {'name': 'Close', 'id': 'Close'},
            {'name': 'Volume', 'id': 'Volume'},
            {'name': 'Market Cap', 'id': 'Market Cap'},
            {'name': 'EPS', 'id': 'EPS'},
            {'name': 'Beta (1Y)', 'id': 'Beta (1Y)'}
        ]
        style_data_conditional = [
            {
                'if': {'column_id': 'Change'},
                'textAlign': 'center'
            },
            # Change > 0: xanh lá
            {
                'if': {'filter_query': '{Change} contains "▲"', 'column_id': 'Change'},
                'color': 'green',
                'fontWeight': 'bold'
            },
            # Change < 0: đỏ
            {
                'if': {'filter_query': '{Change} contains "▼"', 'column_id': 'Change'},
                'color': 'red',
                'fontWeight': 'bold'
            },
            # Change == 0: xám
            {
                'if': {'filter_query': '{Change} contains "▬"', 'column_id': 'Change'},
                'color': 'gray',
                'fontWeight': 'bold'
            },
            
            # Market Cap: tô nền xanh dương theo %
            {
                'if': {'column_id': 'Market Cap'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {MarketCap%}%, transparent {MarketCap%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            # Tô nền cho các cột khác nếu cần
            {
                'if': {'column_id': 'Volume'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Volume%}%, transparent {Volume%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'EPS'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {EPS%}%, transparent {EPS%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Lambda'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Lambda%}%, transparent {Lambda%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Beta (1Y)'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Beta (1Y)%}%, transparent {Beta (1Y)%}%, transparent 100%)"
                ), 
                'color': '#111'
            }, 
            {
                'if': {'column_id': 'Symbol'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Id%}%, transparent {Id%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Sector'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Sector%}%, transparent {Sector%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Close'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Close%}%, transparent {Close%}%, transparent 100%)"
                ),
                'color': '#111'
            },
        ]
        return (
            dff_table.to_dict('records'),
            columns,
            style_data_conditional
        )



# ====================================TAB 2======================================


    # 2. Callback Tab 2: Community Analysis
    @app.callback(
        Output('stacked-bar-chart', 'figure'),
        Output('sankey-diagram', 'figure'),
        Output('tree-map', 'figure'),
        Output('bubble-chart', 'figure'), 
        Input('sector-filter', 'value'),
        Input('community-filter', 'value')
    )
    def update_tab1_charts(selected_sector, selected_community):
        filtered_df = df.copy()

        if selected_sector:
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

        if selected_community:
            filtered_df = filtered_df[filtered_df['Cluster'] == selected_community]

        # 2.1 Create Stacked Horizontal Bar Chart
        # Count stocks per sector and cluster
        stacked_bar_data = filtered_df.groupby(['Sector', 'Cluster']).size().reset_index(name='count')
        # Pivot for stacked bar chart format
        stacked_bar_pivot = stacked_bar_data.pivot(index='Sector', columns='Cluster', values='count').fillna(0)
        stacked_bar_fig = px.bar(
            stacked_bar_pivot,
            x=stacked_bar_pivot.columns, # Clusters as columns
            y=stacked_bar_pivot.index, # Sectors as index
            orientation='h',
            title='Stock Distribution by Sector and Cluster',
            labels={'value': 'Number of Stocks', 'variable': 'Cluster'},
            template='simple_white',
            color='Cluster',
            color_discrete_sequence=["#154360", "#3B6E8D"]
        )
        stacked_bar_fig.update_layout(
            barmode='stack',
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=500,
            showlegend=True,
            margin=dict(t=30, b=30, l=40, r=40),
        )

        # 2.2 Create Sankey Diagram
        # Prepare data for Sankey: source, target, value
        sankey_data = filtered_df.groupby(['Sector', 'Cluster']).size().reset_index(name='count')

        # Create labels and mappings for Sankey diagram
        all_nodes = list(sankey_data['Sector'].unique()) + list(sankey_data['Cluster'].unique())
        node_map = {node: i for i, node in enumerate(all_nodes)}

        source_indices = [node_map[sector] for sector in sankey_data['Sector']]
        target_indices = [node_map[cluster] for cluster in sankey_data['Cluster']]
        values = sankey_data['count'].tolist()

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])

        sankey_fig.update_layout(
            title_text="Sector to Community",
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=500,
            showlegend=True,
            margin=dict(t=30, b=30, l=40, r=40),
        )


        # 2.3 Create Treemap
        treemap_fig = px.treemap(
            filtered_df,
            path=['Sector','Symbol'],
            values='Market Cap',
            color='Lambda',
            color_continuous_scale=["#154360", "#3B6E8D"],
            title='Treemap of Stock Performance'
        )
        treemap_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=600,
            showlegend=True,
            margin=dict(t=30, b=30, l=40, r=40),
        )

        # 2.4 Create Bubble chart
        bubble_fig = px.scatter(
            filtered_df,
            x='Market Cap',
            y='Lambda',
            size='Volume',
            color='Sector',
            hover_name='Id',
            title='Bubble Chart of Stock Performance'
        )
        bubble_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=600,
            showlegend=True,
            margin=dict(t=30, b=0, l=40, r=40),
        )



        # Return all figures
        return stacked_bar_fig, sankey_fig, treemap_fig, bubble_fig
    
    
    

# ====================================TAB 3======================================








    # 3. Callback Tab 3: Stock Analysis
    # Callback cho Candlestick Chart
    @app.callback(
        [Output('candlestick-chart', 'children'),
         Output('volume-by-month-chart', 'children')],
        Input('stock-filter', 'value')
    )
    def update_candlestick_chart(selected_stock):
        end_date = date.today().strftime('%Y-%m-%d')
        data = yf.download(selected_stock, start='2025-01-01', end='2025-07-01', multi_level_index = False)
        
        if data.empty:
            return html.Div(f"No data available for {selected_stock} from '2025-01-01' to {end_date}.", 
                            style={'color': 'white', 'font-family': 'Montserrat', 'textAlign': 'center'})

        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                increasing_line_color='#27ae60',  # Green for increasing
                decreasing_line_color='#e74c3c'   # Red for decreasing
            ),
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='#3498db', width=1)),  # SMA màu xanh dương
            go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='#e67e22', width=1))   # EMA màu cam
        ])
        
        special_date = pd.to_datetime('2025-05-26')
        fig.add_vline(x=special_date, line=dict(color="blue", width=1, dash="dash"))

        fig.update_layout(
            title=dict(
                text=f"Candlestick Chart for {selected_stock}",
                x=0,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', color="#000"),
            xaxis_rangeslider_visible=False,
            #height=350,
            margin=dict(l=20, r=20, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, )
        )
        
        # Tạo biểu đồ sum of volume by month
        data['Month'] = data.index.to_period('M').astype(str) 
        volume_by_month = data.groupby('Month')['Volume'].sum().reset_index()
        volume_by_month = volume_by_month.sort_values(by='Volume', ascending=False)

        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            y=volume_by_month['Month'],
            x=volume_by_month['Volume'],
            orientation='h',
            marker_color='#3498db',  # Màu xanh dương
            hovertemplate='Month: %{y}<br>Total Volume: %{x}<extra></extra>'
        ))

        fig_volume.update_layout(
            title=dict(
                text="Sum of Volume by Month",
                x=0,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            xaxis=dict(
                title="<b>Total Volume</b>",
                title_font=dict(size=12, color='black'),
                tickfont=dict(size=11, color='black')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        return (
            dcc.Graph(figure=fig, style={'height': '100%'}),
            dcc.Graph(figure=fig_volume, style={'height': '500px'}))

    # Callback cho increasing chart
    @app.callback(
        Output('trend-increasing-chart', 'children'),
        Input('refresh', 'n_intervals')
    )
    
    def update_trend_increasing_chart(n):
        df = pd.read_csv('data/increasing_stocks.csv')
        trend_increase_count = len(df[df['Trend'] == 'Tăng'])
        percent_trend_increase = (trend_increase_count / len(df)) * 100

        cluster_1_rows = len(df[df['Cluster'] == 1])
        cluster_1_trend_increase_count = len(df[(df['Cluster'] == 1) & (df['Trend'] == 'Tăng')])
        percent_cluster_1_trend_increase = (cluster_1_trend_increase_count / cluster_1_rows) * 100 if cluster_1_rows > 0 else 0

        # Data for the bar chart
        labels = ['Before', 'After']
        values = [percent_trend_increase, percent_cluster_1_trend_increase]
        #gradient_colors_before = ['#87CEEB', '#1E90FF', '#000080']
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=["#42F1F7", "#42F1F7"],
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
                width=0.5
            )
        ])

        # Add horizontal line at the top of the "Before" bar
        fig.add_shape(
            type="line",
            x0=-0.5,  # Start at the left edge of the chart (before "Before")
            x1=1.5,   # End at the right edge of the chart (after "After")
            y0=percent_trend_increase,  # Height of "Before" bar
            y1=percent_trend_increase,  # Same height for a horizontal line
            line=dict(
                color="gray",
                width=2,
                dash="dash"  # Dashed line
            ),
            xref="x",  # Use x-axis domain
            yref="y"   # Use y-axis domain
        )
        # Update layout to match the desired style
        fig.update_layout(
            title=dict(
                text=f"Comparison of Trend Increase Percentages",
                x=0,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, max(values) * 1.2], tickformat=".1f", showgrid=False),  # Remove horizontal grid lines
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
            hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font=dict(color="white"))
            
        )

        return dcc.Graph(figure=fig, id='trend-increasing-chart')

    @app.callback(
        Output('pie-chart-sector', 'children'),
        Input('community-filter', 'value')
    )
    def update_pie_portfolio(selected_community):
        df = pd.read_csv('data/increasing_stocks.csv')
        if selected_community is not None:
            df_filtered = df[df['Cluster'] == selected_community]
        else:
            df_filtered = df
        sector_counts = df_filtered['Sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Stock Quantity']
        
        pie_fig = px.pie(
            sector_counts,
            names='Sector',
            values='Stock Quantity',
            height=400,
            color_discrete_sequence=px.colors.sequential.Blues[-1::-1]  # ✅ dùng màu gradient xanh dương nhạt
        )

        pie_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            margin=dict(l=0, r=150, t=50, b=0),
            title=dict(
                text="Stock Distribution by Sector",
                x=0.,
                xanchor='left',
                yanchor='top',
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            legend=dict(title='Sector', yanchor="top", y=0.5, xanchor="right", x=1.5, font=dict(color='black', size=13))
        )
        pie_fig.update_traces(
            marker=dict(line=dict(color='#FFFFFF', width=1))  # Optional: Add border for clarity
        )

        return dcc.Graph(figure=pie_fig)



    # Callback cho Top Stocks Strategy
    @app.callback(
        [Output('top-stocks-table', 'children'), 
         Output('top10-strategy-table', 'data'),
         Output('top10-strategy-table', 'columns'),
         Output('top10-strategy-table', 'style_data_conditional')],
        [Input('sector-filter', 'value'),
         Input('community-filter', 'value')]
    )
    def update_top_stocks(selected_sector, selected_community):
        filtered_df = pd.read_csv('data/increasing_stocks.csv')  # Assuming this CSV contains the top stocks data
        if selected_sector:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sector)]
        if selected_community is not None:
            filtered_df = filtered_df[filtered_df['Cluster'] == selected_community]
            stocks_df = filtered_df[(filtered_df['Cluster'] == selected_community) & 
                        (filtered_df['Trend'] == 'Tăng')].sort_values(by='Volume', ascending=False).head(5)
        else:
            stocks_df = filtered_df[(filtered_df['Cluster'] == 1) & 
                        (filtered_df['Trend'] == 'Tăng')].sort_values(by='Volume', ascending=False).head(5)
        
        

        symbols = stocks_df['Symbol'].tolist()
        closes = stocks_df['Close'].tolist()
        volumes = stocks_df['Volume'].tolist()
        market_caps = stocks_df['Market Cap'].tolist()
        lambdas = stocks_df['Lambda'].tolist()
        betas = stocks_df['Beta (1Y)'].tolist()
        image_urls = stocks_df['image_url'].tolist()
        base64_images = [image_to_base64(url) for url in image_urls]
        changes = stocks_df['Change'].tolist()
        names = stocks_df['Name'].tolist()  # Assuming 'Name' column exists in df

        # Create a figure with logos, symbols, names, and changes
        fig = go.Figure()

        fig.update_layout(
            title=dict(
                text=f"Top 5 Stocks in Portfolio",
                x=0,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', color="#000"),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False,
            xaxis=dict(visible=False),  # Hide x-axis
            yaxis=dict(visible=False)   # Hide y-axis
        )

        for i, (symbol, name, change, close, volume, market_cap, lam, beta) in enumerate(zip(symbols, names, changes, closes, volumes, market_caps, lambdas, betas)):
            y_pos = 0.85 - i * 0.16  # Vertical spacing for 5 items
            trend_icon = '▲' if change > 0 else '▼'
            trend_color = '#27ae60' if change > 0 else '#e74c3c'
            wrapped_name = name if len(name) <= 25 else name[:22] + '...'  # Wrap name if too long

            # Add logo image
            fig.add_layout_image(
                dict(
                    source=base64_images[i],
                    xref="paper", yref="paper",
                    x=0.1, y=y_pos,
                    sizex=0.15, sizey=0.15,
                    xanchor="center", yanchor="middle",
                    layer="above"
                )
            )

            # Add symbol annotation
            fig.add_annotation(
                x=0.2, y=y_pos,  # Moved left of center
                xref="paper", yref="paper",
                text=f"<b>{symbol}</b>",
                showarrow=False,
                font=dict(size=14, color="black", family="Montserrat"), 
                xanchor="left", yanchor="middle",
                hovertext=f"Lambda: {lam:.2f}<br>Beta: {beta:.2f}<br>Close: ${close:.2f}<br>Volume: {volume}<br>Market Cap: ${market_cap}",
                hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font=dict(color="white"))
            )

            # Add name annotation below symbol
            fig.add_annotation(
                x=0.2, y=y_pos - 0.05,  # Slightly below symbol
                xref="paper", yref="paper",
                text=wrapped_name,
                showarrow=False,
                font=dict(size=10, color="gray", family="Montserrat"),
                xanchor="left", yanchor="middle"
            )

            # Add change with trend icon on the right
            fig.add_annotation(
                x=0.85, y=y_pos,  # Moved to the right side
                xref="paper", yref="paper",
                text=f"{trend_icon} {change:.2f}%",
                showarrow=False,
                font=dict(size=14, color=trend_color, family="Montserrat"),
                xanchor="center", yanchor="middle"
            )

        # DataTable
        cols = ['Symbol', 'Name', 'Sector', 'Cluster', 'Close', 'Change', 'Volume', 'Market Cap', 'EPS', 'Lambda', 'Beta (1Y)', 'Trend']
        dff_table = filtered_df[cols]
        # dff_table = dff_table[(dff_table['Cluster'] == 1) & 
        #                 (dff_table['Trend'] == 'Tăng')].sort_values(by='Volume', ascending=False)

        # Tạo cột MarketCap% để tô nền
        max_cap = dff_table['Market Cap'].max()
        dff_table['MarketCap%'] = (dff_table['Market Cap'] / max_cap * 100).round(1)

        # Định nghĩa các cột cho DataTable
        dff_table['Market Cap'] = dff_table['Market Cap'].apply(change_marketcap)

        def round_2(val):
            return round(val, 2)
        dff_table['EPS'] = dff_table['EPS'].apply(round_2)
        dff_table['Lambda'] = dff_table['Lambda'].apply(round_2)
        dff_table['Beta (1Y)'] = dff_table['Beta (1Y)'].apply(round_2)
        dff_table['Change'] = dff_table['Change'].apply(round_2)

        # Thêm icon cho cột Change
        dff_table['Change'] = dff_table['Change'].apply(change_icon)


        columns_3 = [
            {'name': 'Name', 'id': 'Symbol'},
            {'name': 'Sector', 'id': 'Sector'},
            {'name': 'Change', 'id': 'Change', 'presentation': 'markdown'},
            {'name': 'Lambda', 'id': 'Lambda'},
            {'name': 'Community', 'id': 'Cluster'},
            {'name': 'Close', 'id': 'Close'},
            {'name': 'Volume', 'id': 'Volume'},
            {'name': 'Market Cap', 'id': 'Market Cap'},
            {'name': 'EPS', 'id': 'EPS'},
            {'name': 'Beta (1Y)', 'id': 'Beta (1Y)'},
            {'name': 'Trend', 'id': 'Trend'}
        ]

        dash_table.DataTable(
            id='top10-strategy-table',
            style_table={'overflowX': 'auto', 'borderRadius': '10px', 'backgroundColor': '#222b3a', 'border': '20px solid #222b3a'},
            style_cell={
                'fontFamily': 'Montserrat',
                'textAlign': 'center',
                'padding': '5px',
                'backgroundColor': '#222b3a',   # background trắng cho cell
                'color': '#222b3a'              # chữ đen cho cell
            },
            style_header={
                'backgroundColor': '#222b3a',   # header xanh
                'color': 'white',            # chữ trắng
                'fontWeight': 'bold'
            },
            style_data_conditional=[]  # giữ nguyên phần này để callback cập nhật
        )

        style_data_conditional_3 = [
            {
                'if': {'column_id': 'Change'},
                'textAlign': 'center'
            },
            # Change > 0: xanh lá
            {
                'if': {'filter_query': '{Change} contains "▲"', 'column_id': 'Change'},
                'color': 'green',
                'fontWeight': 'bold'
            },
            # Change < 0: đỏ
            {
                'if': {'filter_query': '{Change} contains "▼"', 'column_id': 'Change'},
                'color': 'red',
                'fontWeight': 'bold'
            },
            # Change == 0: xám
            {
                'if': {'filter_query': '{Change} contains "▬"', 'column_id': 'Change'},
                'color': 'gray',
                'fontWeight': 'bold'
            },
            
            # Market Cap: tô nền xanh dương theo %
            {
                'if': {'column_id': 'Market Cap'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {MarketCap%}%, transparent {MarketCap%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            # Tô nền cho các cột khác nếu cần
            {
                'if': {'column_id': 'Volume'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Volume%}%, transparent {Volume%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'EPS'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {EPS%}%, transparent {EPS%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Lambda'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Lambda%}%, transparent {Lambda%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Beta (1Y)'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Beta (1Y)%}%, transparent {Beta (1Y)%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Symbol'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Id%}%, transparent {Id%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Sector'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Sector%}%, transparent {Sector%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Close'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Close%}%, transparent {Close%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Trend'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Trend%}%, transparent {Trend%}%, transparent 100%)"
                ),
                'color': '#111'
            }
        ]


        return (
            dcc.Graph(figure=fig, style={'backgroundColor': "#fcfdff"}),
            dff_table.to_dict('records'), columns_3, style_data_conditional_3
        )



