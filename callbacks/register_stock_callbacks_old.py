from pyexpat import features
from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
import community.community_louvain as community_louvain
from datetime import datetime
import pytz
import base64
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
df = pd.read_csv('filtered_lambda_output.csv')
df_lambda = df.nlargest(150, 'Market Cap')

# Màu sắc
COLOR_SCHEME = {
    "lightblue": "#00f2ff",
    "darkblue": "#567396",
    "white": "#FFFFFF",
    "highlight": "#00CED1",
    "background": "rgba(0,0,0,0.7)"
}

# Tạo graph
def create_stock_graph(df, selected_sectors=None, threshold=0.8):
    G = nx.Graph()

    filtered_df = df[df['Sector'].isin(selected_sectors)] if selected_sectors else df
    features = ['Volume', 'Market Cap', 'Change', 'Beta (1Y)']
    filtered_df[features] = StandardScaler().fit_transform(filtered_df[features])
    filtered_df = filtered_df.dropna(subset=['Volume', 'Market Cap', 'Change'])

    for _, row in filtered_df.iterrows():
        symbol = row['Symbol']
        G.add_node(symbol,
                   Lambda=row['Lambda'],
                   sector=row['Sector'],
                   name=row['Name'])
    symbols = filtered_df['Symbol'].tolist()

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            u = symbols[i]
            v = symbols[j]

            u_vector = filtered_df[filtered_df['Symbol'] == u][['Volume', 'Market Cap', 'Change','Beta (1Y)']].values
            v_vector = filtered_df[filtered_df['Symbol'] == v][['Volume', 'Market Cap', 'Change','Beta (1Y)']].values
            if u_vector.size > 0 and v_vector.size > 0:
                weight = (cosine_similarity(u_vector, v_vector)[0][0] + 1) / 2
                print(weight)
                if weight >= threshold:  
                    G.add_edge(u, v, weight=weight)

    return G
def top_lambda_nodes_by_sector(df, selected_sectors=None, total_top=50):
    if total_top is None:
        total_top = 50
    df = pd.read_csv('filtered_lambda_output.csv')
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
    
    # Gán mỗi ngành một tọa độ trung tâm ngẫu nhiên
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
def register_stock_callbacks(app):
    # Callback cho 'network-graph' - Stock Network Visualization
    @app.callback(
        Output('network-graph', 'children'),
        [Input('sector-filter', 'value'),
         Input('relationship-filter', 'value'),
         Input('refresh', 'n_intervals')]
    )


    def update_network_graph(selected_sectors, selected_cluster, n):
        filtered_df = top_lambda_nodes_by_sector(df_lambda, selected_sectors=selected_sectors, total_top=50)
        G = create_stock_graph(filtered_df)
        if selected_cluster is not None:
            nodes_to_keep = [node for node, data in G.nodes(data=True) if data.get('community') == selected_cluster]
            G.remove_nodes_from([node for node in G.nodes() if node not in nodes_to_keep])
        if n is not None and n > 0 and G.number_of_edges() > n:
            all_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            edges_to_keep = set((u, v) for u, v, _ in all_edges[:n])
            G.remove_edges_from([e for e in G.edges() if (e[0], e[1]) not in edges_to_keep and (e[1], e[0]) not in edges_to_keep])

        sector_partition = {node: data.get('sector', 'Unknown') for node, data in G.nodes(data=True)}
        sector_colors = {
            sector: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
            for i, sector in enumerate(sorted(set(sector_partition.values())))
        }
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

        # Node trace
        node_x, node_y, node_text, node_colors, node_size = [], [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            lambda_value = G.nodes[node].get('Lambda', 0.0)
            sector = G.nodes[node].get('sector', 'Unknown')
            name = G.nodes[node].get('name', '')
            symbol = node

            # Màu theo sector
            node_colors.append(sector_colors[sector])

            # Size theo Lambda
            node_size.append(10 + 60 * lambda_value)

            # Text hiển thị khi hover
            node_text.append(
                f"Stock: {symbol}<br>Name: {name}<br>Sector: {sector}<br>Lambda: {lambda_value:.4f}"
            )

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
                line=dict(width=1.5, color='black')  # Viền đen cho nổi bật
            ),
            textfont=dict(size=9, color='black'),  # Chữ đen
            hoverinfo='text',
            showlegend=False
        )

        # Tạo figure
        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        # Layout tổng thể
        fig.update_layout(
            title=dict(
                text="<b>Stock Similarity Network</b>",
                x=0.5,
                font=dict(size=16, color="#000", family="Montserrat")
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=350,
            width=600,
            margin=dict(l=0, r=0, t=50, b=20),
            font=dict(size=10, color='#000')
        )

        # Thêm legend theo Sector
        for sector in sorted(set(sector_partition.values())):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=sector_colors[sector]),
                name=f"Sector: {sector}",
                hoverinfo='none',
                showlegend=True
            ))

        return dcc.Graph(figure=fig)


    # Callback cho các biểu đồ khác
    @app.callback(
        [Output('beta-sector-table', 'children'),
         Output('lambda-histogram', 'figure'),
         Output('beta-histogram', 'figure'),
         Output('sector-cluster-bar', 'figure')],
        Input('refresh', 'n_intervals')
    )
    def update_visualizations(n):
        # Tính trung bình Beta (1Y) theo Sector
        avg_beta_by_sector = df.groupby('Sector')['Beta (1Y)'].mean().reset_index()
        avg_lambda_by_sector = df.groupby('Sector')['Lambda'].mean().reset_index()
        num_sectors = len(avg_beta_by_sector)
        avg_stats_by_sector = pd.merge(avg_beta_by_sector, avg_lambda_by_sector, on='Sector', how='outer')
        num_sectors = len(avg_stats_by_sector)
        avg_stats_by_sector = avg_stats_by_sector.sort_values(by='Beta (1Y)', ascending=False)
        
        # Tạo bảng
        num_cols = 1 if num_sectors <= 6 else 3
        headers = [html.Th("Sector"), html.Th("Average Beta (1Y)"), html.Th("Average Lambda")] * num_cols
        header_row = html.Tr(headers)
        
        sectors_with_negative_beta = df[df['Beta (1Y)'] < 0]['Sector'].unique()
        
        rows = []
        for i in range(0, num_sectors, num_cols):
            row_data = avg_stats_by_sector.iloc[i:i + num_cols].reset_index(drop=True)
            row_cells = []
            for j in range(num_cols):
                if j < len(row_data):
                    sector, beta, lambda_val = row_data.iloc[j]['Sector'], row_data.iloc[j]['Beta (1Y)'], row_data.iloc[j]['Lambda']
                    sector_display = f"{sector} ※" if sector in sectors_with_negative_beta else sector
                    sector_cell = html.Td(sector_display, style={'text-align': 'left'})
                    beta_cell = html.Td(f"{beta:.4f}")
                    lambda_cell = html.Td(f"{lambda_val:.4f}")
                    row_cells.extend([sector_cell, beta_cell, lambda_cell])
                else:
                    row_cells.extend([html.Td("", style={'text-align': 'left'}), html.Td(""), html.Td("")])
            rows.append(html.Tr(row_cells))
        
        beta_sector_table = html.Table([
            html.Thead(header_row),
            html.Tbody(rows)
        ], style={
            'color': "#080c0c",
            'padding': '3px',
            'lineHeight': '1.0',
            'font-size': '20px',
            'border': '1px solid #00f2ff',
            'border-collapse': 'collapse',
            'width': '100%',
            'text-align': 'center'
        })

        color_sequence = [
        "#4da6ff", "#3399ff", "#1a8cff", "#007fff", "#0066cc"
        ]

        lambda_fig = px.histogram(
            df,
            x='Lambda',
            color='Cluster',
            nbins=20,
            title='Lambda Histogram',
            labels={'Lambda': 'Lambda value', 'count': 'Quantity of stock'},
            color_discrete_sequence=color_sequence  # ✅ dùng màu tùy chỉnh
        )

        lambda_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            title=dict(text="<b>Lambda Histogram</b>", x=0.5, font=dict(size=15, color="#FFFFFF")),
            bargap=0.1,
            legend=dict(title='Community', yanchor="top", y=0.99, xanchor="right", x=1.05,font=dict(size=15, color="#FFFFFF"))
        )
        
        # Beta Histogram
        beta_fig = px.histogram(df, x='Beta (1Y)', nbins=20, color='Cluster', 
                                title='Beta (1Y) Histogram',
                                labels={'Beta (1Y)': 'Beta (1Y)'},
                                color_discrete_sequence=px.colors.qualitative.Plotly)
        beta_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            legend=dict(title='Clusters', yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # Sector Distribution by Cluster
        sector_cluster_counts = df.groupby(['Sector', 'Cluster']).size().reset_index(name='count')
        bar_fig = px.bar(sector_cluster_counts,
                        x='Sector',
                        y='count',
                        color='Cluster',
                        title='Sector Distribution by Cluster',
                        color_discrete_sequence=px.colors.qualitative.Plotly)
        bar_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            title=dict(text="<b>Sector Distribution by Cluster</b>", x=0.5, font=dict(size=16, color="#FFFFFF")),
            xaxis=dict(title='Sector', tickangle=45),
            yaxis=dict(title='Count'),
            legend=dict(title='Cluster', yanchor="top", y=0.99, xanchor="right", x=1.05)
        )

        return (beta_sector_table, lambda_fig, beta_fig, bar_fig)

    # Callback cho các biểu đồ bổ sung
    @app.callback(
        [Output('lambda-bar-chart', 'children'),
         Output('pie-chart-sector', 'children'),
         Output('correlation-heatmap', 'figure')],
        Input('refresh', 'n_intervals')
    )
    def update_additional_charts(n):
        # Lambda Bar Chart
        max_lambda_by_cluster = df.loc[df.groupby('Cluster')['Lambda'].idxmax()]
        
        lambda_bar_fig = px.bar(
            max_lambda_by_cluster,
            x='Lambda',
            y='Cluster',
            orientation='h',
            labels={'Lambda': 'Lambda Value'},
            text=max_lambda_by_cluster['Lambda'].round(3),
        )

        lambda_bar_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', color="#000000"),
            xaxis=dict(title="Lambda Value"),
            yaxis=dict(gridcolor='rgba(0, 0, 0, 0)', zerolinecolor='rgba(0, 0, 0, 0)'),
            bargap=0.2,
        )

        lambda_bar_fig.for_each_trace(lambda trace: trace.update(showlegend=False))

        lambda_bar_fig.update_traces(
            marker=dict(line=dict(width=0), opacity=0.9, color="#00FFFF"),
            textposition='inside',
            width=0.3
        )

        # Pie Chart
        sector_counts = df['Sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Stock Quantity']
        
        color_sequence = [
        "#cce5ff", "#b3d8ff", "#99ccff", "#80bfff", "#66b3ff",
        "#4da6ff", "#3399ff", "#1a8cff", "#007fff", "#0066cc"
        ]

        pie_fig = px.pie(
            sector_counts,
            names='Sector',
            values='Stock Quantity',
            title='Stock Distribution by Sector',
            hole=0.2,
            height=400,
            color_discrete_sequence=color_sequence  # ✅ dùng màu gradient xanh dương nhạt
        )

        pie_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            margin=dict(l=50, r=50, t=30, b=80),
            title=dict(text="<b>Stock Distribution by Sector</b>", x=0.5, font=dict(size=16, color="#00CED1")),
            legend=dict(title='Sector', yanchor="top", y=0.2, xanchor="right", x=1.05, font=dict(color='black', size=13))
        )


        # Correlation Heatmap
        correlation_matrix = df[['Lambda', 'Close', 'Change', 'Volume', 'Market Cap', 'EPS', 'Beta (1Y)']].corr()
        corr_fig = px.imshow(correlation_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title='Correlation Heatmap')
        corr_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            title=dict(text="<b>Correlation Heatmap</b>", x=0.5, font=dict(size=16, color="#00CED1")),
            coloraxis_colorbar_title_text='Correlation'
        )
        
        return (
            dcc.Graph(figure=lambda_bar_fig, style={'height': '300px'}),
            dcc.Graph(figure=pie_fig),
            corr_fig
        )

    # Callback cho Community Analysis
    @app.callback(
        [Output('community-stats', 'children'),
         Output('influence-graph', 'children')],
        [Input('community-selector', 'value'),
         Input('refresh', 'n_intervals')]
    )
    def update_community_stats(selected_community, n):
        if selected_community is None:
            return "Select a cluster to analyze", ""

        community_df = df[df['Cluster'] == selected_community]

        # 🎨 Pie chart: Gradient xanh dương nhạt → đậm
        blue_gradient = [
            "#cce5ff", "#b3d8ff", "#99ccff", "#80bfff", "#66b3ff",
            "#4da6ff", "#3399ff", "#1a8cff", "#007fff", "#0066cc"
        ]

        stats_fig = px.pie(
            community_df,
            names='Sector',
            values='Volume',
            title=f'Cluster {selected_community} - Stock Volume',
            hole=0.4,
            color_discrete_sequence=blue_gradient  # ✅ dùng gradient xanh dương
        )

        stats_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',             
            paper_bgcolor='rgba(0,0,0,0)',            
            font=dict(family='Montserrat', color="#fcfdff"),
            title=dict(text=f'Cluster {selected_community} - Stock Volume', font=dict(size=16, color="#00CED1")),
            legend=dict(font=dict(color="#000000"))
        )

        # 🎨 Scatter chart: Gradient xanh dương (sequential)
        influence_fig = px.scatter(
            community_df,
            x='Lambda',
            y='Beta (1Y)',
            size='Volume',
            color='EPS',
            hover_name='Symbol',
            title=f'Cluster {selected_community} - Influence Analysis',
            color_continuous_scale=px.colors.sequential.Blues  # ✅ gradient xanh dương
        )

        influence_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', color="#fcfdff"),
            title=dict(text=f'Cluster {selected_community} - Influence Analysis', font=dict(size=16, color="#00CED1")),
            legend=dict(font=dict(color="#000000")),
            coloraxis_colorbar=dict(
                title=dict(text='EPS', font=dict(color="#000000")),
                tickfont=dict(color="#000000")
            )
        )

        return (
            dcc.Graph(figure=stats_fig, style={'height': '400px', 'width': '100%'}),
            dcc.Graph(figure=influence_fig,style={'height': '400px', 'width': '100%'})
        )

