from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.colors as pc
import dash
import time
import requests
import pickle
from scipy.optimize import minimize
from scipy.linalg import det, inv
import numpy as np
from bs4 import BeautifulSoup
import networkx as nx
import community.community_louvain as community_louvain
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pytz
import base64
import plotly.graph_objects as go
import requests
import random
import pycountry
import json

# ✅ Hàm chuyển mã ISO-2 → ISO-3
def convert_iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=code.strip().upper()).alpha_3
    except:
        return None
# Màu sắc theo phong cách Spotify
SPOTIFY_COLORS = {
    # Màu chính
    'green': '#1DB954',
    'light_green': '#1ED760',
    'dark_green': '#1AA34A',
    'darker_green': '#178A3E',
      # Thêm dòng này (Spotify Green)
    # Màu nền
    'black': '#191414',
    'dark_gray': '#212121',  # Thêm màu này để thay thế cho dark_gray
    'gray': '#535353',
    'light_gray': '#B3B3B3',
    'lighter_gray': '#E5E5E5',
    
    # Màu chữ
    'black': '#FFFFFF',
    'off_black': '#F8F8F8',
    
    # Màu phụ
    'blue': '#2D46B9',
    'purple': '#5038A0'
}
SPOTIFY_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1200px-Spotify_logo_without_text.svg.png"
# Hàm để chuyển đổi hình ảnh URL thành base64
def image_to_base64(url):
    try:
        response = requests.get(url)
        return "data:image/jpeg;base64," + base64.b64encode(response.content).decode('utf-8')
    except:
        return "data:image/jpeg;base64," + base64.b64encode(requests.get("https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228").content).decode('utf-8')
def gradient(lambdas, Y, X, W, zeta, v, alpha, sigma2, theta, delta):
    n = len(Y)
    S = np.eye(len(lambdas)) - W @ np.diag(lambdas)
    S_inv = np.linalg.inv(S)
    S_Y = S @ Y
    #alpha, sigma2 = self._compute_alpha_sigma2(Y, X, S)
    M_X = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T

    grad_logdet = np.array([-np.trace(S_inv @ (np.diag(W[:, i]))) for i in range(n)])
    grad_sigma2 = np.zeros(n)

    residual = S_Y - X @ alpha
    for i in range(n):
        dS = -np.diag(W[:, i])
        dS_Y = dS @ Y
        XTX_inv = np.linalg.pinv(X.T @ X)
        d_residual = dS_Y - X @ XTX_inv @ X.T @ dS_Y
        # Nếu không khớp kích thước, reshape lại chúng
        if residual.shape != d_residual.shape:
            residual = residual.reshape(-1, 1)  # hoặc điều chỉnh sao cho phù hợp
            d_residual = d_residual.reshape(-1, 1)

        reg = 1e-5
        pinv_XTX = np.linalg.pinv(X.T @ X + reg * np.eye(X.shape[1]))

        d_residual = dS_Y - X @ pinv_XTX @ X.T @ dS_Y
        # Cập nhật giá trị grad_sigma2[i]
        grad_sigma2[i] = (residual.T @ d_residual) / (n * sigma2)
    grad_likelihood = grad_logdet - 0.5 * n * grad_sigma2

    Delta_lambda = delta @ lambdas
    grad_penalty = theta * delta.T @ (Delta_lambda - zeta + v / theta)

    return -grad_likelihood + grad_penalty
def loglikelihood(Y, X, W, lambdas, alpha, sigma2):
    S = np.eye(len(lambdas)) - W @ np.diag(lambdas)
    n = len(Y)
    try:
        logdet = np.log(np.abs(det(S)))
    except:
        logdet = -np.inf
    S_Y = S @ Y
    loss = (S_Y - X @ alpha).T @ (S_Y - X @ alpha)
    return logdet - (n/2)*np.log(sigma2)

def penalty(Delta_lambda, zeta, v, theta):
    return (theta/2) * np.sum((Delta_lambda - zeta + v/theta)**2)

def objective(lambdas, Y, X, W, zeta, v, alpha, sigma2, theta, delta):
    n = len(Y)
    Delta_lambda = delta @ lambdas
    return -loglikelihood(Y, X, W, lambdas, alpha, sigma2) + penalty(Delta_lambda, zeta, v, theta)

def update_lambda(lambdas, Y, X, W, zeta, v, alpha, sigma2, theta, delta):
    n = len(Y)
    res = minimize(
        fun=objective,
        x0=lambdas.copy(),
        args=(Y, X, W, zeta, v, alpha, sigma2, theta, delta),
        jac=gradient,
        method='L-BFGS-B',
        bounds=[(0 + 10e-5, 1 - 10e-5)]*n,
        options={'maxiter': 50, 'disp': False}
    )
    return res.x

def simulate_node_removal_impact(node_to_remove, X, Y, W, saved_model):
    """
    Simulate the effect of removing one node from the network on lambda estimates.
    """

    lambda_old = saved_model['lambdas_']
    alpha_hat = saved_model['alpha_']
    sigma2_hat = saved_model['sigma2_']
    delta = saved_model['Delta']
    zeta = saved_model['zeta']
    v = saved_model['v']
    theta = 1.0
    
    n = len(Y)
    keep_indices = np.delete(np.arange(n), node_to_remove)

    # Remove node index from W, X, Y
    W_new = W[np.ix_(keep_indices, keep_indices)]
    delta_new = delta[np.ix_(keep_indices, keep_indices)]
    X_new = X[keep_indices]
    Y_new = Y[keep_indices]
    zeta_new = zeta[keep_indices]
    v_new = v[keep_indices]
    lambda_est = lambda_old[keep_indices]
    
    lambda_new = update_lambda(lambda_est, Y_new, X_new, W_new, zeta_new, v_new, alpha_hat, sigma2_hat, theta, delta_new)

    return lambda_new

def generate_spotify_data():
    df= pd.read_csv("merged_with_lambda.csv")

    # Hiển thị 1 dòng đầu tiên để xác minh có đúng là tên cột không
    print(df.iloc[0, -5:])
    # Nếu thiếu valence thì tạo random
    if 'valence' not in df.columns:
        df['valence'] = np.random.uniform(0, 1, len(df))
    # Mã hóa ảnh nếu cần
    if 'image_base64' not in df.columns and 'image_url' in df.columns:
        df['image_base64'] = df['image_url'].apply(image_to_base64)
    G = nx.Graph()
    artist_col = 'artist_names'
    genre_col = 'genre'

    for idx, row in df.iterrows():
        G.add_node(row['track_name'],
                   artist=row.get(artist_col, ''),
                   genre=row.get(genre_col, ''),
                   popularity=row.get('popularity', 0),
                   type='song')

    for artist in df[artist_col].dropna().unique():
        artist_tracks = df[df[artist_col] == artist]
        tracks = artist_tracks['track_name'].tolist()
        lambda_map = artist_tracks.set_index('track_name')['Lambda'].to_dict()

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                t1, t2 = tracks[i], tracks[j]
                lam1 = lambda_map.get(t1, 0)
                lam2 = lambda_map.get(t2, 0)
                edge_weight = (lam1 + lam2) / 2
                G.add_edge(t1, t2, connection='artist', weight=edge_weight)

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        degree_dict = dict(G.degree())
        betweenness_dict = nx.betweenness_centrality(G, normalized=True)
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000)

        # Gán lại vào df theo track_name
        df['Degree'] = df['track_name'].map(degree_dict).fillna(0)
        df['Betweenness'] = df['track_name'].map(betweenness_dict).fillna(0)
        partition = community_louvain.best_partition(G)
    else:
        df['Degree'] = df['Betweenness'] = df['Lambda'] = 0
        partition = {}

    # Đảm bảo các cột cần thiết không có NaN
    df[['valence', 'genre', 'Degree', 'Betweenness', 'Lambda']] = df[['valence', 'genre', 'Degree', 'Betweenness', 'Lambda']].fillna(0)

    return df, G, partition
def draw_original_network_for_comparison():
    top_nodes = df_songs.nlargest(150, 'popularity')['track_name'].tolist()
    H = G.subgraph(top_nodes).copy()

    # Thêm cạnh giữa các bài hát cùng thể loại
    genre_groups = df_songs[df_songs['track_name'].isin(H.nodes)].groupby('genre')
    for genre, group in genre_groups:
        tracks = group['track_name'].tolist()
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if not H.has_edge(tracks[i], tracks[j]):
                    H.add_edge(tracks[i], tracks[j], connection='genre')

    pos = nx.kamada_kawai_layout(H)
    genre_list = df_songs['genre'].dropna().unique().tolist()
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Dark24
    genre_color_map = {g: color_palette[i % len(color_palette)] for i, g in enumerate(genre_list)}
    top5 = set(df_songs.nlargest(5, 'Lambda')['track_name'])

    edge_traces = []
    for u, v, data in H.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = '#1DB954' if data.get('connection') == 'artist' else '#888'
        width = 2.5 if data.get('connection') == 'artist' else 1
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none',
            showlegend=False
        ))

    node_x, node_y, node_text, node_color, node_size, node_border = [], [], [], [], [], []
    for node in H.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = H.nodes[node]
        genre = df_songs[df_songs['track_name'] == node]['genre'].values[0] if node in df_songs['track_name'].values else 'Other'
        eigen = df_songs[df_songs['track_name'] == node]['Lambda'].values[0] if node in df_songs['track_name'].values else 0.1
        popularity = df_songs[df_songs['track_name'] == node]['popularity'].values[0] if node in df_songs['track_name'].values else 0
        artist = df_songs[df_songs['track_name'] == node]['artist_names'].values[0] if node in df_songs['track_name'].values else ''
        
        node_text.append(
            f"<b>{node}</b><br>"
            f"Nghệ sĩ: {artist}<br>"
            f"Thể loại: {genre}<br>"
            f"Độ ảnh hưởng: {eigen:.2f}<br>"
            f"Độ phổ biến: {popularity}"
        )
        node_color.append(genre_color_map.get(genre, '#CCCCCC'))
        base_size = 35 if node in top5 else 18
        node_size.append(base_size + 30 * abs(eigen))
        node_border.append('rgba(255,255,255,0.7)')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=4, color=node_border),
            opacity=0.97,
            symbol='circle'
        )
    )

    # Legend theo thể loại
    legend_traces = []
    for genre, color in genre_color_map.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=genre,
            showlegend=True
        ))

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(
                text='<b>SPOTIFY MUSIC NETWORK</b><br><sup>Mạng lưới gốc (trước khi xoá)</sup>',
                font=dict(color='black', size=18, family="Montserrat"),
                x=0.05
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                title='Thể loại',
                font=dict(color='black', size=10),
                orientation='v',
                x=1.05,
                y=1,
                xanchor='left'
            ),
            height=400,
            width=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black'),
            hoverlabel=dict(
                bgcolor='rgba(30,30,30,0.8)',
                font_size=14,
                font_family="Montserrat"
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
    )
    return fig

df_songs, G, partition = generate_spotify_data()
def simulate_propagation(G, seed_nodes, max_steps=5):
    """Mô phỏng quá trình lan truyền ảnh hưởng trong đồ thị
    
    Args:
        G: Đồ thị networkx
        seed_nodes: Danh sách node bắt đầu
        max_steps: Số bước lan truyền tối đa
    
    Returns:
        Dict chứa kết quả mô phỏng
    """
    infection_time = {node: 0 for node in seed_nodes}
    current_frontier = set(seed_nodes)
    all_infected = set(seed_nodes)
    propagation_paths = set()
    
    for step in range(1, max_steps + 1):
        next_frontier = set()
        
        for node in current_frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in all_infected:
                    # Tính xác suất lây nhiễm dựa trên trọng số edge
                    infection_prob = G[node][neighbor].get('weight', 0.5)
                    if random.random() < infection_prob:
                        infection_time[neighbor] = step
                        next_frontier.add(neighbor)
                        all_infected.add(neighbor)
                        propagation_paths.add((node, neighbor))
        
        current_frontier = next_frontier
        if not current_frontier:
            break
    
    return {
        'all_nodes': list(all_infected),
        'infection_time': infection_time,
        'propagation_paths': propagation_paths
    }

# Hàm lấy dữ liệu lan truyền
def get_propagation_data(selected_artists, selected_songs, df, G):
    """Hàm phụ trợ lấy dữ liệu lan truyền từ đồ thị G và dataframe df
    
    Args:
        selected_artists: Danh sách nghệ sĩ được chọn
        selected_songs: Danh sách bài hát được chọn
        df: DataFrame chứa dữ liệu bài hát
        G: Đồ thị networkx đã được xây dựng
    
    Returns:
        Dict chứa dữ liệu nodes và edges cho visualization
    """
    if not selected_artists and not selected_songs:
        return None
    
    # Chuyển đổi input thành list nếu là string
    if isinstance(selected_artists, str):
        selected_artists = [selected_artists]
    if isinstance(selected_songs, str):
        selected_songs = [selected_songs]
    
    # Lấy tất cả các bài hát liên quan đến nghệ sĩ được chọn
    related_songs = []
    if selected_artists:
        related_songs.extend(df[df['artist_names'].isin(selected_artists)]['track_name'].tolist())
    if selected_songs:
        related_songs.extend(selected_songs)
    
    if not related_songs:
        return None
    
    # Tìm các node có ảnh hưởng trong đồ thị
    nodes_to_analyze = [song for song in related_songs if song in G]
    
    if not nodes_to_analyze:
        return None
    
    # Mô phỏng quá trình lan truyền
    propagation_results = simulate_propagation(G, nodes_to_analyze)
    
    # Chuẩn bị dữ liệu nodes cho visualization
    nodes_data = []
    node_id_map = {node: idx for idx, node in enumerate(propagation_results['all_nodes'])}
    
    for node in propagation_results['all_nodes']:
        artist = G.nodes[node].get('artist', 'Unknown')
        genre = G.nodes[node].get('genre', 'Unknown')
        popularity = G.nodes[node].get('popularity', 0)
        lambda_val = df[df['track_name'] == node]['Lambda'].values[0] if node in df['track_name'].values else 0
        
        nodes_data.append({
            'id': node_id_map[node],
            'label': f"{node[:15]}..." if len(node) > 15 else node,
            'full_label': node,
            'artist': artist,
            'time': propagation_results['infection_time'].get(node, 0),
            'value': lambda_val,
            'size': 10 + (lambda_val * 30),
            'color': '#1DB954' if node in nodes_to_analyze else '#FF5733',
            'genre': genre,
            'popularity': popularity
        })
    
    # Chuẩn bị dữ liệu edges cho visualization
    edges_data = []
    for edge in propagation_results['propagation_paths']:
        source, target = edge
        edges_data.append({
            'source': node_id_map[source],
            'target': node_id_map[target],
            'source_time': propagation_results['infection_time'].get(source, 0),
            'target_time': propagation_results['infection_time'].get(target, 0),
            'width': 0.5 + (G[source][target]['weight'] * 3 if 'weight' in G[source][target] else 1)
        })
    
    return {
        'nodes': nodes_data,
        'edges': edges_data
    }
def register_spotify_callbacks(app):
    # Cập nhật thời gian
    @app.callback(
        Output('time-display', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_time(n):
        tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(tz)
        return current_time.strftime('%H:%M %d/%m/%Y')
    @app.callback(
        [
            Output('avg-popularity-display', 'children'),
            Output('top-genre-display', 'children'),
            Output('avg-duration-display', 'children'),
            Output('unique-artists-display', 'children')
        ],
        [Input('tabs', 'active_tab')]
    )
    def update_metrics(tab):
        if tab != 'overview':
            return [dash.no_update] * 6
        df_songs, _, _ = generate_spotify_data() 
        # 1. Độ phổ biến trung bình
        avg_popularity = df_songs['popularity'].mean()
        avg_popularity_str = f"{avg_popularity:.1f}"

        # 2. Thể loại phổ biến nhất
        top_genre = df_songs['genre'].value_counts().idxmax()

        # 3. Thời lượng trung bình từ track_duration_ms
        avg_duration_ms = df_songs['track_duration_ms'].mean()
        minutes = int(avg_duration_ms // 60000)
        seconds = int((avg_duration_ms % 60000) // 1000)
        avg_duration_str = f"{minutes}:{seconds:02d}"

        unique_artists = df_songs['artist_names'].nunique()

        # 5. Tempo (BPM)
        avg_tempo = df_songs['BPM'].mean()
        avg_tempo_str = f"{avg_tempo:.0f}"

        # 6. Energy
        avg_energy = df_songs['energy'].mean()
        avg_energy_str = f"{avg_energy:.2f}"

        return (
            avg_popularity_str,
            top_genre,
            avg_duration_str,
            unique_artists
        )
    # Biểu đồ ảnh hưởng theo thể loại
    @app.callback(
        Output('genre-influence-chart', 'figure'),
        Input('tabs', 'active_tab')
    )

    def update_genre_influence(tab):
        if tab != 'overview':
            return {}
        genre_counts = df_songs['genre'].value_counts().reset_index()
        genre_counts.columns = ['genre', 'count']

        fig = go.Figure(go.Pie(
                labels=genre_counts['genre'],
                values=genre_counts['count'],
                hole=0.4,
                pull=[0.08]*len(genre_counts),  # hiệu ứng nổi
                marker=dict(colors=[
                    'rgba(36, 176, 85, 0.4)',
                    'rgba(29, 185, 84, 0.63)',
                    'rgba(21, 194, 81, 0.74)',
                    'rgba(29, 185, 84, 1.0)',
                    'rgba(14, 174, 70, 1.0)',
                    'rgba(3, 120, 44, 1.0)'
                ]),
                textfont=dict(
                    color='black',      
                    size=12,            
                    family='Montserrat'
                )
            ))
        # Chèn logo Spotify vào giữa donut
        fig.update_layout(
            title=dict(
            text='<b>DISTRIBUTION OF SONGS BY GENRE</b>',
            font=dict(size=18, color='black', family='Montserrat'),
            x=0.5,  # căn giữa theo chiều ngang
            xanchor='center'  # neo điểm căn giữa
                ),
            font=dict(color=SPOTIFY_COLORS['black'], size=11, family="Montserrat"),
            images=[dict(
                source="https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg",  # URL ảnh SVG
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                sizex=0.22, sizey=0.22, 
                xanchor="center",
                yanchor="middle",
                layer="above"
            )],
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',   # nền biểu đồ trong suốt
            paper_bgcolor='rgba(0,0,0,0)',  # nền toàn khung hình trong suốt
            font_color='black',
            showlegend=True,
            legend_title_text='Genre',
        )

        return fig

    @app.callback(
        Output('top-songs-chart', 'figure'),
        Input('tabs', 'active_tab')
    )
    def update_top_songs(tab):
        if tab != 'overview':
            return go.Figure()
        
        # Lấy và sắp xếp dữ liệu
        top_songs = df_songs.nlargest(5, 'Lambda').sort_values('Lambda', ascending=True)
        x = top_songs['Lambda']
        y = top_songs['track_name']
        popularity = top_songs['popularity']

        # Tạo figure
        fig = go.Figure()
        
        # Thêm cột hình trụ (sử dụng cornerradius)
        fig.add_trace(go.Bar(
            y=y,
            x=x,
            orientation='h',
            name='Influence',
            marker=dict(
                color=popularity,
                    colorscale=[
                    [0.0, "#178A3E"],   # xanh đậm nhất
                    [0.2, "#1AA34A"],   # xanh đậm
                    [0.5, "#1DB954"],   # xanh Spotify
                    [0.8, "#1ED760"],   # xanh nhạt
                    [1.0, "#B2F2C9"]    # xanh rất nhạt
                ],
                cmin=0,
                cmax=100,
                line=dict(width=0),
                cornerradius=20  # Bo tròn các góc tạo hiệu ứng hình trụ
            ),
            hoverinfo='text',
            hovertext=[
                f"<b>{song}</b><br>"
                f"Artist: {artist}<br>"
                f"Genre: {genre}<br>"
                f"Influence: {infl:.8f}<br>"
                f"Popularity: {pop}/100"
                for song, artist, genre, infl, pop in zip(
                    top_songs['track_name'],
                    top_songs['artist_names'],
                    top_songs['genre'],
                    top_songs['Lambda'],
                    top_songs['popularity']
                )
            ],
            textposition='auto',
            texttemplate='%{x:.3f}',
            width=0.5  # Độ rộng của cột
        ))

        fig.update_layout(
            title={
                'text': "<b>TOP 5 MOST INFLUENTIAL SONGS</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 18,
                    'color': 'black',
                    'family': "Montserrat"
                }
            },
            xaxis=dict(
                title="<b>Influence Score (Lambda Centrality)</b>",
                title_font=dict(size=12, color='black'),
                tickfont=dict(size=11, color='black'),
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis=dict(
                title=None,
                tickfont=dict(size=12, color='black'),
                autorange="reversed"
            ),
            margin=dict(l=120, r=20, t=30, b=40),
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',   # nền biểu đồ trong suốt
            paper_bgcolor='rgba(0,0,0,0)',  # nền toàn khung hình trong suốt
            legend_title_text='Genre',
            hoverlabel=dict(
                bgcolor='rgba(29,185,84,0.8)',
                font_size=13,
                font_family="Montserrat"
            ),
            showlegend=False
        )

        return fig
    # Mạng lưới bài hát
   # Callback
    def draw_song_network(tab_value, clickData, genre_click_data=None):
        if tab_value != 'overview':
            return go.Figure()

        top_nodes = df_songs.nlargest(150, 'popularity')['track_name'].tolist()
        H = G.subgraph(top_nodes).copy()

        # Thêm cạnh giữa các bài hát cùng thể loại (genre)
        genre_groups = df_songs[df_songs['track_name'].isin(H.nodes)].groupby('genre')
        for genre, group in genre_groups:
            tracks = group['track_name'].tolist()
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    if not H.has_edge(tracks[i], tracks[j]):
                        H.add_edge(tracks[i], tracks[j], connection='genre')

        pos = nx.kamada_kawai_layout(H)
        genre_list = df_songs['genre'].unique().tolist()
        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Dark24
        genre_color_map = {g: color_palette[i % len(color_palette)] for i, g in enumerate(genre_list)}

        # Top 5 influential nodes
        top5 = set(df_songs.nlargest(5, 'Lambda')['track_name'])

        # Xác định node được chọn
        selected_node = None
        selected_genre = None
        
        # Kiểm tra click từ bảng thống kê thể loại
        if genre_click_data and 'points' in genre_click_data and len(genre_click_data['points']) > 0:
            point = genre_click_data['points'][0]
            selected_genre = point.get('label') or point.get('x')  # Tùy thuộc vào cách bạn xây dựng bảng
        
        # Kiểm tra click trực tiếp trên đồ thị
        elif clickData and 'points' in clickData and len(clickData['points']) > 0:
            point = clickData['points'][0]
            node_label = point.get('hovertext') or point.get('text')
            if node_label:
                selected_node = node_label.split('<br>')[0].replace('<b>', '').replace('</b>', '')
                # Lấy thể loại của node được chọn
                selected_genre = df_songs[df_songs['track_name'] == selected_node]['genre'].values[0] if not df_songs[df_songs['track_name'] == selected_node].empty else None

        edge_traces = []
        for edge in H.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if edge[2].get('connection') == 'artist':
                color = '#1DB954'
                width = 2.5
            else:
                # Highlight các edge cùng thể loại nếu đang chọn thể loại
                if selected_genre:
                    node1_genre = df_songs[df_songs['track_name'] == edge[0]]['genre'].values[0] if not df_songs[df_songs['track_name'] == edge[0]].empty else None
                    node2_genre = df_songs[df_songs['track_name'] == edge[1]]['genre'].values[0] if not df_songs[df_songs['track_name'] == edge[1]].empty else None
                    if node1_genre == selected_genre and node2_genre == selected_genre:
                        color = genre_color_map.get(selected_genre, '#FFD700')
                        width = 3
                    else:
                        color = 'rgba(200,200,200,0.2)'
                        width = 0.5
                else:
                    color = '#888'
                    width = 1
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=width, color=color),
                hoverinfo='none',
                mode='lines',
                showlegend=False 
            ))

        node_x, node_y, node_text, node_color, node_size, node_border = [], [], [], [], [], []
        genre_nodes = []
        
        for node in H.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = H.nodes[node]
            genre = node_data.get('genre', 'Other')
            eigen = df_songs[df_songs['track_name'] == node]['Lambda'].values
            eigen = eigen[0] if len(eigen) > 0 else 0.1
            popularity = df_songs[df_songs['track_name'] == node]['popularity'].values
            popularity = popularity[0] if len(popularity) > 0 else 0
            artist = node_data.get('artist', '')
            node_text.append(
                f"<b>{node}</b><br>"
                f"Nghệ sĩ: {artist}<br>"
                f"Thể loại: {genre}<br>"
                f"Độ ảnh hưởng: {eigen:.2f}<br>"
                f"Độ phổ biến: {popularity}"
            )
            
            # Xác định màu sắc và kích thước node
            base_color = genre_color_map.get(genre, '#CCCCCC')
            
            # Nếu đang chọn thể loại, làm mờ các node không thuộc thể loại đó
            if selected_genre:
                if genre == selected_genre:
                    node_color.append(base_color)
                    genre_nodes.append(node)
                    # Node top 5 lớn hơn
                    if node in top5:
                        base_size = 35
                    else:
                        base_size = 25  # Tăng kích thước cho các node cùng thể loại
                    node_size.append(base_size + 40 * abs(eigen))
                    node_border.append('yellow')
                else:
                    node_color.append('rgba(200,200,200,0.3)')
                    node_size.append(10 + 10 * abs(eigen))  # Giảm kích thước cho các node khác thể loại
                    node_border.append('rgba(200,200,200,0.5)')
            else:
                node_color.append(base_color)
                # Node top 5 lớn hơn
                if node in top5:
                    base_size = 35
                else:
                    base_size = 18
                # Nếu là node được chọn thì phát sáng (tăng size, đổi màu viền)
                if selected_node and node == selected_node:
                    node_size.append(base_size + 40 * abs(eigen))
                    node_border.append('yellow')
                else:
                    node_size.append(base_size + 30 * abs(eigen))
                    node_border.append('rgba(255,255,255,0.7)')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=4, color=node_border),
                opacity=0.97,
                symbol='circle'
            ),
            textfont=dict(
                family="Montserrat",
                size=13,
                color="black"
            )
        )
        
        highlight_trace = None
        highlight_genre_trace = None
        annotations = []
        x_range = y_range = None

        # Tính toán phạm vi zoom nếu có thể loại được chọn
        if selected_genre and genre_nodes:
            # Lấy tọa độ của tất cả các node thuộc thể loại được chọn
            genre_x = [pos[node][0] for node in genre_nodes]
            genre_y = [pos[node][1] for node in genre_nodes]
            
            # Tính toán min/max để xác định phạm vi zoom
            if genre_x and genre_y:
                x_min, x_max = min(genre_x), max(genre_x)
                y_min, y_max = min(genre_y), max(genre_y)
                
                # Thêm padding xung quanh
                x_padding = (x_max - x_min) * 0.3
                y_padding = (y_max - y_min) * 0.3
                
                x_range = [x_min - x_padding, x_max + x_padding]
                y_range = [y_min - y_padding, y_max + y_padding]

        # Hiệu ứng highlight cho node được chọn
        if selected_node and selected_node in H.nodes:
            x, y = pos[selected_node]
            highlight_trace = go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=node_size[list(H.nodes()).index(selected_node)] * 2.2,
                    color='rgba(255, 215, 0, 0.25)',
                    line=dict(width=4, color='rgba(255, 215, 0, 0.7)'),
                    symbol='circle'
                ),
                hoverinfo='skip',
                showlegend=False
            )
            
            # Hiện thông tin ngay trên node
            node_data = H.nodes[selected_node]
            eigen = df_songs[df_songs['track_name'] == selected_node]['Lambda'].values
            eigen = eigen[0] if len(eigen) > 0 else 0.1
            popularity = df_songs[df_songs['track_name'] == selected_node]['popularity'].values
            popularity = popularity[0] if len(popularity) > 0 else 0
            artist = node_data.get('artist', '')
            genre = node_data.get('genre', '')
            annotations.append(dict(
                x=x, y=y,
                xref='x', yref='y',
                text=(
                    f"<b>{selected_node}</b><br>"
                    f"Nghệ sĩ: {artist}<br>"
                    f"Thể loại: {genre}<br>"
                    f"Độ ảnh hưởng: {eigen:.2f}<br>"
                    f"Độ phổ biến: {popularity}"
                ),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-60,
                bgcolor="#fff",
                bordercolor="#1DB954",
                borderwidth=2,
                borderpad=6,
                font=dict(size=13, color="black", family="Montserrat"),
                align="center"
            ))
        
        # Hiệu ứng highlight cho toàn bộ thể loại được chọn
        if selected_genre and genre_nodes:
            # Tạo một trace bao quanh toàn bộ nhóm thể loại
            x_values = [pos[node][0] for node in genre_nodes]
            y_values = [pos[node][1] for node in genre_nodes]
            
            # Tính toán hình bao (convex hull) cho các node cùng thể loại
            if len(genre_nodes) > 2:
                from scipy.spatial import ConvexHull
                points = np.array([(pos[node][0], pos[node][1]) for node in genre_nodes])
                hull = ConvexHull(points)
                
                # Lấy các điểm trên convex hull
                hull_x = points[hull.vertices, 0]
                hull_y = points[hull.vertices, 1]
                
                # Đóng kín hình bao bằng cách thêm điểm đầu vào cuối
                hull_x = np.append(hull_x, hull_x[0])
                hull_y = np.append(hull_y, hull_y[0])
                
                highlight_genre_trace = go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode='lines',
                    fill='toself',
                    fillcolor=genre_color_map.get(selected_genre, '#FFD700'),
                    opacity=0.2,
                    line=dict(
                        color=genre_color_map.get(selected_genre, '#FFD700'),
                        width=3,
                        dash='dot'
                    ),
                    hoverinfo='text',
                    hovertext=f"Thể loại: {selected_genre}<br>Số bài hát: {len(genre_nodes)}",
                    showlegend=False
                )
            
            # Thêm annotation hiển thị tên thể loại
            if x_values and y_values:
                center_x = sum(x_values) / len(x_values)
                center_y = sum(y_values) / len(y_values)
                annotations.append(dict(
                    x=center_x,
                    y=center_y,
                    xref='x',
                    yref='y',
                    text=f"<b>{selected_genre}</b>",
                    showarrow=False,
                    font=dict(
                        size=16,
                        color=genre_color_map.get(selected_genre, '#FFD700'),
                        family="Montserrat"
                    ),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=genre_color_map.get(selected_genre, '#FFD700'),
                    borderwidth=2,
                    borderpad=4
                ))

        legend_traces = []
        for genre, color in genre_color_map.items():
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=genre,
                showlegend=True,
                name=genre
            ))

        # Tạo figure với các traces
        traces = edge_traces + [node_trace] + legend_traces
        if highlight_trace:
            traces.append(highlight_trace)
        if highlight_genre_trace:
            traces.append(highlight_genre_trace)

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(
                    text='<b>SPOTIFY MUSIC NETWORK</b> (Select node or genre to see details)',
                    font=dict(color='black', size=20, family="Montserrat"),
                    x=0.05,
                    y=0.95,
                    xanchor='left',
                    yanchor='top'
                ),
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False,
                    range=x_range if x_range else None
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False,
                    range=y_range if y_range else None
                ),
                annotations=annotations,
                showlegend=True,
                legend=dict(
                    title='Genre',
                    font=dict(color='black', size=10),
                    orientation='v',
                    x=1.05,
                    y=1,
                    xanchor='left',
                    itemsizing='constant',
                    traceorder='normal'
                ),
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=40),
                height=350,
                width=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black'),
                hoverlabel=dict(
                    bgcolor='rgba(30,30,30,0.8)',
                    font_size=15,
                    font_family="Montserrat"
                ),
                dragmode='pan'
            )
        )

        return fig
    @app.callback(
    Output('network-song', 'figure'),
    [
        Input('network-song', 'clickData'),  # click vào node
        Input('genre-lambda-table', 'selected_rows'),  # click bảng
        Input('tabs', 'active_tab')
    ],
        State('genre-lambda-table', 'data')
    )
    def update_network(clickData, selected_rows, tab_value, table_data):
        genre_click_data = None
        if selected_rows:
            selected_genre = table_data[selected_rows[0]]['Genre']
            genre_click_data = {'points': [{'x': selected_genre}]}

        return draw_song_network(
            tab_value=tab_value,
            clickData=clickData,  # <== vẫn truyền clickData từ Input
            genre_click_data=genre_click_data
        )

    # Bài hát trong cộng đồng và hiển thị hình ảnh
    @app.callback(
        Output('top-artist-chart', 'figure'),
        Input('tabs', 'active_tab')
    )
    def update_top_artist_chart(tab):
        if tab != 'overview':
            return go.Figure()

        # Lấy top 5 nghệ sĩ phổ biến
        top_artists_df = (
           df_songs[['artist_names', 'artist_popularity', 'spotify_cover_url']]
            .dropna(subset=['artist_popularity'])
            .sort_values(by='artist_popularity', ascending=False)
            .drop_duplicates(subset=['artist_names'])
            .head(5)
        )
        artists = top_artists_df['artist_names'].tolist()
        image_urls = top_artists_df['spotify_cover_url'].tolist()
        base64_images = [image_to_base64(url) for url in image_urls]

        fig = go.Figure()
        fig.update_layout(
            title={
                'text': '<b>TOP 5 POPULAR ARTISTS</b>',
                'y': 1,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='black', family='Montserrat')
            },
            margin=dict(t=100)
        )

        for i, artist in enumerate(artists):
            wrapped_name = artist if len(artist) <= 25 else artist[:22] + '...'
            y_pos = 0.85 - i * 0.16
            fig.add_layout_image(
                dict(
                    source=base64_images[i],
                    xref="paper", yref="paper",
                    x=0.13, y=y_pos,
                    sizex=0.13, sizey=0.13,
                    xanchor="center", yanchor="middle",
                    layer="above"
                )
            )
            fig.add_annotation(
                x=0.5, y=y_pos,
                xref="paper", yref="paper",
                text=wrapped_name,
                showarrow=False,
                font=dict(size=16, color="black", family="Montserrat"),
                xanchor="center", yanchor="middle"
            )

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(color="rgba(0,0,0,0.05)", width=1),
                    layer="below"
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )

        return fig

    #---------------------------------------------------------Tab2---------------------------------------------
    @app.callback(
    Output('distribution-by-channel', 'figure'),
    [Input('tabs', 'active_tab'),
     Input('attribute-filter', 'value'),
     Input('community-dropdown', 'value')]  # thêm genre
    )
    def update_channel_distribution_chart(tab, selected_attribute, selected_genre):
        if tab != 'community':
            return go.Figure()

        filtered_df = df_songs.copy()
        if selected_genre:
            filtered_df = filtered_df[filtered_df['genre'] == selected_genre]

        x_values = filtered_df[selected_attribute].dropna()

        # Tính histogram
        counts, bins = np.histogram(x_values, bins='auto')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Tâm của các bin

        # Tạo dữ liệu mượt bằng interpolation (nội suy bậc 3)
        from scipy import interpolate
        x_smooth = np.linspace(min(bin_centers), max(bin_centers), 500)
        spline = interpolate.make_interp_spline(bin_centers, counts, k=3)  # Bậc 3 để đường cong mượt
        y_smooth = spline(x_smooth)

        # Tạo màu gradient cho cột
        colorscale = pc.sample_colorscale('Greens', [i / (len(counts)-1) for i in range(len(counts))])
        
        fig = go.Figure()

        # Vẽ các cột histogram
        for i in range(len(counts)):
            fig.add_trace(go.Bar(
                x=[bin_centers[i]],
                y=[counts[i]],
                width=[bins[1] - bins[0]],
                marker=dict(color=colorscale[i]),
                showlegend=False,
                name='Histogram',
                customdata=[[round(bin_centers[i], 2)]],
                hovertemplate="<b>Value:</b> %{customdata[0]}<br><b>Count:</b> %{y}<extra></extra>"
            ))

        # Thêm đường cong mượt màu tím nối các đỉnh
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            line=dict(color="#177E6D", width=3),  # Màu tím
            name='Smooth Line',
            hovertemplate="<b>Value:</b> %{x:.2f}<br><b>Smoothed Count:</b> %{y:.2f}<extra></extra>"
        ))

        # Thêm các điểm nối màu xanh lá đậm tại đỉnh mỗi cột
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=counts,
            mode='markers',
            marker=dict(
                color='#006400',  # Xanh lá đậm
                size=8,
                line=dict(width=2, color='white')
            ),
            name='Peak Points',
            hovertemplate="<b>Bin Center:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>"
        ))

        # Tính mean và median
        mean_val = np.mean(x_values)
        median_val = np.median(x_values)
        
        # Đường mean (trung bình)
        fig.add_vline(
            x=mean_val,
            line_width=2,
            line_dash="dash",
            line_color="#1DB954",
            annotation_text="Mean",
            annotation_position="top",
            annotation_font=dict(color="#1DB954", size=12)
        )
        
        # Đường median (trung vị)
        fig.add_vline(
            x=median_val,
            line_width=2,
            line_dash="dot",
            line_color="#FF5733",
            annotation_text="Median",
            annotation_position="bottom",
            annotation_font=dict(color="#FF5733", size=12)
        )

        # Cấu hình layout
        fig.update_layout(
            xaxis_title=selected_attribute.capitalize(),
            yaxis_title='Frequency',
            title=f'<b>Distribution of {selected_attribute.capitalize()}</b>',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black'),
            bargap=0.05,
            height=600,
            hovermode='closest',
            margin=dict(l=40, r=30, t=60, b=40),
        )

        # Cấu hình grid
        fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.3)')

        return fig
    @app.callback(
        Output('density-chart', 'figure'),
        [Input('tabs', 'active_tab'),
        Input('y-attribute-filter', 'value'),
        Input('attribute-filter', 'value'),
        Input('community-dropdown', 'value')]  # thêm genre
    )
    def update_density_chart(tab, y_channel, selected_attribute, selected_genre):
        if tab != 'community':
            return go.Figure()

        filtered_df = df_songs.copy()
        if selected_genre:
            filtered_df = filtered_df[filtered_df['genre'] == selected_genre]
      
            fig = px.density_heatmap(
                filtered_df,
                x=selected_attribute,
                y=y_channel,
                nbinsx=20,
                nbinsy=20,
                color_continuous_scale=[
                    [0.0, "rgba(26,163,74, 0.05)"],  # light mint at 30% opacity
                    [0.2, "rgba(26,163,74, 0.2)"],
                    [0.4, "rgba(26,163,74, 0.4)"],
                    [0.6, "rgba(26,163,74, 0.6)"],
                    [0.8, "rgba(26,163,74, 0.8)"],
                    [1.0, "rgba(26,163,74, 1.0)"]
                ]
            )
            fig.update_layout(
                title=f'<b>Density of {y_channel.capitalize()} and {selected_attribute.capitalize()}<b>',
                xaxis_title=selected_attribute.capitalize(),
                yaxis_title=y_channel.capitalize(),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black'),
                height=300,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 1.0)'  # light gray with 20% opacity
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 1.0)'  # light gray with 20% opacity
                )
            )
            return fig
        
    @app.callback(
        Output('popularity-by-genre-chart', 'figure'),
        [Input('tabs', 'active_tab'),
        Input('y-attribute-filter', 'value')]  # thêm genre
    )
    def update_popularity_by_genre_chart(tab, y_channel):
        if tab != 'community' or not y_channel or y_channel not in df_songs.columns:
            return go.Figure()

        # Dữ liệu không lọc theo genre
        filtered_df = df_songs.dropna(subset=['genre', y_channel]).copy()

        fig = px.scatter(
            filtered_df,
            x=y_channel,
            y='genre',  # Trục tung là tên genre
            title=f'<b>{y_channel.capitalize()} by Genre (Horizontal Scatter)</b>',
            color=y_channel,
            color_continuous_scale='Viridis',
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title=y_channel.capitalize(),
            yaxis_title='Genre',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black'),
            height=300,
            margin=dict(t=30, l=60, r=30, b=40),  # 🔻 Giảm khoảng cách trên (top)
            title=dict(
                y=0.92,  # 🔻 Gần hơn so với mặc định (0.95 ~ 1)
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.6)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.6)',
                categoryorder='total ascending'
            )
        )
        return fig

    # Hàm hỗ trợ chuyển hex sang rgba
    def hex_to_rgb(hex_color, alpha=1):
        """Chuyển mã màu hex sang tuple rgba"""
        hex_color = hex_color.lstrip('#')
        hlen = len(hex_color)
        rgb = tuple(int(hex_color[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
        return rgb + (alpha,)
    def create_genre_distribution_map(df, top_n=30, min_count=5, color_scale=None, 
                                map_style="natural earth", show_animation=True):
        """
        Create an advanced global music genre distribution visualization with location icons
        """
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go

            # Input data validation
            required_columns = ['genre', 'region_list', 'region_name_list', 'track_name', 'artist_name']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")

            df = df.dropna(subset=['genre', 'region_list', 'region_name_list']).copy()
            if df.empty:
                raise ValueError("No valid data after filtering")

            # Prepare region data
            region_data = []
            for _, row in df.iterrows():
                regions = str(row['region_list']).strip().split(',')
                region_names = str(row['region_name_list']).strip().split(',')

                regions, region_names = zip(*[
                    (r.strip(), n.strip()) 
                    for r, n in zip(regions, region_names) 
                    if r.strip() and n.strip()
                ][:min(len(regions), len(region_names))])

                for region, name in zip(regions, region_names):
                    region_data.append({
                        'region_code': region,
                        'region_name': name,
                        'genre': row['genre'],
                        'track_name': row.get('track_name', 'Unknown'),
                        'artist_name': row.get('artist_name', 'Unknown')
                    })

            region_df = pd.DataFrame(region_data)

            # Filter countries with at least min_count occurrences
            region_counts = region_df.groupby(['region_code', 'genre']).size().reset_index(name='count')
            region_counts = region_counts[region_counts['count'] >= min_count]

            # Convert ISO2 to ISO3
            region_counts['region_code_iso3'] = region_counts['region_code'].apply(convert_iso2_to_iso3)
            region_counts = region_counts.dropna(subset=['region_code_iso3'])

            # Add country names
            unique_regions = region_df[['region_code', 'region_name']].drop_duplicates()
            region_counts = region_counts.merge(unique_regions, on='region_code', how='left')

            # Filter top N countries
            top_countries = region_counts.groupby('region_code')['count'].sum().nlargest(top_n).index
            region_counts = region_counts[region_counts['region_code'].isin(top_countries)]

            stats = {
                'total_countries': len(top_countries),
                'total_genres': region_counts['genre'].nunique(),
                'max_count': region_counts['count'].max(),
                'min_count': region_counts['count'].min(),
                'avg_count': round(region_counts['count'].mean(), 1)
            }
            
            top_content = region_df.groupby(['region_code', 'genre']).agg({
                'track_name': lambda x: x.mode()[0] if not x.empty else 'Unknown',
                'artist_name': lambda x: x.mode()[0] if not x.empty else 'Unknown',
                'region_name': lambda x: x.mode()[0] if not x.empty else 'Unknown'
            }).reset_index().rename(columns={'region_name': 'region_name_popular'})

            region_counts = region_counts.merge(top_content, on=['region_code', 'genre'], how='left')

            # Create hover text column
            region_counts['hover_region_name'] = region_counts.get('region_name', region_counts['region_name_popular'])

            def create_hover_text(row):
                return (
                    f"<b>{row['hover_region_name']}</b><br>"
                    f"<b>Genre:</b> {row['genre']}<br>"
                    f"<b>Appearances:</b> {row['count']}<br>"
                    f"<b>Top Track:</b> {row['track_name']}<br>"
                    f"<b>Top Artist:</b> {row['artist_name']}<br>"
                    f"<b>Country Code:</b> {row['region_code']} (ISO3: {row['region_code_iso3']})"
                )

            region_counts['hover_text'] = region_counts.apply(create_hover_text, axis=1)
            
            # Spotify-inspired color scale
            spotify_green_scale = ['#e1f5e8', '#b7e6c5', '#8dd6a3', '#63c781', '#39b85f', '#1DB954', '#189e47']
            color_scale = color_scale or spotify_green_scale

            # Create choropleth map
            fig = px.choropleth(
                region_counts,
                locations='region_code_iso3',
                locationmode='ISO-3',
                color='count',
                hover_name='hover_text',
                animation_frame='genre' if show_animation else None,
                projection=map_style,
                color_continuous_scale=color_scale,
                title=(
                    f"🌍 Global Music Genre Heatmap<br>"
                    f"<sup>Top {stats['total_countries']} countries | {stats['total_genres']} genres | "
                    f"Max: {stats['max_count']} | Avg: {stats['avg_count']}</sup>"
                ),
                labels={'count': 'Appearance Count'},
                custom_data=['hover_region_name', 'genre', 'count', 'track_name', 'artist_name', 'region_code', 'region_code_iso3']
            )

            # Enhanced layout for better visibility
            fig.update_layout(
                margin=dict(l=0, r=0, t=100, b=0),  # Increased top margin for title
                coloraxis_colorbar=dict(
                    title=dict(
                        text='<b>Appearance<br>Count</b>',
                        font=dict(size=12, family="Arial", color="#333333")
                    ),
                    thickness=20,  # Thicker colorbar
                    len=0.8,
                    yanchor="middle",
                    y=0.5,
                    x=1.05  # Move colorbar slightly right
                ),
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    landcolor='#f0f2f6',  # Lighter land color
                    subunitcolor='white',
                    countrycolor='#dddddd',  # Slightly visible country borders
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#e9ecef',
                    oceancolor='#e9ecef',
                    showcountries=True,
                    countrywidth=0.5
                ),
                height=600,  # Taller for better visibility
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", color='#333333'),
                title={
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='#111111', family="Arial", weight="bold")
                },
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial",
                    bordercolor='#ddd',
                    align="left"
                ),
                hovermode="closest",
                transition={'duration': 500} if show_animation else None
            )

            # Animation controls
            if show_animation:
                fig.layout.updatemenus = [dict(
                    type="buttons",
                    buttons=[dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 800, "redraw": True}, 
                            "fromcurrent": True, 
                            "transition": {"duration": 400}
                        }]
                    )],
                    x=0.1,
                    xanchor="right",
                    y=1.15,  # Higher position
                    yanchor="top",
                    pad=dict(t=0, l=10),
                    bgcolor="#1DB954",  # Spotify green
                    font=dict(color="white")
                )]
                
                for frame in fig.frames:
                    frame.layout.update(transition_duration=400)

            # Add location markers with custom icons
            if not show_animation:  # Only add markers if not in animation mode
                # Get coordinates for each country (using country centroids as fallback)
                # Note: In a real implementation, you should use actual coordinates from your data
                # Here we'll use a simple approach with country centroids
                
                # First get unique countries
                unique_countries = region_counts[['region_code_iso3', 'hover_region_name']].drop_duplicates()
                
                # Add scatter plot with icons for each country
                fig.add_trace(go.Scattergeo(
                    lon = [0] * len(unique_countries),  # Replace with actual longitudes
                    lat = [0] * len(unique_countries),  # Replace with actual latitudes
                    text = unique_countries['hover_region_name'],
                    customdata = unique_countries['region_code_iso3'],
                    hovertemplate = "<b>%{text}</b><extra></extra>",
                    mode = 'markers+text',
                    marker = dict(
                        size = 12,
                        symbol = 'music',  # Music note icon
                        color = '#FF5733',  # Orange color for visibility
                        opacity = 0.9,
                        line = dict(
                            width = 1,
                            color = 'white'
                        )
                    ),
                    textposition = "top center",
                    textfont = dict(
                        family = "Arial",
                        size = 10,
                        color = "#1DB954"  # Spotify green
                    ),
                    name = 'Music Locations'
                ))

                # Add another layer with custom emoji icons
                fig.add_trace(go.Scattergeo(
                    lon = [0] * len(unique_countries),  # Replace with actual longitudes
                    lat = [0] * len(unique_countries),  # Replace with actual latitudes
                    text = ["🎵"] * len(unique_countries),  # Music note emoji
                    textfont = dict(size=14),
                    mode = 'text',
                    hoverinfo = 'none',
                    showlegend = False
                ))

            # Footer annotation
            fig.add_annotation(
                x=0.5,
                y=0.02,
                text=(
                    f"Display threshold: ≥{min_count} appearances | "
                    f"Data from {len(df)} tracks | "
                    f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
                ),
                showarrow=False,
                font=dict(size=11, color="gray"),
                xref="paper",
                yref="paper"
            )

            # Visual enhancements
            fig.update_traces(
                marker_line_width=0.8,
                marker_line_color='rgba(255,255,255,0.8)',
                selector=dict(type='choropleth')
            )

            return fig

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")

            error_fig = go.Figure()
            error_fig.update_layout(
                title={
                    'text': f"<b>VISUALIZATION ERROR</b><br><sub>{str(e)}</sub>",
                    'y':0.5,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'middle',
                    'font': dict(size=16, color='red')
                },
                annotations=[
                    dict(
                        text="Please check your input data",
                        x=0.5,
                        y=0.4,
                        showarrow=False,
                        font=dict(size=14, color='gray'))
                ],
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return error_fig

    #Top Artists (Horizontal Bar Chart)
    def create_top_artists_bar(community_data):
        top_artists = community_data['artist_names'].value_counts().nlargest(5)

        MAX_LABEL_LENGTH = 18
        short_artist_names = [
            name if len(name) <= MAX_LABEL_LENGTH else name[:MAX_LABEL_LENGTH - 3] + "..."
            for name in top_artists.index
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=short_artist_names,
            x=top_artists.values,
            orientation='h',
            text=top_artists.values,
            textposition='outside',
            textangle=0,
            textfont=dict(color='black', size=12),
            marker=dict(
                color=top_artists.values,
                colorscale=[
                    [0.0, "#178A3E"],
                    [0.2, "#3AC86C"],
                    [0.5, "#42E079"],
                    [0.8, "#37F57A"],
                    [1.0, "#B2F2C9"]
                ],
                cmin=0,
                cmax=top_artists.max(),
                line=dict(width=0),
                cornerradius=15
            ),
            width=0.4,
            hoverinfo='text',
            hovertext=[
                f"<b>{artist}</b><br>"
                f"Number of songs: {count}"
                for artist, count in zip(top_artists.index, top_artists.values)
            ]
        ))

        fig.update_layout(
            title=dict(
                text='<b>TOP 5 ARTISTS IN COMMUNITY</b>',
                font=dict(size=18, color='black', family='Montserrat'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='<b>Number of Songs</b>',
                title_font=dict(size=12, color='black'),
                tickfont=dict(size=11, color='black'),
                gridcolor='rgba(0,0,0,0.1)',
                zerolinecolor='rgba(0,0,0,0.3)'
            ),
            yaxis=dict(
                title=None,
                tickfont=dict(size=12, color='black')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black', size=11, family="Montserrat"),
            height=350,
            margin=dict(l=120, r=20, t=80, b=40),
            showlegend=False,
            hoverlabel=dict(
                bgcolor='rgba(29,185,84,0.8)',
                font_size=13,
                font_family="Montserrat"
            )
        )

        return fig
    #Song Recommendations Cards
    def get_image_from_tunebat(url_tunebat):
        try:
            if not url_tunebat or pd.isna(url_tunebat) or str(url_tunebat).strip() == "":
                raise ValueError("URL Tunebat không hợp lệ")
            
            print(f"Đang truy cập URL: {url_tunebat}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            # Thêm timeout và retry
            for _ in range(2):  # Thử tối đa 2 lần
                try:
                    response = requests.get(url_tunebat, headers=headers, timeout=10)
                    print(f"Status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Cách 1: Tìm meta og:image
                        og_image = soup.find('meta', property='og:image')
                        if og_image and og_image.get('content'):
                            image_url = og_image['content']
                            print(f"Tìm thấy ảnh qua og:image: {image_url}")
                            return image_url
                        
                        # Cách 2: Tìm thẻ img có class đặc biệt
                        img_tag = soup.find('img', {'class': 'track-cover-image'})
                        if img_tag and img_tag.get('src'):
                            image_url = img_tag['src']
                            print(f"Tìm thấy ảnh qua img tag: {image_url}")
                            return image_url
                        
                        # Cách 3: Tìm trong JSON-LD
                        script_tag = soup.find('script', {'type': 'application/ld+json'})
                        if script_tag:
                            try:
                                json_data = json.loads(script_tag.string)
                                if json_data.get('image'):
                                    print(f"Tìm thấy ảnh qua JSON-LD: {json_data['image']}")
                                    return json_data['image']
                            except json.JSONDecodeError:
                                pass
                        
                        print("Không tìm thấy ảnh trong trang")
                        break
                    
                    elif response.status_code == 403:
                        print("Bị chặn truy cập, thử thay đổi User-Agent...")
                        headers["User-Agent"] = random.choice([
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
                        ])
                        continue
                    
                except requests.exceptions.RequestException as e:
                    print(f"Lỗi request: {e}")
                    time.sleep(1)  # Đợi 1 giây trước khi thử lại
        
        except Exception as e:
            print(f"Lỗi khi lấy ảnh từ Tunebat: {e}")
        
        print("→ Trả về ảnh mặc định.")
        return "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228"
    def create_song_recommendations(community_data, current_index=0, n_recommendations=10):
    # Sắp xếp và lấy danh sách đề xuất
        recommendations = community_data.sort_values(
            by=['popularity', 'valence', 'energy'],
            ascending=[False, False, False]
        ).head(n_recommendations)

        # Lấy bài hát hiện tại dựa trên index
        row = recommendations.iloc[current_index % len(recommendations)]

        # Xử lý URL và hình ảnh
        url_tunebat = row.get('url_tunebat', '')
        track_id = url_tunebat.strip().split('/')[-1] if url_tunebat and '/' in url_tunebat else ''
        spotify_url = f"https://open.spotify.com/track/{track_id}" if track_id else "#"

        # 🔄 DÙNG spotify_cover_url thay vì image_url
        image_url = row.get('spotify_cover_url', '')
        if not image_url or pd.isna(image_url) or str(image_url).strip() == "":
            image_url = "https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228"  # fallback ảnh

        # Xử lý nội dung
        track_name = row.get('track_name', 'Không rõ')[:40]
        artist_name = row.get('artist_names', 'Không rõ')[:40]
        genre = row.get('genre', 'Không rõ')[:25]
        popularity = row.get('popularity', 0)

        # Tạo card
        card = dbc.Card(
    [
        dbc.CardImg(
            src=image_url,
            top=True,
            style={
                'height': '160px',  # 🔽 Giảm chiều cao ảnh
                'objectFit': 'cover',
                'borderRadius': '12px 12px 0 0'
            }
        ),
        dbc.CardBody([
            html.H4(track_name, style={
                'fontWeight': 'bold',
                'fontSize': '1em',  # 🔽 Nhỏ hơn
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign': 'center',
                'marginBottom': '4px'  # 🔽 Thu hẹp
            }),
            html.H5(artist_name, style={
                'fontSize': '0.9em',
                'color': '#666',
                'marginBottom': '4px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign': 'center'
            }),
            html.P(f"Thể loại: {genre}", style={
                'fontSize': '0.8em',
                'color': '#888',
                'marginBottom': '4px',
                'textAlign': 'center'
            }),
            html.P(f"Phổ biến: {popularity}/100", style={
                'fontSize': '0.8em',
                'color': '#1DB954',
                'textAlign': 'center',
                'marginBottom': '4px'
            }),
            html.A(
                "▶ Nghe trên Spotify",
                href=spotify_url,
                target="_blank",
                style={
                    'display': 'block',
                    'textAlign': 'center',
                    'padding': '6px',
                    'fontSize': '0.8em',
                    'color': 'white',
                    'background': '#1DB954',
                    'borderRadius': '6px',
                    'textDecoration': 'none',
                    'marginTop': '8px'
                }
            )
        ])
    ],
    style={
        'width': '200px',
        'height': '100px', 
        'margin': '10px auto',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
        'overflow': 'hidden'
    }
)
        card = dbc.Card(
            [
                dbc.CardImg(
                    src=image_url,
                    top=True,
                    style={
                        'height': '180px',  # 🔽 Giảm chiều cao ảnh
                        'objectFit': 'cover',
                        'borderRadius': '12px 12px 0 0'
                    }
                ),
                dbc.CardBody([
                    html.H4(track_name, style={
                        'fontWeight': 'bold',
                        'fontSize': '1em',  # 🔽 Nhỏ hơn
                        'whiteSpace': 'nowrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'textAlign': 'center',
                        'marginBottom': '4px'  # 🔽 Thu hẹp
                    }),
                    html.H5(artist_name, style={
                        'fontSize': '0.9em',
                        'color': '#666',
                        'marginBottom': '4px',
                        'whiteSpace': 'nowrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'textAlign': 'center'
                    }),
                    html.P(f"Thể loại: {genre}", style={
                        'fontSize': '0.8em',
                        'color': '#888',
                        'marginBottom': '4px',
                        'textAlign': 'center'
                    }),
                    html.P(f"Phổ biến: {popularity}/100", style={
                        'fontSize': '0.8em',
                        'color': '#1DB954',
                        'textAlign': 'center',
                        'marginBottom': '4px'
                    }),
                    html.A(
                        "▶ Nghe trên Spotify",
                        href=spotify_url,
                        target="_blank",
                        style={
                            'display': 'block',
                            'textAlign': 'center',
                            'padding': '6px',
                            'fontSize': '0.8em',
                            'color': 'white',
                            'background': '#1DB954',
                            'borderRadius': '6px',
                            'textDecoration': 'none',
                            'marginTop': '8px'
                        }
                    )
                ])
            ],
            style={
                'width': '200px',
                'height': '350px',  # 🔒 Khóa tổng chiều cao
                'margin': '10px auto',
                'borderRadius': '12px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'overflow': 'hidden'
            }
        )


        return card

    # 1. Tạo dữ liệu giả lập ban đầu
    df_songs, G, partition = generate_spotify_data()

    # 2. Định nghĩa callback để cập nhật các biểu đồ
    # 1. Thêm Store component để lưu trạng thái index
    dcc.Store(id='recommendation-index-store', data=0)

    # 2. Sửa lại callback để bao gồm nút điều hướng
    @app.callback(
        [
            #Output('distribution-by-channel', 'figure'),
            #Output('density-chart', 'figure'),
            Output('community-distribution-map', 'figure'),
            Output('community-top-artists', 'figure'),
            Output('community-song-recommendations', 'children'),
            Output('recommendation-index-store', 'data'),
            Output('lambda-mean-display', 'children'),
            Output('total-songs-display', 'children')
        ],
        [
            Input('community-dropdown', 'value'),
            Input('recommendation-prev-button', 'n_clicks'),
            Input('recommendation-next-button', 'n_clicks')
        ],
        [State('recommendation-index-store', 'data')]
    )
    def update_community_analysis(selected_community, prev_clicks, next_clicks, current_index):
        ctx = dash.callback_context

        # Xử lý khi click nút Previous/Next
        if ctx.triggered and ctx.triggered[0]['prop_id'] != 'community-dropdown.value':
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'recommendation-prev-button':
                current_index = max(0, current_index - 1)
            elif button_id == 'recommendation-next-button':
                current_index += 1

        print(f"Đang cập nhật biểu đồ cho genre: {selected_community}")
        communities = df_songs['genre'].unique()

        if selected_community not in communities:
            print(f"Genre {selected_community} không tồn tại, chọn genre mặc định.")
            selected_community = sorted(communities)[0]

        try:
            # Lọc dữ liệu theo genre được chọn
            community_data = df_songs[df_songs['genre'] == selected_community].copy()

            # Kiểm tra dữ liệu trống
            if community_data.empty:
                raise ValueError(f"Không có dữ liệu cho genre {selected_community}")

            # Xử lý NaN cho các cột thường dùng
            num_cols = ['popularity', 'valence', 'danceability', 'energy',
                        'acousticness', 'instrumentalness', 'speechiness', 'liveness']
            for col in num_cols:
                if col in community_data.columns:
                    community_data[col] = community_data[col].fillna(community_data[col].mean())

            community_data['track_name'] = community_data['track_name'].fillna("Unknown Track")
            if 'artist_names' in community_data.columns:
                community_data['artist_names'] = community_data['artist_names'].fillna("Unknown Artist")

            viral_track = community_data.sort_values('popularity', ascending=False)['track_name'].iloc[0]

            # Tạo các biểu đồ
            #radar_fig = create_audio_features_radar(community_data, [selected_community])
            #sankey_fig = create_sankey_community(community_data, viral_track)
            bar_fig = create_genre_distribution_map(community_data)
            artists_fig = create_top_artists_bar(community_data)
            recommendations = create_song_recommendations(community_data, current_index)

            # Tính toán Lambda trung bình và số lượng bài hát
            valid_lambda = community_data['Lambda'].dropna()
            lambda_mean = f"{valid_lambda.mean():.3f}" if not valid_lambda.empty else "0.000"
            total_songs = str(len(valid_lambda))

            return (
                bar_fig,
                artists_fig,
                recommendations,
                current_index,
                lambda_mean,
                total_songs
            )

        except Exception as e:
            print(f"Lỗi khi cập nhật biểu đồ: {str(e)}")

            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor=SPOTIFY_COLORS['black'],
                paper_bgcolor=SPOTIFY_COLORS['black'],
                title=f"Lỗi: {str(e)}",
                font=dict(color='black')
            )
            return (
                empty_fig, empty_fig, empty_fig, empty_fig,
                [], current_index,
                "0.000", "0"
            )

   #------------------Tab3-----------------------------
    @app.callback(
        [Output('song-risk-dropdown', 'options'),
        Output('center-artists', 'options')],
        [Input('song-risk-dropdown', 'value'),
        Input('center-artists', 'value')]
    )
    def update_all_dropdowns(song_value, artist_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        df_filtered = df_songs.copy()

        if triggered_id == 'song-risk-dropdown' and song_value:
            df_filtered = df_filtered[df_filtered['track_name'].isin(song_value if isinstance(song_value, list) else [song_value])]
        elif triggered_id == 'center-artists' and artist_value:
            df_filtered = df_filtered[df_filtered['artist_names'].isin(artist_value if isinstance(artist_value, list) else [artist_value])]

        # Lấy danh sách lọc
        song_options = [{'label': s, 'value': s} for s in sorted(df_filtered['track_name'].unique())]
        artist_options = [{'label': a, 'value': a} for a in sorted(df_filtered['artist_names'].unique())]
        return song_options, artist_options

    
    @app.callback(
        Output('sensitivity-network', 'figure'),
        Input('network-view-toggle', 'value'),
        Input('tabs', 'active_tab'),
        Input('center-artists', 'value'),
        Input('song-risk-dropdown', 'value')
    )
    def draw_crisis_network(view_mode, tab, selected_artists, selected_songs):
        df_songs, G, partition = generate_spotify_data()

        if tab != 'risk' or not selected_songs:
            return go.Figure()

        if view_mode == 'before':
            return draw_original_network_for_comparison()

        # Xử lý dữ liệu
        df_songs['loudness'] = df_songs['loudness'].astype(str).str.replace('db', '', case=False).str.replace('−', '-', regex=False).str.replace(',', '.').str.strip().astype(float)

        if isinstance(selected_artists, str):
            selected_artists = [selected_artists]
        if isinstance(selected_songs, str):
            selected_songs = [selected_songs]

        removed_song = selected_songs[0]
        removed_artists = []
        H = G.copy()
        removed_node_info = {}
        removed_edges_info = []

        # Lấy thông tin node sẽ xóa
        if removed_song in H.nodes:
            if removed_song in df_songs['track_name'].values:
                artists_str = df_songs[df_songs['track_name'] == removed_song]['artist_names'].values
                if len(artists_str) > 0:
                    removed_artists = [a.strip() for a in artists_str[0].split(',')]

            removed_node_info = {
                'name': removed_song,
                'genre': df_songs[df_songs['track_name'] == removed_song]['genre'].values[0] if not df_songs[df_songs['track_name'] == removed_song].empty else 'Other',
                'artist': removed_artists[0] if removed_artists else '',
                'popularity': df_songs[df_songs['track_name'] == removed_song]['popularity'].values[0] if not df_songs[df_songs['track_name'] == removed_song].empty else 0,
                'lambda': df_songs[df_songs['track_name'] == removed_song]['Lambda'].values[0] if not df_songs[df_songs['track_name'] == removed_song].empty else 0
            }

            # Lưu thông tin các cạnh sẽ bị xóa
            for neighbor in H.neighbors(removed_song):
                removed_edges_info.append({
                    'source': removed_song,
                    'target': neighbor,
                    'color': '#1DB954' if neighbor in removed_artists else '#FFD700'  # Màu vàng cho cạnh sẽ xóa
                })
            track_names = df_songs['track_name'].tolist()
            removed_song = selected_songs[0]

            if removed_song in track_names:
                idx_remove = track_names.index(removed_song)
                print("[DEBUG] simulate_node_removal_impact được gọi TRƯỚC khi xoá node thật", flush=True)

                # Chuẩn hóa dữ liệu đầu vào
                feature_cols = ['energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness',
                                'liveness', 'speechiness', 'loudness', 'artist_total_followers',
                                'playlist_num_followers', 'BPM']
                df_songs[feature_cols] = df_songs[feature_cols].fillna(df_songs[feature_cols].median())
                X_raw = df_songs[feature_cols].copy()
                scaler = StandardScaler()
                X = scaler.fit_transform(X_raw)
                Y = df_songs['track_popularity'].values

                genre_ids = df_songs['genre'].astype('category').cat.codes.values
                genre_matrix = (genre_ids[:, None] == genre_ids[None, :])
                np.fill_diagonal(genre_matrix, False)
                W = genre_matrix.astype(float)
                W = W / (W.sum(axis=1, keepdims=True) + 1e-10)

                # Load mô hình đã huấn luyện
                with open('cim_model.pkl', 'rb') as f:
                    saved_model = pickle.load(f)

                print("[LOG] Gọi simulate_node_removal_impact...", flush=True)
                start_time = time.time()
                lambda_new = simulate_node_removal_impact(idx_remove, X, Y, W, saved_model)
                print(f"[LOG] Tính lambda xong sau {time.time() - start_time:.2f} giây", flush=True)

                # Tạo dict Lambda mới
                track_names_wo_removed = [node for node in track_names if node != removed_song]
                lambda_dict = {
                    node: lambda_new[i]
                    for i, node in enumerate(track_names_wo_removed)
                    if not np.isnan(lambda_new[i])
                }

                print("[DEBUG] lambda_new[:5]:", list(lambda_dict.items())[:5], flush=True)
            else:
                print(f"[WARNING] removed_song '{removed_song}' không có trong track_names!", flush=True)

            # Fallback nếu lambda_dict rỗng
            if not lambda_dict:
                print("[DEBUG] Fallback: gán lambda = 0.1 cho toàn bộ node", flush=True)
                lambda_dict = {node: 0.1 for node in H.nodes}

        ## Tạo layout đặc biệt để gom nhóm theo thể loại
        # Tạo dict ánh xạ node -> thể loại và lambda
        node_genre = {}
        for node in G.nodes():
            genre = df_songs[df_songs['track_name'] == node]['genre'].values
            node_genre[node] = genre[0] if len(genre) > 0 else 'Other'

        # Gán lambda mới cho các node còn lại
        node_lambda = {}
        for node in H.nodes():
            if node in lambda_dict:
                node_lambda[node] = lambda_dict[node]
            else:
                node_lambda[node] = 0.1  # fallback
        
        # Tạo positions ban đầu với spring layout
        pos = nx.kamada_kawai_layout(H)
        
        # Gom node theo thể loại
        node_genre = {}
        genre_centers = {}
        genre_nodes = {}

        for node in H.nodes():
            genre = df_songs[df_songs['track_name'] == node]['genre'].values
            genre = genre[0] if len(genre) > 0 else 'Other'
            node_genre[node] = genre

            x, y = pos[node]
            if genre not in genre_centers:
                genre_centers[genre] = [x, y, 1]
                genre_nodes[genre] = [node]
            else:
                genre_centers[genre][0] += x
                genre_centers[genre][1] += y
                genre_centers[genre][2] += 1
                genre_nodes[genre].append(node)

        for genre in genre_centers:
            genre_centers[genre][0] /= genre_centers[genre][2]
            genre_centers[genre][1] /= genre_centers[genre][2]

        alpha = 0.2
        for genre, nodes in genre_nodes.items():
            cx, cy, _ = genre_centers[genre]
            for node in nodes:
                pos[node][0] = (1 - alpha) * pos[node][0] + alpha * cx
                pos[node][1] = (1 - alpha) * pos[node][1] + alpha * cy
        # Lưu vị trí node sẽ xóa
        if removed_song in pos:
            removed_node_info['pos'] = pos[removed_song]
            
            # Cập nhật vị trí target cho các cạnh bị xóa
            for edge in removed_edges_info:
                edge['target_pos'] = pos[edge['target']]
                edge['source_pos'] = pos[removed_song]

        # Thực hiện xóa node sau khi đã lưu thông tin
        if removed_song in H.nodes:
            H.remove_node(removed_song)
    
        # Tạo bảng màu theo thể loại
        genre_list = df_songs['genre'].dropna().unique().tolist()
        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Dark24
        genre_color_map = {g: color_palette[i % len(color_palette)] for i, g in enumerate(genre_list)}

        # Tính toán phạm vi giá trị Lambda để chuẩn hóa kích thước node
        lambda_values = list(node_lambda.values())
        min_lambda = min(lambda_values) if lambda_values else 0
        max_lambda = max(lambda_values) if lambda_values else 1
        
        # Hàm chuẩn hóa kích thước node theo Lambda (từ 10 đến 40)
        def normalize_size(lambda_val):
            if max_lambda == min_lambda:
                return 25  # Giá trị mặc định nếu tất cả Lambda bằng nhau
            return 10 + 30 * ((lambda_val - min_lambda) / (max_lambda - min_lambda))

        # Tạo animation frames
        frames = []

        ## Frame 1: Trạng thái ban đầu với node sẽ xóa
        edge_traces_init = []
        node_x_init, node_y_init, node_color_init, node_size_init = [], [], [], []
        
        # Vẽ tất cả các cạnh (mờ)
        for u, v, data in H.edges(data=True):
            edge_traces_init.append(go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
                hoverinfo='none',
                mode='lines'
            ))
        
        # Vẽ các cạnh sẽ bị xóa (nổi bật)
        for edge in removed_edges_info:
            edge_traces_init.append(go.Scatter(
                x=[edge['source_pos'][0], edge['target_pos'][0]],
                y=[edge['source_pos'][1], edge['target_pos'][1]],
                line=dict(width=3, color=edge['color']),
                hoverinfo='none',
                mode='lines'
            ))
        
        # Vẽ tất cả các node (mờ)
        for node in G.nodes():
            if node != removed_song:
                node_x_init.append(pos[node][0])
                node_y_init.append(pos[node][1])
                genre = node_genre[node]
                node_color_init.append(f"rgba({int(genre_color_map[genre][1:3],16)},{int(genre_color_map[genre][3:5],16)},{int(genre_color_map[genre][5:7],16)},0.5)")
                node_size_init.append(normalize_size(node_lambda[node]) * 0.7)  # Giảm kích thước cho các node mờ
        
        # Thêm node sẽ xóa (nổi bật)
        if removed_node_info.get('pos') is not None:

            node_x_init.append(removed_node_info['pos'][0])
            node_y_init.append(removed_node_info['pos'][1])
            node_color_init.append('red')
            node_size_init.append(normalize_size(removed_node_info['lambda']) * 1.5)  # Tăng kích thước node sẽ xóa
        
        frames.append(go.Frame(
            data=[
                *edge_traces_init,
                go.Scatter(
                    x=node_x_init,
                    y=node_y_init,
                    mode='markers',
                    marker=dict(
                        color=node_color_init,
                        size=node_size_init,
                        line=dict(width=1, color='rgba(255,255,255,0.8)'),
                        opacity=0.9
                    ),
                    hoverinfo='none'
                )
            ],
            name="initial"
        ))

        ## Frame 2-4: Hiệu ứng xóa node và cạnh
        for progress in [0.3, 0.6, 0.9]:
            edge_traces = []
            node_x, node_y, node_color, node_size = [], [], [], []
            
            # Vẽ các cạnh thường (mờ)
            for u, v, data in H.edges(data=True):
                edge_traces.append(go.Scatter(
                    x=[pos[u][0], pos[v][0], None],
                    y=[pos[u][1], pos[v][1], None],
                    line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
                    hoverinfo='none',
                    mode='lines'
                ))
            
            # Vẽ các cạnh đang "biến mất"
            for edge in removed_edges_info:
                edge_traces.append(go.Scatter(
                    x=[edge['source_pos'][0], edge['target_pos'][0]],
                    y=[edge['source_pos'][1], edge['target_pos'][1]],
                    line=dict(width=3*(1-progress), color=edge['color']),
                    hoverinfo='none',
                    mode='lines'
                ))
            
            # Vẽ các node thường
            for node in H.nodes():
                node_x.append(pos[node][0])
                node_y.append(pos[node][1])
                genre = node_genre[node]
                node_color.append(f"rgba({int(genre_color_map[genre][1:3],16)},{int(genre_color_map[genre][3:5],16)},{int(genre_color_map[genre][5:7],16)},0.5)")
                node_size.append(normalize_size(node_lambda[node]) * 0.7)
            
            # Vẽ node đang "biến mất"
            if removed_node_info.get('pos') is not None:

                node_x.append(removed_node_info['pos'][0])
                node_y.append(removed_node_info['pos'][1])
                node_color.append(f'rgba(255,0,0,{1-progress})')
                node_size.append(normalize_size(removed_node_info['lambda']) * 1.5 * (1-progress))
            
            frames.append(go.Frame(
                data=[
                    *edge_traces,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers',
                        marker=dict(
                            color=node_color,
                            size=node_size,
                            line=dict(width=1, color='rgba(255,255,255,0.8)'),
                            opacity=0.9
                        ),
                        hoverinfo='none'
                    )
                ],
                name=f"disappearing_{progress}"
            ))

        ## Frame cuối: Trạng thái sau khi xóa
        edge_traces_final = []
        node_x_final, node_y_final, node_text_final, node_color_final, node_size_final = [], [], [], [], []
        
        # Vẽ các cạnh
        for u, v, data in H.edges(data=True):
            edge_traces_final.append(go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            ))
        
        # Vẽ các node với thông tin đầy đủ
        for node in H.nodes():
            node_x_final.append(pos[node][0])
            node_y_final.append(pos[node][1])
            genre = node_genre[node]
            artist = df_songs[df_songs['track_name'] == node]['artist_names'].values[0] if not df_songs[df_songs['track_name'] == node].empty else ''
            popularity = df_songs[df_songs['track_name'] == node]['popularity'].values[0] if not df_songs[df_songs['track_name'] == node].empty else 0
            lambda_val = node_lambda[node]
            
            node_color_final.append(genre_color_map.get(genre, '#CCCCCC'))
            node_size_final.append(normalize_size(lambda_val))
            node_text_final.append(
                f"<b>{node}</b><br>"
                f"Thể loại: {genre}<br>"
                f"Nghệ sĩ: {artist}<br>"
                f"Độ phổ biến: {popularity}<br>"
                f"Lambda: {lambda_val:.3f}"
            )
        
        # Thêm highlight các node từng nối với node bị xóa
        neighbor_nodes = [edge['target'] for edge in removed_edges_info]
        for node in neighbor_nodes:
            if node in pos:  # Đảm bảo node vẫn còn trong đồ thị
                idx = list(H.nodes()).index(node)
                node_size_final[idx] = normalize_size(node_lambda[node]) * 1.3  # Tăng kích thước
                node_color_final[idx] = '#FFA500'  # Màu cam
        
        frames.append(go.Frame(
            data=[
                *edge_traces_final,
                go.Scatter(
                    x=node_x_final,
                    y=node_y_final,
                    mode='markers',
                    marker=dict(
                        color=node_color_final,
                        size=node_size_final,
                        line=dict(width=1.5, color='rgba(255,255,255,0.8)'),
                        opacity=0.95
                    ),
                    hoverinfo='text',
                    hovertext=node_text_final
                )
            ],
            name="final"
        ))

        # Tính toán phạm vi zoom tập trung vào node bị xóa và các node lân cận
        if removed_node_info.get('pos') is not None:

            neighbor_positions = [pos[edge['target']] for edge in removed_edges_info if edge['target'] in pos]
            all_x = [removed_node_info['pos'][0]] + [p[0] for p in neighbor_positions]
            all_y = [removed_node_info['pos'][1]] + [p[1] for p in neighbor_positions]
            
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            # Thêm padding
            x_padding = max((x_max - x_min) * 0.3, 0.2)  # Đảm bảo không zoom quá gần
            y_padding = max((y_max - y_min) * 0.3, 0.2)
            
            x_range = [x_min - x_padding, x_max + x_padding]
            y_range = [y_min - y_padding, y_max + y_padding]
        else:
            x_range = [-1.1, 1.1]
            y_range = [-1.1, 1.1]

        # Tạo figure
        fig = go.Figure(
            data=frames[0]['data'],
            frames=frames[1:],
            layout=go.Layout(
                title=dict(
                    text=f'<b>QUÁ TRÌNH XOÁ: {removed_song.upper()}</b><br><sup>Kích thước node thể hiện giá trị Lambda</sup>',
                    font=dict(size=16, family="Montserrat", color='black'),
                    x=0.05,
                    y=0.95,
                    xanchor='left'
                ),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=x_range
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=y_range
                ),
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=100),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶️ Play",
                            "method": "animate",
                            "args": [
                                None, 
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, 
                                    "transition": {"duration": 300}
                                }
                            ]
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [
                                [None], 
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ]
                        },
                        {
                            "label": "🔄 Reset",
                            "method": "animate",
                            "args": [
                                [None], 
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ]
                        }
                    ],
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
                    "yanchor": "top"
                }],
                annotations=[
                    dict(
                        x=0.5,
                        y=0.02,
                        xref='paper',
                        yref='paper',
                        text=f"Đang xóa: {removed_song} (Thể loại: {removed_node_info.get('genre', 'Unknown')}, Lambda: {removed_node_info.get('lambda', 0):.3f}",
                        showarrow=False,
                        font=dict(size=12, color='red'),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='red',
                        borderwidth=1,
                        borderpad=4
                    ),
                    dict(
                        x=0.5,
                        y=0.95,
                        xref='paper',
                        yref='paper',
                        text="Các node màu cam từng kết nối với node bị xóa",
                        showarrow=False,
                        font=dict(size=12, color='#FFA500'),
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='#FFA500',
                        borderwidth=1,
                        borderpad=4
                    )
                ]
            )
        )

        return fig
    
    @app.callback(
    Output('community-impact-bar', 'figure'),
    [Input('tabs', 'active_tab'),
     Input('center-artists', 'value'),
     Input('song-risk-dropdown', 'value')]
    )
    def update_community_impact(tab, selected_artists, selected_songs):
        if tab != 'risk':
            return empty_figure()
        
        # Tính ảnh hưởng trung bình theo thể loại
        genre_impact = df_songs.groupby('genre')['Lambda'].mean().reset_index()
        genre_impact = genre_impact.sort_values('Lambda', ascending=False).reset_index(drop=True)
        
        # Chuẩn hóa giá trị Lambda để ánh xạ màu
        norm = (genre_impact['Lambda'] - genre_impact['Lambda'].min()) / (genre_impact['Lambda'].max() - genre_impact['Lambda'].min() + 1e-6)
        
        # Dùng dải màu xanh (Tealgrn hoặc Greens)
        colors = px.colors.sample_colorscale('Tealgrn', norm)

        # Tạo biểu đồ
        fig = go.Figure(go.Bar(
            x=genre_impact['Lambda'],
            y=genre_impact['genre'],
            orientation='h',
            marker=dict(color=colors),
            text=genre_impact['Lambda'].round(2),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Mean λ: %{x:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text="Ảnh hưởng theo thể loại",
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Montserrat', color='black')
            ),
            xaxis_title="Điểm ảnh hưởng trung bình (λ)",
            yaxis_title="Thể loại",
            xaxis=dict(color='black'),
            yaxis=dict(color='black'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Montserrat', color='black'),
            height=400,
            margin=dict(l=100, r=20, t=60, b=40)
        )

        return fig
    @app.callback(
    Output('influence-heatmap', 'figure'),
    [Input('tabs', 'active_tab'),
     Input('center-artists', 'value'),
     Input('song-risk-dropdown', 'value')]
    )
    def draw_influence_heatmap(tab, selected_artists, selected_songs):
        if tab != 'risk':
            return go.Figure()

        # Lấy dữ liệu dựa trên selection
        df_filtered = df_songs.copy()
        
        if selected_artists or selected_songs:
            conditions = []
            if selected_artists:
                if isinstance(selected_artists, str):
                    selected_artists = [selected_artists]
                conditions.append(df_filtered['artist_names'].isin(selected_artists))
            if selected_songs:
                if isinstance(selected_songs, str):
                    selected_songs = [selected_songs]
                conditions.append(df_filtered['track_name'].isin(selected_songs))
            
            df_filtered = df_filtered[np.logical_or.reduce(conditions)]
        
        # Tính toán influence theo genre và artist
        genre_artist_impact = df_filtered.pivot_table(
            values='Lambda',
            index='genre',
            columns='artist_names',
            aggfunc='mean',
            fill_value=0
        )
        
        # Chuẩn hóa dữ liệu
        genre_artist_impact = genre_artist_impact.div(
            genre_artist_impact.max(axis=1), axis=0)
        
        fig = px.imshow(
            genre_artist_impact,
            color_continuous_scale='Tealgrn',
            labels=dict(x="Nghệ sĩ", y="Thể loại", color="Ảnh hưởng"),
            aspect="auto"
        )
        
        fig.update_layout(
            title={
                'text': 'BẢN ĐỒ NHIỆT ẢNH HƯỞNG NGHỆ SĨ-THỂ LOẠI',
                'font': {'family': 'Montserrat', 'size': 18},
                'x': 0.5
            },
            xaxis_title="Nghệ sĩ",
            yaxis_title="Thể loại",
            height=400,
            hovermode='closest',
            font=dict(family='Montserrat')
        )
        
        fig.update_traces(
            hovertemplate=(
                "<b>Thể loại</b>: %{y}<br>"
                "<b>Nghệ sĩ</b>: %{x}<br>"
                "<b>Điểm ảnh hưởng</b>: %{z:.2f}<extra></extra>"
            )
        )
        
        return fig

    @app.callback(
        [Output('risk-prediction', 'figure'),
        Output('high-risk-songs', 'data')],
        [Input('tabs', 'active_tab'),
        Input('center-artists', 'value'),
        Input('song-risk-dropdown', 'value')]
    )
    def update_propagation_timeline(tab, selected_artists, selected_songs):
        if tab != 'risk':
            return empty_figure(), []

        if not selected_artists and not selected_songs:
            return empty_figure("Vui lòng chọn nghệ sĩ hoặc bài hát"), []

        df, G, _ = generate_spotify_data()
        propagation_data = get_propagation_data(selected_artists, selected_songs, df, G)

        if not propagation_data or len(propagation_data['nodes']) == 0:
            return empty_figure("Không có dữ liệu lan truyền"), []

        # Tạo palette màu Spotify-inspired
        spotify_colors = {
            'background': '#ffffff',
            'primary': '#1DB954',       # Màu xanh lá Spotify
            'secondary': '#111111',     # Chữ đậm
            'accent': '#1ED760',        # Xanh sáng
            'neutral': '#4d4d4d',       # Xám đậm hơn
            'edge': 'rgba(30, 215, 96, 0.2)'
        }

        # Tạo figure
        fig = go.Figure()

        # Thêm edges với màu mới
        for edge in propagation_data['edges']:
            fig.add_trace(go.Scatter(
                x=[edge['source_time'], edge['target_time']],
                y=[edge['source'], edge['target']],
                mode='lines',
                line=dict(width=edge['width'], color=spotify_colors['edge']),
                hoverinfo='none',
                showlegend=False
            ))

        # Thêm nodes với màu gradient dựa trên giá trị
        for node in propagation_data['nodes']:
            # Tính màu gradient từ giá trị (value) của node
            value_norm = node['value'] / max(n['value'] for n in propagation_data['nodes'])
            base_color = spotify_colors['primary']
            # Tạo màu gradient từ xanh đậm đến xanh sáng
            node_color = f'rgba(29, 185, 84, {0.5 + 0.5*value_norm})'
            
            fig.add_trace(go.Scatter(
                x=[node['time']],
                y=[node['id']],
                mode='markers+text',
                marker=dict(
                    size=node['size'],
                    color=node_color,
                    line=dict(width=1.5, color=spotify_colors['secondary'])
                ),
                name=node['full_label'],
                text=node['label'],
                textposition="top center",
                hoverinfo='text',
                hovertext=f"""
                <b>{node['full_label']}</b><br>
                Nghệ sĩ: {node['artist']}<br>
                Thể loại: {node['genre']}<br>
                Độ phổ biến: {node['popularity']}<br>
                Giá trị Lambda: {node['value']:.2f}<br>
                Thời điểm ảnh hưởng: {node['time']}
                """,
                showlegend=False
            ))

        # Cập nhật layout với theme mới
        fig.update_layout(
        title='TIMELINE LAN TRUYỀN ẢNH HƯỞNG',
        xaxis=dict(
            title=dict(
                text='Thời gian (bước lan truyền)',
                font=dict(color=spotify_colors['secondary'])
            ),
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, max([n['time'] for n in propagation_data['nodes']]) + 0.5],
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color=spotify_colors['neutral'])
        ),
        yaxis=dict(
            title=dict(
                text='Bài hát',
                font=dict(color=spotify_colors['secondary'])
            ),
            tickmode='array',
            tickvals=list(range(len(propagation_data['nodes']))),
            ticktext=[n['label'] for n in propagation_data['nodes']],
            range=[-1, len(propagation_data['nodes'])],
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(color=spotify_colors['neutral'])
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',     # Nền plot trong suốt
        paper_bgcolor='rgba(0,0,0,0)',    # Nền tổng thể trong suốt
        height=450,
        margin=dict(l=100, r=50, t=80, b=50),
        font=dict(family="Montserrat", color=spotify_colors['secondary']),
        title_font=dict(color=spotify_colors['primary'], size=18)
    )

        # Tạo bảng dữ liệu liên quan
        table_data = [
            {
                'track_name': node['label'],
                'artist_names': node['artist'],
                'genre': node['genre'],
                'influence': node['value']
            }
            for node in propagation_data['nodes']
        ]

        # Sort theo điểm ảnh hưởng giảm dần
        table_data = sorted(table_data, key=lambda x: x['influence'], reverse=True)

        return fig, table_data


    # Hàm tạo figure rỗng
    def empty_figure(message="Không có dữ liệu để hiển thị"):
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text=message,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ],
            plot_bgcolor='white',
            height=400
        )
        return fig



