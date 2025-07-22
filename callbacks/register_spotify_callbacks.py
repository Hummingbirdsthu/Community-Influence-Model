from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash
import requests
from bs4 import BeautifulSoup
import numpy as np
import networkx as nx
import community as community_louvain
from datetime import datetime
import pytz
import base64
import plotly.graph_objects as go
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import random
import string
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

def generate_spotify_data():
    df = pd.read_csv("data/merged_with_lambda.csv")
    # Nếu thiếu valence thì tạo random
    if 'valence' not in df.columns:
        df['valence'] = np.random.uniform(0, 1, len(df))
    # Mã hóa ảnh nếu cần
    if 'image_base64' not in df.columns and 'image_url' in df.columns:
        df['image_base64'] = df['image_url'].apply(image_to_base64)

    # Tạo graph G
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
        tracks = df[df[artist_col] == artist]['track_name'].tolist()
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                G.add_edge(tracks[i], tracks[j], connection='artist')

    # ✅ Tính metrics thật sự bằng networkx
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


df_songs, G, partition = generate_spotify_data()
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

    # Biểu đồ ảnh hưởng theo thể loại
    @app.callback(
        Output('genre-influence-chart', 'figure'),
        Input('main-tabs', 'value')
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
        Input('main-tabs', 'value')
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
            height=350,
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
    @app.callback(
        Output('network-song', 'figure'),
        [Input('main-tabs', 'value'),
        Input('network-song', 'clickData')]
    )
    def draw_song_network(tab_value, clickData):
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
        # Sinh màu tự động nếu genre nhiều hơn 10
        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Dark24
        genre_color_map = {g: color_palette[i % len(color_palette)] for i, g in enumerate(genre_list)}

        # Top 5 influential nodes
        top5 = set(df_songs.nlargest(5, 'Lambda')['track_name'])

        # Xác định node được chọn
        selected_node = None
        if clickData and 'points' in clickData and len(clickData['points']) > 0:
            point = clickData['points'][0]
            node_label = point.get('hovertext') or point.get('text')
            if node_label:
                selected_node = node_label.split('<br>')[0].replace('<b>', '').replace('</b>', '')

        edge_traces = []
        for edge in H.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            if edge[2].get('connection') == 'artist':
                color = '#1DB954'
                width = 2.5
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
            node_color.append(genre_color_map.get(genre, '#CCCCCC'))
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
        annotations = []
        x_range = y_range = None

        if selected_node and selected_node in H.nodes:
            x, y = pos[selected_node]
            # Hiệu ứng động khoanh vùng
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
            # Zoom vào node (giá trị 0.15 là mức zoom, có thể điều chỉnh)
            x_range = [x - 0.15, x + 0.15]
            y_range = [y - 0.15, y + 0.15]
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

            fig = go.Figure(
                data=edge_traces + ([highlight_trace] if highlight_trace else []) + [node_trace] + legend_traces,
                layout=go.Layout(
                    title=dict(
                        text='<b>SPOTIFY MUSIC NETWORK</b> (Select node to see details)',
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
                        x=1.05,     # dịch sát mép phải hơn
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
        num_nodes = H.number_of_nodes()
        num_edges = H.number_of_edges()
        total_streams = int(df_songs[df_songs['track_name'].isin(H.nodes)]['popularity'].sum())

        network_stats = html.Div([
            html.P(f"Số lượng node: {num_nodes}", style={'color': '#1DB954', 'display': 'inline-block', 'marginRight': '20px'}),
            html.P(f"Số lượng edge: {num_edges}", style={'color': '#1DB954', 'display': 'inline-block', 'marginRight': '20px'}),
            html.P(f"Tổng số stream: {total_streams:,}", style={'color': '#1DB954', 'display': 'inline-block'})
        ], style={'textAlign': 'center', 'marginBottom': '10px', 'fontWeight': 'bold', 'fontSize': '16px'})

        # Thông tin node được chọn (hiện ảnh nếu có)
        node_info = ""
        if selected_node and selected_node in H.nodes:
            node_data = H.nodes[selected_node]
            eigen = df_songs[df_songs['track_name'] == selected_node]['Lambda'].values
            eigen = eigen[0] if len(eigen) > 0 else 0.1
            popularity = df_songs[df_songs['track_name'] == selected_node]['popularity'].values
            popularity = popularity[0] if len(popularity) > 0 else 0
            artist = node_data.get('artist', '')
            genre = node_data.get('genre', '')
            node_info = html.Div([
                html.H5(f"Bài hát: {selected_node}", style={'color': '#FFD700'}),
                html.P(f"Nghệ sĩ: {artist}"),
                html.P(f"Thể loại: {genre}"),
                html.P(f"Độ ảnh hưởng: {eigen:.2f}"),
                html.P(f"Độ phổ biến: {popularity}")
            ], style={'textAlign': 'center'})

        return fig

    @app.callback(
        Output('popularity-spread-correlation', 'figure'),
        [Input('main-tabs', 'value'),
        Input('genre-filter', 'value')]
    )
    def update_correlation_chart(tab, selected_genres):
        if tab != 'overview':
            return go.Figure()

        # Bắt đầu với toàn bộ dữ liệu gốc
        filtered_df = df_songs.copy()

        # Lọc theo genre nếu có chọn
        if selected_genres and 'All' not in selected_genres:
            filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]

        # Kiểm tra dữ liệu đủ tính toán không
        if filtered_df.empty or 'popularity' not in filtered_df.columns or 'Lambda' not in filtered_df.columns:
            return go.Figure()
        # Loại bỏ các dòng có NaN hoặc không hợp lệ
        filtered_df = filtered_df.dropna(subset=['popularity', 'Lambda'])

        # Nếu dữ liệu không đủ → trả về biểu đồ trống
        if len(filtered_df) < 2 or filtered_df['popularity'].nunique() == 1:
            return go.Figure()
        # Tính z-score
        filtered_df['popularity_zscore'] = (filtered_df['popularity'] - filtered_df['popularity'].mean()) / filtered_df['popularity'].std()
        filtered_df['size'] = np.where(filtered_df['popularity_zscore'] > 1, 12, 6)

        x = filtered_df['popularity']
        y = filtered_df['Lambda']
        slope, intercept = np.polyfit(x, y, 1)
        r_value = np.corrcoef(x, y)[0, 1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=filtered_df['size'],
                color=x,
                colorscale='Viridis',
                showscale=True,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey'),
                colorbar=dict(title='Popularity')
            ),
            text=filtered_df.apply(lambda row: f"<b>{row['track_name']}</b><br>Artist: {row['artist_names']}<br>Genre: {row['genre']}", axis=1),
            hoverinfo='text',
            name='Tracks'
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=slope * x + intercept,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Linear Fit (r={r_value:.2f})'
        ))

        fig.add_shape(type="rect", x0=0, x1=50, y0=0, y1=0.5,
                    fillcolor="rgba(65,105,225,0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=50, x1=100, y0=0.5, y1=1,
                    fillcolor="rgba(0,128,0,0.1)", line=dict(width=0))

        fig.update_layout(
            title=dict(
                text='<b>POPULARITY AND SPREADABILITY</b>',
                y=0.97,  # đẩy lên gần sát đỉnh
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(family="Montserrat", size=18, color="#1DB954")
            ),
            xaxis_title='Popularity (0-100)',
            yaxis_title='Spreadability (Lambda Centrality)',
            plot_bgcolor='rgba(240,240,240,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Montserrat", size=12, color="#333"),
            hovermode='closest',
            annotations=[
                dict(x=0.01, y=0.99, xref='paper', yref='paper',
                    text=f"Total Tracks: {len(filtered_df)}<br>Correlation Coefficient: {r_value:.2f}",
                    showarrow=False, bgcolor='white', align='left'),
                dict(x=50, y=0.05, text="Popularity Threshold",
                    showarrow=True, arrowhead=1, ax=0, ay=-40),
                dict(x=90, y=0.5, text="High Influence Zone",
                    showarrow=True, arrowhead=1, ax=0, ay=40)
            ],
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.3,
                    buttons=[
                        dict(label="All Data", method="update", args=[{"visible": [True, True, True, True]}]),
                        dict(label="Trend Only", method="update", args=[{"visible": [False, True, False, False]}]),
                        dict(label="Zones Only", method="update", args=[{"visible": [False, False, True, True]}])
                    ],
                     font=dict(
                            color="#1DB954",              # màu chữ
                            size=12
                        ),
                )
            ]
        )

        return fig
    # Bài hát trong cộng đồng và hiển thị hình ảnh
    @app.callback(
        Output('top-artist-chart', 'figure'),
        Input('main-tabs', 'value')
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
    
    #Audio Features Overview 
    def create_audio_features_radar(community_data=None, community_ids=[0], 
                                compare_mode=False, custom_colors=None, 
                                title=None, show_stats=False, dark_mode=False):
        try:
            # Kiểm tra và khởi tạo dữ liệu
            if community_data is None:
                df_songs, G, partition = generate_spotify_data()
                community_data = df_songs.copy()
            elif not isinstance(community_data, pd.DataFrame):
                community_data = pd.DataFrame(community_data)
            
            # Validate input data
            if community_data.empty:
                print("Cảnh báo: Dữ liệu đầu vào trống")
                return go.Figure()
                
            if 'genre' not in community_data.columns:
                print("Lỗi: Dữ liệu không có cột 'genre'")
                return go.Figure()
                
            # Thiết lập màu sắc theo theme
            if dark_mode:
                bg_color = '#121212'
                text_color = '#FFFFFF'
                grid_color = 'rgba(255, 255, 255, 0.1)'
                paper_color = '#121212'
            else:
                bg_color = '#FFFFFF'
                text_color = '#2E2E2E'
                grid_color = 'rgba(0, 0, 0, 0.1)'
                paper_color = '#FFFFFF'
        
            features = ['valence 😊', 'energy ⚡', 'danceability 💃', 
                    'acousticness 🎸', 'instrumentalness 🎻', 'liveness 🎭']
            
            # Giá trị mặc định
            default_vals = {
                'valence': 0.4,
                'energy': 0.8,
                'danceability': 0.11,
                'acousticness': 0.7,
                'instrumentalness': 0.2,
                'liveness': 0.5
            }
            
            # Màu sắc hiện đại (Spotify + gradient)
            if custom_colors is None:
                if compare_mode:
                    colors = [
                        '#1DB954',  # Spotify green
                        '#FF9F1C',  # Vivid orange
                        '#FF6B6B',  # Light red
                        '#4ECDC4',  # Tiffany blue
                    ]
                else:
                    colors = ['#1DB954']  # Spotify green
            else:
                colors = custom_colors
                
            # Tạo figure với template hiện đại
            fig = go.Figure()
            
            # Tính toán và thêm dữ liệu
            for i, comm_id in enumerate(community_ids):
                comm_data = community_data[community_data['genre'] == comm_id]
                
                if comm_data.empty:
                    print(f"Cảnh báo: Không tìm thấy dữ liệu cho community {comm_id}")
                    continue
                    
                # Lấy feature names không có emoji để tính toán
                clean_features = [f.split(' ')[0] for f in features]
                avg_values = [
                    comm_data[f].mean() if f in comm_data.columns else default_vals[f]
                    for f in clean_features
                ]
                
                # Tính độ lệch chuẩn nếu cần
                if show_stats:
                    std_values = [
                        comm_data[f].std() if f in comm_data.columns else 0
                        for f in clean_features
                    ]
                
                # Tạo tên hiển thị với số lượng bài hát
                name = f'Genre {comm_id}'
                if show_stats:
                    name += f' (n={len(comm_data):,}'
                
                # Thêm trace chính với hiệu ứng mượt mà
                fig.add_trace(go.Scatterpolar(
                    r=avg_values,
                    theta=features,
                    fill='toself',
                    name=name,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2.5,
                        shape='spline',
                        smoothing=1.3
                    ),
                    fillcolor=f'rgba{hex_to_rgb(colors[i % len(colors)], 0.15)}',
                    hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=colors[i % len(colors)],
                        font_size=12,
                        font_color='white'
                    )
                ))
                
                # Thêm dải độ lệch với hiệu ứng trong suốt
                if show_stats and not compare_mode:
                    fig.add_trace(go.Scatterpolar(
                        r=[max(0, avg-std) for avg, std in zip(avg_values, std_values)],
                        theta=features,
                        fill=None,
                        line=dict(color=colors[i % len(colors)], width=0.1),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=[min(1, avg+std) for avg, std in zip(avg_values, std_values)],
                        theta=features,
                        fill='tonext',
                        line=dict(color=colors[i % len(colors)], width=0.1),
                        fillcolor=f'rgba{hex_to_rgb(colors[i % len(colors)], 0.05)}',
                        showlegend=False,
                        hoverinfo='none'
                    ))
            
            # Thiết lập layout hiện đại
            title_text = title if title else (
                'Audio Features Comparison' if compare_mode else 
                f'Audio Features - Genre {community_ids[0]}'
            )
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor=grid_color,
                        linecolor=grid_color,
                        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        tickfont=dict(color=text_color, size=10),
                        tickformat='.1f',
                        angle=90
                    ),
                    angularaxis=dict(
                        linecolor=grid_color,
                        gridcolor=grid_color,
                        rotation=90,
                        hoverformat='.2f',
                        tickfont=dict(color=text_color, size=11)
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.15,
                    xanchor="center",
                    x=0.5,
                    font=dict(color=text_color, size=12),
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=text_color,
                title={
                    'text': title_text,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, color=text_color, family='Montserrat')
                },
                margin=dict(l=50, r=50, t=100, b=50),
                hoverlabel=dict(
                    bgcolor=text_color,
                    font_size=12,
                    font_color=paper_color,
                    bordercolor=text_color
                ),
                autosize=True,
                height=350
            )
            
            # Thêm annotation nếu có
            if not compare_mode and show_stats:
                fig.add_annotation(
                    text=f"Based on {len(comm_data):,} tracks",
                    xref="paper", yref="paper",
                    x=0.5, y=1.1,
                    showarrow=False,
                    font=dict(size=10, color=text_color))
                
            return fig
            
        except Exception as e:
            print(f"Lỗi khi tạo biểu đồ radar: {str(e)}")
            return go.Figure()

    # Hàm hỗ trợ chuyển hex sang rgba
    def hex_to_rgb(hex_color, alpha=1):
        """Chuyển mã màu hex sang tuple rgba"""
        hex_color = hex_color.lstrip('#')
        hlen = len(hex_color)
        rgb = tuple(int(hex_color[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
        return rgb + (alpha,)

    def create_feature_comparison_bar(community_data, top_n=3, corner_radius=10):
        top_genres = community_data['genre'].value_counts().nlargest(top_n).index.tolist()
        features = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness']

        # 🎨 Gradient màu theo feature
        gradient_colors = {
            'valence': '#A7FFEB',
            'energy': '#1ED760',
            'danceability': '#B2F2C9',
            'acousticness': '#C8F560',
            'instrumentalness': '#1DB954',
            'liveness': '#F8E0E4'
        }

        data = []
        for i, genre in enumerate(top_genres):
            genre_data = community_data[community_data['genre'] == genre]
            means = [genre_data[f].mean() if f in genre_data.columns else 0 for f in features]

            data.append(go.Bar(
                name=genre,
                x=features,
                y=means,
                text=[f"{v:.2f}" for v in means],
                width=0.6,
                textposition='outside',
                marker=dict(
                    color=[gradient_colors[f] for f in features],  # ✅ mỗi cột một màu
                    line=dict(width=0),
                    cornerradius=corner_radius  # (Plotly chưa hỗ trợ trực tiếp, đây là để minh họa)
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Average: %{y:.2f}<br>" +
                    "Genre: %{fullData.name}<extra></extra>"
                ),
                opacity=0.92
            ))

            # 🔁 Trend line cho từng genre
        data.append(go.Scatter(
            name=f"{genre} (trend)",
            x=features,
            y=means,
            mode='lines+markers',
            line=dict(color='rgba(30, 215, 96, 1.0)', dash='dot', width=2.5),  # 🔆 màu sáng hơn, đậm hơn
            marker=dict(symbol='circle', size=7, color='rgba(30, 215, 96, 0.9)', line=dict(width=1, color='white')),
            hoverinfo='skip',
            showlegend=False
        ))

        # 📊 Tạo biểu đồ
        fig = go.Figure(data=data)

        fig.update_layout(
            autosize=True,
            barmode='group',
            bargap=0.15,
            title=dict(
                text='<b>COMPARE MUSIC CHARACTERISTICS BY GENRE</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#191414')
            ),
            xaxis=dict(
                title='Music Features',
                tickangle=-30,
                gridcolor='rgba(29,185,84,0.08)',
                automargin=True 
            ),
            yaxis=dict(
                title='Average Value',
                range=[0, 70],  # các feature đều ở [0, 1]
                gridcolor='rgba(29,185,84,0.08)',
                automargin=True 
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family="Montserrat",
                color='#191414'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                borderwidth=1
            ),
            hoverlabel=dict(
                bgcolor='black',
                font_size=14,
                font_family="Montserrat"
            ),
            height=350,
            margin=dict(l=50, r=20, t=90, b=60),
            uniformtext=dict(minsize=10, mode='show')
        )

        return fig

    
    # Song sankey community
    def create_sankey_community(df_songs, viral_track, top_n_similar=3, similarity_threshold=0.8, dark_mode=False):
        # Validate input
        if viral_track not in df_songs['track_name'].values:
            raise ValueError(f"Viral track '{viral_track}' not found in dataset")
        
        try:
            viral_row = df_songs[df_songs['track_name'] == viral_track].iloc[0]
        except IndexError:
            raise ValueError(f"Could not retrieve data for viral track '{viral_track}'")
        
        viral_artist = viral_row['artist_names']
        viral_genre = viral_row.get('genre', 'Unknown Genre')
        
        # Thiết lập màu sắc theo theme
        if dark_mode:
            bg_color = '#121212'
            text_color = '#FFFFFF'
            grid_color = 'rgba(255, 255, 255, 0.1)'
        else:
            bg_color = '#FFFFFF'
            text_color = '#2E2E2E'
            grid_color = 'rgba(0, 0, 0, 0.1)'
        
        # Giới hạn số lượng playlist hiển thị (tối đa 5)
        viral_playlists = []
        if 'playlist_name' in viral_row and pd.notna(viral_row['playlist_name']):
            all_playlists = [p.strip() for p in viral_row['playlist_name'].split(',') if p.strip()]
            viral_playlists = all_playlists[:5]
        
        # Tìm bài hát tương tự
        features = ['valence', 'energy', 'danceability', 'acousticness', 'instrumentalness', 'liveness']
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(df_songs[features].values.astype(float))
        viral_vec = feature_matrix[df_songs['track_name'] == viral_track][0]
        
        # Tính độ tương đồng cosine
        similarities = cosine_similarity([viral_vec], feature_matrix)[0]
        df_songs['sim_score'] = similarities
        
        # Lọc bài hát tương tự
        similar_mask = (df_songs['track_name'] != viral_track) & (df_songs['sim_score'] >= similarity_threshold)
        similar_tracks = (df_songs[similar_mask]
                        .sort_values('sim_score', ascending=False)
                        .head(top_n_similar)['track_name'].tolist())
        
        # Rút gọn tên nếu quá dài
        def shorten_name(name, max_length=20):
            return (name[:max_length] + '...') if len(name) > max_length else name
        
        # Chuẩn bị các node
        labels = [f"🎵 {shorten_name(viral_track)}"]  # Bài hát viral (0)
        labels += [f"🎤 {shorten_name(viral_artist)}"]  # Nghệ sĩ (1)
        labels += [f"📻 {shorten_name(p)}" for p in viral_playlists]  # Playlists (2...)
        labels += [f"🎶 {shorten_name(t)}" for t in similar_tracks]  # Bài hát tương tự (...)
        
        # Tạo mapping vị trí node
        node_positions = {
            "viral": 0,
            "artist": 1,
            "playlists": {p: i+2 for i, p in enumerate(viral_playlists)},
            "similar": {t: i+2+len(viral_playlists) for i, t in enumerate(similar_tracks)}
        }
        
        # Tạo các liên kết
        sources, targets, values, link_labels = [], [], [], []
        
        # Viral track -> Artist
        sources.append(node_positions["viral"])
        targets.append(node_positions["artist"])
        values.append(1.0)
        link_labels.append("Created by")
        
        # Viral track -> Playlists (giới hạn số lượng)
        for p, idx in node_positions["playlists"].items():
            sources.append(node_positions["viral"])
            targets.append(idx)
            values.append(1.0)
            link_labels.append("Featured in")
        
        # Viral track -> Similar tracks
        for t, idx in node_positions["similar"].items():
            sim_score = df_songs[df_songs['track_name'] == t]['sim_score'].values[0]
            sources.append(node_positions["viral"])
            targets.append(idx)
            values.append(sim_score)
            link_labels.append(f"Similar: {sim_score:.2f}")
        
        # Artist -> Similar tracks (chỉ hiển thị nếu cùng nghệ sĩ)
        for t, idx in node_positions["similar"].items():
            artist_of_t = df_songs[df_songs['track_name'] == t]['artist_names'].values[0]
            if artist_of_t == viral_artist:
                sources.append(node_positions["artist"])
                targets.append(idx)
                values.append(1.0)
                link_labels.append("Same artist")
        
        # Playlist -> Similar tracks (chỉ hiển thị nếu có playlist chung)
        for t, idx in node_positions["similar"].items():
            if 'playlist_names' in df_songs.columns:
                playlist_data = df_songs[df_songs['track_name'] == t]['playlist_names'].values
                if len(playlist_data) > 0 and pd.notna(playlist_data[0]):
                    common_playlists = set(p.strip() for p in playlist_data[0].split(',')) & set(node_positions["playlists"].keys())
                    if common_playlists:
                        sim_score = df_songs[df_songs['track_name'] == t]['sim_score'].values[0]
                        sources.append(node_positions["playlists"][next(iter(common_playlists))])
                        targets.append(idx)
                        values.append(sim_score * 0.7)
                        link_labels.append("Shared playlist")
        
        # Màu sắc
        viral_color = "#FF69B4"      # Pink
        artist_color = "#FFB347"    # Orange
        playlist_color = "#7EC8E3"  # Blue
        similar_color = "#B39DDB"   # Purple
        
        node_colors = [viral_color, artist_color] + [playlist_color]*len(viral_playlists) + [similar_color]*len(similar_tracks)
        
        # Màu liên kết với độ trong suốt
        link_colors = []
        for label in link_labels:
            if "Created by" in label:
                link_colors.append(f"rgba{hex_to_rgb(artist_color, 0.7)}")
            elif "Featured in" in label:
                link_colors.append(f"rgba{hex_to_rgb(playlist_color, 0.7)}")
            elif "Same artist" in label:
                link_colors.append(f"rgba{hex_to_rgb(artist_color, 0.5)}")
            elif "Shared playlist" in label:
                link_colors.append(f"rgba{hex_to_rgb(playlist_color, 0.5)}")
            else:
                link_colors.append(f"rgba{hex_to_rgb(similar_color, 0.6)}")
        
        # Scale giá trị liên kết
        scaling_factor = 0.3
        scaled_values = [v * scaling_factor for v in values]
        
        # Tính toán vị trí node để phân bố đều
        def calculate_node_positions():
            # Viral track ở giữa bên trái
            x_pos = [0.1]  # Viral track
            y_pos = [0.5]
            
            # Artist ở giữa cột thứ 2
            x_pos.append(0.3)
            y_pos.append(0.5)
            
            # Playlists phân bố đều ở cột thứ 3
            playlist_count = len(viral_playlists)
            x_pos.extend([0.5] * playlist_count)
            if playlist_count > 1:
                y_pos.extend(np.linspace(0.1, 0.9, playlist_count))
            else:
                y_pos.append(0.5)
            
            # Similar tracks phân bố đều ở cột thứ 4
            similar_count = len(similar_tracks)
            x_pos.extend([0.8] * similar_count)
            if similar_count > 1:
                y_pos.extend(np.linspace(0.1, 0.9, similar_count))
            else:
                y_pos.append(0.5)
            
            return x_pos, y_pos
        
        x_pos, y_pos = calculate_node_positions()
        
        # Tạo figure
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                x=x_pos,
                y=y_pos,
                hovertemplate="<b>%{label}</b><extra></extra>",
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    font_size=10,
                    font_family="Montserrat"
                )
            ),
            link=dict(
                source=sources,
                target=targets,
                value=scaled_values,
                color=link_colors,
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=link_labels,
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    font_size=10,
                    font_family="Montserrat"
                )
            )
        ))
        
        # Layout hiện đại
        title = f"<b>NETWORK OF INFLUENCE:</b> {shorten_name(viral_track, 25)}"
        subtitle = f"<span style='color:#666'>{shorten_name(viral_artist, 30)} • {viral_genre}</span>"
        
        fig.update_layout(
            title_text=f"{title}<br>{subtitle}",
            font=dict(family="Montserrat", size=10, color=text_color),
            height=400,
            width=550,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=80, b=100, t=100),
            hovermode="x unified",
            annotations=[
                dict(
                    x=0.5, y=-0.15, xref="paper", yref="paper",
                    text=(
                        "<b>Chú thích:</b> "
                        "<span style='color:#FF69B4'>■</span> Viral • "
                        "<span style='color:#FFB347'>■</span> Nghệ sĩ • "
                        "<span style='color:#7EC8E3'>■</span> Playlist • "
                        "<span style='color:#B39DDB'>■</span> Bài tương tự"
                    ),
                    showarrow=False,
                    font=dict(size=10, color=text_color),
                    align="center"
                )
            ]
        )
        
        # Thêm nút reset view
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "buttons": [{
                    "args": [{"node.x": x_pos, "node.y": y_pos}],
                    "label": "⟲ Reset View",
                    "method": "restyle"
                }],
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 1.1,
                "yanchor": "top"
            }]
        )
        
        return fig
    #Top Artists (Horizontal Bar Chart)
    def create_top_artists_bar(community_data):
        top_artists = community_data['artist_names'].value_counts().nlargest(5)
        
        # Create figure with horizontal bar style
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_artists.index,
            x=top_artists.values,
            orientation='h',
            text=top_artists.values,
            textposition='outside',
            textangle=0,  # ✅ Đảm bảo chữ nằm ngang
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
        'height': '150px', 
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
            Output('community-audio-features', 'figure'),
            Output('community-sankey-distribution', 'figure'),
            Output('community-feature-comparison', 'figure'),
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
            radar_fig = create_audio_features_radar(community_data, [selected_community])
            sankey_fig = create_sankey_community(community_data, viral_track)
            bar_fig = create_feature_comparison_bar(community_data)
            artists_fig = create_top_artists_bar(community_data)
            recommendations = create_song_recommendations(community_data, current_index)

            # Tính toán Lambda trung bình và số lượng bài hát
            valid_lambda = community_data['Lambda'].dropna()
            lambda_mean = f"{valid_lambda.mean():.3f}" if not valid_lambda.empty else "0.000"
            total_songs = str(len(valid_lambda))

            return (
                radar_fig,
                sankey_fig,
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
        Output('center-artists', 'options'),
        Output('genre-risk-dropdown', 'options')],
        [Input('song-risk-dropdown', 'value'),
        Input('center-artists', 'value'),
        Input('genre-risk-dropdown', 'value')]
    )
    def update_all_dropdowns(song_value, artist_value, genre_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        df_filtered = df_songs.copy()

        if triggered_id == 'song-risk-dropdown' and song_value:
            df_filtered = df_filtered[df_filtered['track_name'].isin(song_value if isinstance(song_value, list) else [song_value])]
        elif triggered_id == 'center-artists' and artist_value:
            df_filtered = df_filtered[df_filtered['artist_names'].isin(artist_value if isinstance(artist_value, list) else [artist_value])]
        elif triggered_id == 'genre-risk-dropdown' and genre_value:
            df_filtered = df_filtered[df_filtered['genre'].isin(genre_value if isinstance(genre_value, list) else [genre_value])]

        # Lấy danh sách lọc
        song_options = [{'label': s, 'value': s} for s in sorted(df_filtered['track_name'].unique())]
        artist_options = [{'label': a, 'value': a} for a in sorted(df_filtered['artist_names'].unique())]
        genre_options = [{'label': g, 'value': g} for g in sorted(df_filtered['genre'].unique())]

        return song_options, artist_options, genre_options


    @app.callback(
        Output('sensitivity-network', 'figure'),
        [Input('main-tabs', 'value'),
        Input('center-artists', 'value'), 
        Input('genre-risk-dropdown', 'value'),
        Input('song-risk-dropdown', 'value')]
    )
    def draw_crisis_network(tab, selected_artists, selected_genre, selected_songs):
        if tab != 'risk' or not (selected_artists or selected_songs):
            return go.Figure()

        # Process selections
        if isinstance(selected_artists, str):
            selected_artists = [selected_artists]
        if isinstance(selected_songs, str):
            selected_songs = [selected_songs]
        
        # Get artists from selected songs if provided
        song_artists = []
        if selected_songs:
            song_artists = df_songs[df_songs['track_name'].isin(selected_songs)]['artist_names'].unique().tolist()
        
        # Combine all selected artists
        all_selected_artists = list(set((selected_artists or []) + song_artists))
        
        # Filter by genre if specified
        if selected_genre:
            all_selected_artists = df_songs[
                (df_songs['artist_names'].isin(all_selected_artists)) &
                (df_songs['genre'] == selected_genre)
            ]['artist_names'].unique().tolist()

        # CREATE NETWORK
        G = nx.Graph()
        edge_weights = []

        # Add selected songs as special nodes
        song_nodes = {}
        if selected_songs:
            for song in selected_songs:
                song_data = df_songs[df_songs['track_name'] == song].iloc[0]
                artists = [a.strip() for a in song_data['artist_names'].split(',')]
                G.add_node(f"SONG:{song}", size=25, color='#FF5722', group=2, 
                        influence=1.0, type='song', 
                        customdata=f"{song} - {song_data['artist_names']}")
                
                # Connect song to its artists
                for artist in artists:
                    if artist in all_selected_artists:
                        G.add_edge(f"SONG:{song}", artist, weight=2, strength=0.9)
                        edge_weights.append(2)

        # Add artist nodes and connections
        for artist in all_selected_artists:
            G.add_node(artist, size=30, color='#E53935', group=0, influence=1.0, type='artist')
            collabs = df_songs[df_songs['artist_names'].str.contains(artist, na=False, regex=False)]
            
            # Find related artists through collaborations
            related = {}
            for _, row in collabs.iterrows():
                others = [o.strip() for o in row['artist_names'].split(',') if o.strip() != artist]
                for other in others:
                    related[other] = related.get(other, 0) + 1

            max_collabs = max(related.values()) if related else 1
            for other, count in related.items():
                influence_score = min(1.0, count / max_collabs * 0.8)
                if other not in G.nodes():
                    G.add_node(other, size=15 + (count / max_collabs) * 25, 
                            color='#4ECDC4', group=1, influence=influence_score, type='related')
                G.add_edge(artist, other, weight=count, strength=influence_score)
                edge_weights.append(count)

        affected = set(all_selected_artists)
        frontier = set(all_selected_artists)
        spread_prob = 0.5
        steps = 3
        spread_history = []

        for step in range(steps):
            new_frontier = set()
            for node in frontier:
                # Skip song nodes (they don't propagate risk)
                if isinstance(node, str) and node.startswith('SONG:'):
                    continue
                    
                for neighbor in G.neighbors(node):
                    if neighbor not in affected:
                        edge_strength = G[node][neighbor].get('strength', 0.2)
                        prob = min(1.0, spread_prob * edge_strength)
                        if np.random.rand() < prob:
                            new_frontier.add(neighbor)
                            G.nodes[neighbor]['influence'] = max(
                                G.nodes[neighbor]['influence'],
                                G.nodes[node]['influence'] * 0.7
                            )
            affected.update(new_frontier)
            frontier = new_frontier
            spread_history.append((step, set(affected)))

        # PREPARE VISUALIZATION
        pos = nx.spring_layout(G, seed=42, k=0.6, iterations=100)
        
        # Edge traces
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=0.5 + 2 * G[u][v].get('strength', 0.5),
                        color=f'rgba(100,100,100,{0.3 + 0.5 * G[u][v].get("strength", 0.5)})'),
                hoverinfo='none',
                mode='lines',
                opacity=0.8
            ))

        # Node traces
        node_x, node_y, node_size, node_color, node_text, node_hover = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            inf = G.nodes[node].get('influence', 0)
            node_x.append(x)
            node_y.append(y)
            node_size.append(G.nodes[node]['size'])
            
            # Determine node color and hover text
            if node.startswith('SONG:'):
                node_color.append('#FF5722')  # Orange for songs
                node_text.append(node.split(':')[1][:15] + '...')
                node_hover.append(G.nodes[node].get('customdata', node))
            elif node in all_selected_artists:
                node_color.append('#E53935')  # Red for selected artists
                node_text.append(node[:15] + ('...' if len(node) > 15 else ''))
                node_hover.append(f"Artist: {node}<br>Influence: {inf:.2f}")
            elif node in affected:
                node_color.append(f'rgba(255, 204, 0, {inf})')  # Yellow for affected
                node_text.append(node[:15] + ('...' if len(node) > 15 else ''))
                node_hover.append(f"Artist: {node}<br>Influence: {inf:.2f}")
            else:
                node_color.append("#000000")  # Black for others
                node_text.append('')
                node_hover.append(f"Related Artist: {node}<br>Influence: {inf:.2f}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hovertext=node_hover,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.9
            )
        )

        # ANIMATION FRAMES
        frames = []
        for step, affected_nodes in spread_history:
            frame_size = []
            frame_color = []
            frame_border = []

            for node in G.nodes():
                inf = G.nodes[node].get('influence', 0)
                base_size = G.nodes[node]['size']
                
                if node.startswith('SONG:'):
                    frame_color.append('#FF5722')
                    frame_size.append(base_size)
                    frame_border.append('black')
                elif node in all_selected_artists:
                    frame_color.append('#E53935')
                    frame_size.append(base_size)
                    frame_border.append('black')
                elif node in affected_nodes:
                    frame_color.append(f'rgba(255, 204, 0, {inf})')
                    frame_size.append(base_size + 10 * inf)
                    frame_border.append('rgba(255,255,0,0.8)')
                else:
                    frame_color.append("#000000")
                    frame_size.append(base_size)
                    frame_border.append('rgba(200,200,200,0.4)')

            frames.append(go.Frame(
                data=[
                    *edge_traces,
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition='top center',
                        hovertext=node_hover,
                        hoverinfo='text',
                        marker=dict(
                            size=frame_size,
                            color=frame_color,
                            line=dict(width=3, color=frame_border),
                            opacity=0.9
                        )
                    )
                ],
                name=f"Step {step+1}"
            ))

        # CREATE FIGURE
        fig = go.Figure(
            data=edge_traces + [node_trace],
            frames=frames
        )

        # LAYOUT & ANIMATION
        title_text = '<b>RISK PROPAGATION NETWORK</b>'
        if selected_songs:
            title_text += f'<br><span style="font-size:14px">Selected Songs: {", ".join([s[:15] + ("..." if len(s)>15 else "") for s in selected_songs])}</span>'
        if all_selected_artists:
            title_text += f'<br><span style="font-size:14px">Artists: {", ".join([a[:15] + ("..." if len(a)>15 else "") for a in all_selected_artists])}</span>'
        if selected_genre:
            title_text += f'<br><span style="font-size:14px">Genre: {selected_genre}</span>'

        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=20, color='#191414', family='Montserrat'),
                x=0.5, xanchor='center', y=0.95, yanchor='top'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=100),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode='pan',
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(label="▶️ Play", method="animate",
                        args=[None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                steps=[dict(
                    method="animate",
                    args=[[f"Step {i+1}"], {"mode": "immediate", "frame": {"duration": 800}, "transition": {"duration": 300}}],
                    label=f"Step {i+1}"
                ) for i in range(len(frames))],
                transition={"duration": 300},
                x=0.1,
                y=-0.1,
                currentvalue={"prefix": "Current: ", "font": {"size": 14}},
                len=0.8
            )],
            transition=dict(duration=500)
        )

        return fig

    @app.callback(
    Output('community-impact-bar', 'figure'),
    Input('main-tabs', 'value')
    )
    def update_risk_estimate(tab):
        if tab != 'risk':
            return go.Figure()

        genres = df_songs['genre'].dropna().unique()
        results = []

        for genre in genres:
            genre_songs = df_songs[df_songs['genre'] == genre]['track_name'].tolist()
            if not genre_songs:
                results.append(0)
                continue

            affected_ratios = []
            for _ in range(300):  # số lần mô phỏng
                affected = set(genre_songs)
                frontier = set(genre_songs)
                while frontier:
                    new_frontier = set()
                    for node in frontier:
                        for nb in G.neighbors(node):
                            if nb not in affected and np.random.rand() < 0.25:
                                new_frontier.add(nb)
                    affected.update(new_frontier)
                    frontier = new_frontier
                affected_ratios.append(
                    (len(affected) - len(genre_songs)) / max(1, G.number_of_nodes() - len(genre_songs))
                )
            results.append(np.mean(affected_ratios))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=genres,
            x=results,
            orientation='h',
            name='Impact',
            marker=dict(
                color=results,
                colorscale=[
                        [0.0, "#178A3E"],   # đậm nhất
                        [0.3, "#1AA34A"],
                        [0.6, "#1DB954"],   # Spotify green
                        [0.85, "#1ED760"],
                        [1.0, "#B2F2C9"]    # nhạt nhất
                    ],
                line=dict(color='rgba(255,255,255,0.5)', width=1),
                cmin=0,
                cmax=max(results)
            ),
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>",
            text=[f"{x:.2f}" for x in results],
            textposition='outside',
            textfont=dict(color='black')
        ))


        fig.update_layout(
            title='<b>Genre Impact Analysis</b>',
            xaxis_title='Propagation Risk Score',
            yaxis_title='Genre',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black'),
            xaxis=dict(
                showgrid=True,
                tick0=0,
                dtick=0.1,
                range=[0, 0.3],
                ticks="outside",
                ticklen=8,
                tickwidth=1,
                tickcolor='rgba(150,150,150,0.6)'
            ),
            bargap=0.25,
            transition={'duration': 500}
        )

        return fig

    @app.callback(
        Output('influence-heatmap', 'figure'),
        [Input('main-tabs', 'value'),
        Input('center-artists', 'value'),
        Input('genre-risk-dropdown', 'value'),
        Input('song-risk-dropdown', 'value')]
    )
    def draw_influence_heatmap(tab, selected_artists, selected_genre, selected_songs):
        if tab != 'risk':
            return go.Figure()

        # Get all affected artists from selections
        affected_artists = set()
        
        if selected_artists:
            if isinstance(selected_artists, str):
                selected_artists = [selected_artists]
            affected_artists.update(selected_artists)
        
        if selected_songs:
            if isinstance(selected_songs, str):
                selected_songs = [selected_songs]
            song_artists = df_songs[df_songs['track_name'].isin(selected_songs)]['artist_names'].unique().tolist()
            affected_artists.update(song_artists)
        
        # Filter by genre if specified
        if selected_genre:
            affected_artists = set(df_songs[
                (df_songs['artist_names'].isin(affected_artists)) &
                (df_songs['genre'] == selected_genre)
            ]['artist_names'].unique().tolist())

        # Calculate influence scores
        if affected_artists:
            # Calculate average eigenvector centrality for genres based on affected artists
            genre_influence = df_songs[df_songs['artist_names'].isin(affected_artists)].groupby('genre')['Lambda'].mean().reset_index()
        else:
            # Fallback to all data if no selections
            genre_influence = df_songs.groupby('genre')['Lambda'].mean().reset_index()
        
        genre_influence['Lambda'] = genre_influence['Lambda'].fillna(0)
        genre_influence = genre_influence.sort_values('Lambda', ascending=False)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[genre_influence['Lambda'].values],
            x=genre_influence['genre'],
            y=['Mức độ ảnh hưởng'],
            colorscale='Tealgrn',
            colorbar=dict(
                title=dict(
                    text="Lambda",
                    side="right",
                    font=dict(size=12, family='Montserrat')
                ),
                tickfont=dict(size=10, family='Montserrat')
            ),
            hovertemplate=(
                "<b>Thể loại</b>: %{x}<br>"
                "<b>Điểm ảnh hưởng</b>: %{z:.3f}<extra></extra>"
            )
        ))

        fig.update_layout(
            title={
                'text': 'ẢNH HƯỞNG TRUNG BÌNH CỦA THỂ LOẠI',
                'font': {
                    'family': 'Montserrat',
                    'size': 18,
                    'color': '#191414'
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat'),
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                title=dict(
                    text='Thể loại âm nhạc',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12),
                tickangle=-45
            ),
            yaxis=dict(
                showticklabels=True,
                title=None,
                tickfont=dict(size=14)
            ),
            height=400,
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family='Montserrat'
            )
        )
        
        max_influence = genre_influence['Lambda'].max()
        if max_influence > 0:
            fig.add_annotation(
                text=f"Thể loại ảnh hưởng nhất: {genre_influence.iloc[0]['genre']}",
                xref="paper", yref="paper",
                x=0.95, y=1.1,
                showarrow=False,
                font=dict(size=12, color='#1DB954')
            )
        
        return fig

    @app.callback(
    [Output('risk-prediction', 'figure'),
     Output('high-risk-songs', 'data')],
    [Input('main-tabs', 'value'),
     Input('center-artists', 'value'),
     Input('genre-risk-dropdown', 'value'),
     Input('song-risk-dropdown', 'value')]
    )
  
    def update_risk_forecast(tab, selected_artists, selected_genre, selected_songs, song_reduction_percent=0):
        if tab != 'risk':
            return go.Figure(), []

        if not selected_songs:
            return go.Figure(), []

        df_songs['release_date'] = pd.to_datetime(df_songs['release_date'], format='mixed', errors='coerce')
        selected_release_dates = df_songs[df_songs['track_name'].isin(selected_songs)]['release_date']
        if selected_release_dates.empty:
            return go.Figure(), []

        viral_start_date = pd.to_datetime(selected_release_dates.min())
        df_recent = df_songs[pd.to_datetime(df_songs['release_date']) >= viral_start_date]

        if selected_artists:
            df_recent = df_recent[df_recent['artist_names'].isin(selected_artists)]
        if selected_genre:
            df_recent = df_recent[df_recent['genre'] == selected_genre]

        affected_artists = df_recent['artist_names'].unique().tolist()
        artist_to_index = {a: i for i, a in enumerate(affected_artists)}
        num_artists = len(affected_artists)

        if num_artists == 0:
            return go.Figure(), []

        W = np.zeros((num_artists, num_artists))
        song_counts = {}

        for _, row in df_recent.iterrows():
            artist = row['artist_names']
            idx = artist_to_index[artist]
            song_counts[artist] = song_counts.get(artist, 0) + 1
            same_genre_artists = df_recent[df_recent['genre'] == row['genre']]['artist_names'].unique()
            for other_artist in same_genre_artists:
                if other_artist in artist_to_index and other_artist != artist:
                    jdx = artist_to_index[other_artist]
                    W[idx][jdx] += 1

        if song_reduction_percent > 0:
            factor = 1 - song_reduction_percent / 100
            for artist in song_counts:
                idx = artist_to_index[artist]
                W[idx] *= factor
                W[:, idx] *= factor

        # Thêm tự ảnh hưởng và chuẩn hóa
        W += np.eye(num_artists) * 0.5
        W = W / (W.sum(axis=1, keepdims=True) + 1e-6)

        λ = 1.0
        num_simulations = 500
        num_days = 30
        risk_over_time = np.zeros((num_simulations, num_days))
        infected_counts = np.zeros((num_simulations, num_days))

        for sim in range(num_simulations):
            state = np.zeros(num_artists)
            for song in selected_songs:
                artist = df_songs[df_songs['track_name'] == song]['artist_names'].values
                if len(artist) > 0 and artist[0] in artist_to_index:
                    state[artist_to_index[artist[0]]] = 1

            for t in range(num_days):
                new_state = state.copy()
                for i in range(num_artists):
                    if state[i] == 1:
                        continue
                    influence_sum = np.dot(W[i], state)
                    p = 1 - np.exp(-λ * influence_sum)
                    new_state[i] = np.random.rand() < p
                state = new_state
                risk_over_time[sim, t] = np.mean(state)
                infected_counts[sim, t] = np.sum(state)

        risk_mean = np.mean(risk_over_time, axis=0)
        infected_mean = np.mean(infected_counts, axis=0)
        days = pd.date_range(start=viral_start_date, periods=num_days).strftime('%Y-%m-%d')

        fig = go.Figure()

        # Đường tỷ lệ ảnh hưởng
        fig.add_trace(go.Scatter(
            x=days,
            y=risk_mean,
            name='Tỷ lệ nghệ sĩ bị ảnh hưởng',
            mode='lines+markers',
            line=dict(color='firebrick', width=3),
            marker=dict(size=6),
            hovertemplate='Ngày: %{x}<br>Tỷ lệ: %{y:.2%}<extra></extra>',
        ))

        # Biểu đồ phụ: số nghệ sĩ bị ảnh hưởng
        fig.add_trace(go.Scatter(
            x=days,
            y=infected_mean,
            name='Số nghệ sĩ bị ảnh hưởng',
            mode='lines',
            yaxis='y2',
            line=dict(color='royalblue', width=2, dash='dot'),
            hovertemplate='Ngày: %{x}<br>Số nghệ sĩ: %{y:.0f}<extra></extra>'
        ))

        fig.add_shape(
            type="rect", x0=days[0], x1=days[-1], y0=0.6, y1=1.0,
            fillcolor="rgba(255, 0, 0, 0.08)", layer="below", line_width=0
        )
        fig.add_annotation(
            x=days[np.argmax(risk_mean)],
            y=np.max(risk_mean),
            text=f'⚠ Đỉnh rủi ro: {np.max(risk_mean):.2%}',
            showarrow=True,
            arrowhead=2,
            bgcolor='white',
            font=dict(size=13, color='crimson'),
        )

        fig.update_layout(
            title='📊 Dự báo lan truyền rủi ro sau bài hát viral',
            xaxis=dict(title='Ngày', tickangle=45),
            yaxis=dict(title='Tỷ lệ ảnh hưởng', range=[0, 1], tickformat=".0%"),
            yaxis2=dict(title='Số nghệ sĩ', overlaying='y', side='right', showgrid=False),
            font=dict(family="Arial", size=14),
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,255,255,0.95)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Top 10 rủi ro
        top_risk = df_recent[['track_name', 'artist_names', 'genre', 'Betweenness']].nlargest(10, 'Betweenness')
        top_risk.rename(columns={'Betweenness': 'influence'}, inplace=True)

        return fig, top_risk.to_dict('records')

