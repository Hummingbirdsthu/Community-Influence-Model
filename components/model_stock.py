from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
import requests
import base64

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
    'white': '#FFFFFF',
    'off_white': '#F8F8F8',
    
    # Màu phụ
    'blue': '#2D46B9',
    'purple': '#5038A0'
}
SPOTIFY_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1200px-Spotify_logo_without_text.svg.png"
def image_to_base64(url):
    try:
        response = requests.get(url)
        return "data:image/jpeg;base64," + base64.b64encode(response.content).decode('utf-8')
    except:
        return "data:image/jpeg;base64," + base64.b64encode(requests.get("https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228").content).decode('utf-8')

def generate_stock_data():
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

def dropdown_range(label, from_id_stock, to_id_stock, options):
    return dbc.Col([
        html.Label(label, className="fw-bold mb-1", style={'color': '#222b3a'}),
        dcc.Dropdown(id=from_id_stock, options=[{'label': str(o), 'value': o} for o in options],
                     value=min(options), clearable=False, className="mb-2",
                     style={
                        'backgroundColor': 'white',
                        'color': 'black',
                        'width': '100%',
                        'borderRadius': '10px'
                    }),
        dcc.Dropdown(id=to_id_stock, options=[{'label': str(o), 'value': o} for o in options],
                     value=max(options), clearable=False, className="mb-2",
                     style={
                        'backgroundColor': 'white',
                        'color': 'black',
                        'width': '100%',
                        'borderRadius': '10px'
                    })
    ],
    style={
        'padding': '0.5em',
        'border': '1px solid #888',         # Gray border
        'borderRadius': '8px'          # Optional: rounded corners
    },
    width=10)


model_stock = html.Div(
    style={
        "background": "transparent",
        'borderRadius': '20px',
        'padding': '10px',
        'overflowY': 'scroll',
        'minHeight': 'auto',
        'height': 'calc(94vh - 20px)',
    },
    children=[
        dbc.Container(
            fluid=True,
            style={
                'backgroundColor': 'transparent',
                'minHeight': '100vh',
                'margin': '0',
                'padding': '0',
                'color': 'rgb(0,0,0,1)',
                'fontFamily': 'Montserrat',
                'maxWidth': '1500px',         # Giới hạn chiều rộng tối đa
                'marginLeft': 'auto',
                'overflow': 'hidden'
            },
            children = [
                html.Div(
                    [
                        html.H1("Stock Model Performance", style={'color': '#222b3a', 'fontWeight': 'bold'}),
                        html.P("This section will display Stock model performance metrics and visualizations.", style={'color': '#222b3a'})
                    ],
                    style={
                        'marginBottom': '2em',
                        'marginTop': '1em'
                    }
                ),

                html.Div(
                    [
                        dbc.Row([
                            dbc.Col([
                                html.H3(
                                    "Hyperparameter Tunning",
                                    id='tunning-title_stock'
                                ),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dropdown_range("theta grid", "theta-from_stock", "theta-to_stock", [1.0, 10.0]),
                                        html.Div(style={'height': '0.5em'}),
                                        dropdown_range("gamma grid", "gamma-from_stock", "gamma-to_stock", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]),
                                        dbc.Button("Run Grid Search", id='grid-search-button_stock',
                                            color=["secondary"], n_clicks=0, className="mb-3 d-block", style={'marginTop': '1em', 'color': 'white'}),
                                        html.Div(style={'height': '2em'}),
                                        html.Div(
                                            [
                                                html.Label("Select fixed Theta:", style={'fontWeight': 'bold', 'color': '#222b3a'}),
                                                dcc.Dropdown(
                                                    id='theta-dropdown_stock',
                                                    clearable=False,
                                                    style={
                                                        'backgroundColor': 'white',
                                                        'color': 'black',
                                                        'width': '100%',
                                                        'marginBottom': '1em',
                                                        'borderRadius': '6px',
                                                        'width': '200px'
                                                    }
                                                ),
                                            ],
                                            id='grid-chart-title-container_stock'
                                        )
                                    ], width=3),

                                    dbc.Col([
                                        html.Div([
                                            dcc.Graph(id='silhouette-chart_stock')
                                        ],
                                        id='silhouette-chart-container_stock')
                                    ],width=4)
                                    
                                ], style={'marginBottom': '0.5em'}),
                            ], width=7),

                            dbc.Col([
                                html.H3(id='res-table-name_stock', className="mt-0",
                                        style={
                                            #'padding': '0.2em',
                                            'fontFamily': "Montserrat",
                                            'fontSize': '26px',
                                            'textAlign': 'center'
                                        }),
                                dcc.Loading(id="loading-spinner_stock", type="default", children=[
                                    html.Div(id='combo-display_stock', className="mb-2 fw-semibold", style={'textAlign': 'center'})
                                ]),
                                html.Div(
                                    id='results-table-container_stock',
                                    style={
                                        'width': '96%',      # or '600px', '80%', etc.
                                        'maxHeight': '434px', # for vertical scrolling
                                        'overflowY': 'auto',  # enables scrolling if table overflows
                                        'border': '0px solid #ccc',
                                        'overflowX': 'hidden', # prevent scroll
                                        'padding': '0em'
                                    }
                                ),

                                dcc.Store(id='filtered-grid-results_stock')
                            ], width=5)
                        ], align="start"),

                        dbc.Row([
                            dbc.Col(
                                [
                                    html.Div([dcc.Graph(id='bic-combined-chart_stock')],
                                        id='combine-chart-container_stock'
                                    )
                                ], width=7
                            ),
                            dbc.Col(
                                [
                                    # Plot container placed below the table
                                    html.Div(
                                        id='statistic-table_stock'
                                    )
                                ], width=5
                            )
                        ])
                    ]
                )
            ]
        )
    ]
)