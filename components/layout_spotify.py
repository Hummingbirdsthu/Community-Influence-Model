from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import dash_table
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
    'white': '#FFFFFF',
    'off_white': '#F8F8F8',
    
    # Màu phụ
    'blue': '#2D46B9',
    'purple': '#5038A0'
}
SPOTIFY_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1200px-Spotify_logo_without_text.svg.png"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'])
server = app.server

# Hàm để chuyển đổi hình ảnh URL thành base64
def image_to_base64(url):
    try:
        response = requests.get(url)
        return "data:image/jpeg;base64," + base64.b64encode(response.content).decode('utf-8')
    except:
        return "data:image/jpeg;base64," + base64.b64encode(requests.get("https://i.scdn.co/image/ab67616d00001e02ff9ca10b55ce82ae553c8228").content).decode('utf-8')
def generate_spotify_data():
    df = pd.read_csv("data/merged_with_lambda.csv")
    audio_cols = {
        'valence': lambda n: np.random.uniform(0, 1, n)
    }
    for col, func in audio_cols.items():
        if col not in df.columns:
            df[col] = func(len(df))
    if 'Degree' not in df.columns:
        df['Degree'] = np.random.uniform(0, 1, size=len(df))
    if 'Betweenness' not in df.columns:
        df['Betweenness'] = np.random.uniform(0, 1, size=len(df))
    if 'Eigenvector' not in df.columns:
        eigen_scores = np.random.lognormal(mean=0.5, sigma=1.0, size=len(df))
        df['Eigenvector'] = (eigen_scores - eigen_scores.min()) / (eigen_scores.max() - eigen_scores.min())
    if 'image_base64' not in df.columns and 'image_url' in df.columns:
        df['image_base64'] = df['image_url'].apply(image_to_base64)
    cols_needed = ['valence', 'genre', 'Degree', 'Betweenness', 'Eigenvector']
    df[cols_needed] = df[cols_needed].fillna(0)
    artist_col = 'artist_names'
    genre_col = 'genre' if 'genre' in df.columns else None

    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(row['track_name'],
                   artist=row.get(artist_col, ''),
                   genre=row.get(genre_col, '') if genre_col else '',
                   popularity=row.get('popularity', 0),
                   type='song')
    for artist in df[artist_col].unique():
        tracks = df[df[artist_col] == artist]['track_name'].tolist()
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                G.add_edge(tracks[i], tracks[j], connection='artist')
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G)
    else:
        partition = {}

    return df, G, partition

# Sử dụng:
df_songs, G, partition = generate_spotify_data()

layout_spotify = dbc.Container(
    fluid=True,
    style={
        'backgroundColor': SPOTIFY_COLORS["dark_gray"],
        'minHeight': '100vh',
        'margin': '0',
        'padding': '0',
        'color': SPOTIFY_COLORS['white'],
        'fontFamily': 'Montserrat',
        'maxWidth': '1500px',         # Giới hạn chiều rộng tối đa
        'marginLeft': 'auto',
        'overflow': 'hidden' 
    },
    children=[
        # Nội dung chính
        html.Div(
            style={
                'marginLeft': '0px',
                'padding': '0px',
                'background': 'linear-gradient(135deg, #f8e0e4 0%, #fff 100%)',
                'height': 'calc(100vh - 20px)',
                'overflowY': 'auto' 
            },
            children=[

                # Header
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Img(src=SPOTIFY_LOGO, style={'height': '50px', 'marginRight': '15px'}),
                            html.H1(
                                "SPOTIFY SONG NETWORK DASHBOARD",
                                style={
                                    'color': SPOTIFY_COLORS['white'],
                                    'fontWeight': 900,
                                    'margin': '0',
                                    'textShadow': '2px 2px 4px rgba(0, 0, 0, 0.6)',
                                    'fontFamily': "Montserrat",
                                    'fontSize': '24px'
                                }
                            )
                        ],
                        style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'gap': '15px',
                            'justifyContent': 'center' 
                        }
                    ),
                    width=12
                )
            ),
                
                # Tabs chính
                dcc.Tabs(
                    id='main-tabs',
                    value='overview',
                    children=[
                        # Tab 1: Tổng quan ảnh hưởng
                        dcc.Tab(
                                label='Song Influence Overview',
                                value='overview',
                                style={
                                    'backgroundColor': '#fff',
                                    'color': '#191414',
                                    'border': f'2px solid {SPOTIFY_COLORS["gray"]}',
                                    'padding': '6px 12px',
                                    'fontWeight': 'bold',
                                    'fontFamily': 'Montserrat',
                                    'fontSize': '10px',
                                    'height': '36px',
                                    'lineHeight': '24px',
                                    'boxShadow': 'none',
                                    'marginBottom': '0px'
                                },
                                selected_style={
                                    'backgroundColor': SPOTIFY_COLORS['darker_green'],
                                    'color': SPOTIFY_COLORS['white'],
                                    'border': f'2px solid {SPOTIFY_COLORS["light_green"]}',
                                    'padding': '6px 12px',
                                    'fontWeight': 'bold',
                                    'fontFamily': 'Montserrat',
                                    'fontSize': '10px',
                                    'height': '36px',
                                    'lineHeight': '24px',
                                    'boxShadow': 'none',
                                    'marginBottom': '0px'
                                },
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dcc.Graph(id='genre-influence-chart', style={'height': '300px'},config={'responsive': True}),
                                            width=4,
                                            style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'marginRight': '5px'
                                        }
                                        ),
                                        dbc.Col(
                                            dcc.Graph(id='top-songs-chart',style={'height': '300px'},config={'responsive': True}),
                                            width=4,
                                            style={
                                            'backgroundColor': '#f9f0f1',  
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '100%',
                                            'marginRight': '5px'
                                        }
                                        ),
                                        dbc.Col(
                                         dcc.Graph(
                                            id='top-artist-chart',
                                            style={'height': '300px'},config={'responsive': True}
                                        ),
                                            style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng 
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '100%',
                                            'marginRight': '5px'
                                        })
                                    ],style={'marginTop': '20px','justifyContent': 'center'}
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Graph(id='network-song', style={'height': '400px'},config={'responsive': True})
                                            ],
                                            width=6,
                                            style={
                                                'backgroundColor': '#f9f0f1',  # Nền trắng 
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'height': '100%',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col([
                                            html.Label("Chọn genre:", style={'fontWeight': 'bold', 'color': "#020C05",'marginBottom': '3px'}),
                                            dcc.Dropdown(
                                                id='genre-filter',
                                                options=[{'label': genre, 'value': genre} for genre in sorted(df_songs['genre'].unique())] + [{'label': 'All', 'value': 'All'}],
                                                value='All',
                                                multi=True,
                                                placeholder="Tất cả thể loại",
                                                style={
                                                    'backgroundColor': '#f9f0f1',
                                                    'color': "#020C05",
                                                    'padding': '4px 8px',  # 👈 Ít padding hơn
                                                    'fontFamily': 'Montserrat',
                                                    'fontWeight': '400',
                                                    'fontSize': '13px',  # 👈 Nhỏ chữ dropdown
                                                    'marginBottom': '8px'
                                                }
                                            ),

                                            dcc.Graph(id='popularity-spread-correlation', style={'height': '400px'}, config={'responsive': True})
                                        ],
                                        width=5,
                                        style={
                                            'backgroundColor': '#f9f0f1',
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '100%',
                                            'marginRight': '5px'
                                        })

                                    ],
                                    style={'marginTop': '10px', 'marginBottom': '10px','justifyContent': 'center'}
                                )
                            ]
                        ),
                        
                        # Tab 2: Phân tích cộng đồng
                        dcc.Tab(
                            label='Genre Analysis',
                            value='community',
                             style={
                                    'backgroundColor': '#fff',
                                    'color': '#191414',
                                    'border': f'2px solid {SPOTIFY_COLORS["gray"]}',
                                    'padding': '6px 12px',
                                    'fontWeight': 'bold',
                                    'fontFamily': 'Montserrat',
                                    'fontSize': '10px',
                                    'height': '36px',
                                    'lineHeight': '24px',
                                    'boxShadow': 'none',
                                    'marginBottom': '0px'
                                },
                                selected_style={
                                    'backgroundColor': SPOTIFY_COLORS['darker_green'],
                                    'color': SPOTIFY_COLORS['white'],
                                    'border': f'2px solid {SPOTIFY_COLORS["light_green"]}',
                                    'padding': '6px 12px',
                                    'fontWeight': 'bold',
                                    'fontFamily': 'Montserrat',
                                    'fontSize': '10px',
                                    'height': '36px',
                                    'lineHeight': '24px',
                                    'boxShadow': 'none',
                                    'marginBottom': '0px'
                                },
                            children=[
                                 # Hidden div to store intermediate data
                                html.Div(id='community-data-store', style={'display': 'none'}),

                                # ✅ Thêm Store để lưu trạng thái chỉ số recommendation
                                dcc.Store(id='recommendation-index-store', data=0),
                                # genre Selection Dropdown
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id='community-dropdown',
                                            options=[{'label': f'genre {i}', 'value': i} 
                                                    for i in sorted(df_songs['genre'].unique())],
                                            value=sorted(df_songs['genre'].unique())[0],
                                            style={'color': 'black'}
                                        ),
                                        width=4,
                                        style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            'marginRight': '5px'
                                        }
                                    ),
                                    style={'marginTop': '20px'}
                                ),
                                
                                # First Row: Overview and Genre Distribution
                                dbc.Row([
                                # 1️⃣ Cột 1: Audio Chart + Mean Lambda
                                dbc.Col(
                                    html.Div([
                                        dcc.Graph(
                                            id='community-audio-features',
                                            figure={
                                                'data': [],
                                                'layout': {
                                                    'title': 'Audio Features Overview',
                                                    'plot_bgcolor': '#f9f0f1',
                                                    'paper_bgcolor': '#f9f0f1',
                                                    'font': {'color': 'black'},
                                                    'margin': {'t': 50, 'b': 30, 'l': 30, 'r': 30}
                                                }
                                            },
                                            style={'height': '350px'}
                                        ),
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Mean Lambda", className="card-title", style={'fontWeight': 'bold'}),
                                                html.H2(id='lambda-mean-display', className="card-text",
                                                        style={'color': '#1DB954', 'fontWeight': 'bold'})
                                            ])
                                        ],
                                        style={
                                            'backgroundColor': '#f9f0f1',
                                            'borderRadius': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'padding': '10px',
                                            'height': '100px',
                                            'textAlign': 'center',
                                            'marginTop': '5px'
                                        })
                                    ]),
                                    width=4,
                                    style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            'marginRight': '5px'
                                        }
                                ),

                                # 2️⃣ Cột 2: Sankey + Total Songs
                                dbc.Col(
                                    html.Div([
                                        dcc.Graph(
                                            id='community-sankey-distribution',
                                            style={'height': '350px'},
                                            figure={
                                                'data': [],
                                                'layout': {
                                                    'title': 'Genre Distribution',
                                                    'plot_bgcolor': '#f9f0f1',
                                                    'paper_bgcolor': '#f9f0f1',
                                                    'font': {'color': 'black'},
                                                    'margin': {'t': 50, 'b': 30, 'l': 30, 'r': 30}
                                                }
                                            }
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    html.I(className="fas fa-music fa-2x", 
                                                                        style={
                                                                            'color': 'white',
                                                                            'background': '#1DB954',
                                                                            'padding': '15px',
                                                                            'borderRadius': '50%',
                                                                            'boxShadow': '0 4px 8px rgba(29, 185, 84, 0.3)'
                                                                        }),
                                                                    style={'marginRight': '2px','flexShrink': '0', 'textAlign': 'center'}
                                                                ),
                                                                html.Div([
                                                                html.H6("TOTAL SONGS", 
                                                                    className="card-title", 
                                                                    style={
                                                                        'fontWeight': '600',
                                                                        'letterSpacing': '1.5px',
                                                                        'color': '#6c757d',
                                                                        'fontSize': '0.9rem',
                                                                        'marginBottom': '5px'
                                                                    }),
                                                                html.H2(
                                                                    id='total-songs-display',
                                                                    className="card-text",
                                                                    style={
                                                                        'color': '#212529', 
                                                                        'fontWeight': '600',
                                                                        'fontSize': '2rem',
                                                                        'marginBottom': '0',
                                                                        'textAlign': 'center'
                                                                    }
                                                                )
                                                        
                                                            ], style={'flexGrow': '1'}
            
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center'
                                                    }
                                                )
                                            ])
                                            ],
                                        style={
                                            'backgroundColor': '#ffffff',
                                            'borderRadius': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'padding': '10px',
                                            'height': '100px',
                                            'textAlign': 'center',
                                            'marginTop': '5px'
                                        })
                                    ]), width=5,
                                    style={
                                            'backgroundColor': "#ffffff",  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            'marginRight': '5px'
                                        }
                                ),

                                # 3️⃣ Cột 3: Recommended Songs
                                dbc.Col(
                                    html.Div([
                                        html.H4("Recommended Songs", style={
                                            'color': '#000000',
                                            'fontWeight': 'bold',
                                            'fontFamily': 'Montserrat',
                                            'textAlign': 'center',
                                            'fontSize': '15px'
                                        }),
                                        html.Div([
                                            dbc.Button("← Bài trước", id="recommendation-prev-button", n_clicks=0,
                                                    color="light", className="me-2",
                                                    style={'width': '100%', 'fontSize': '12px', 'marginBottom': '5px'}),
                                            dbc.Button("Bài sau →", id="recommendation-next-button", n_clicks=0,
                                                    color="primary",
                                                    style={'width': '100%', 'fontSize': '12px'})
                                        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '5px'}),
                                        html.Div(id='community-song-recommendations',
                                                style={'marginTop': '10px'})
                                    ]),
                                    width=2,
                                    style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            'marginRight': '5px'
                                        }
                                )
                            ], style={'marginTop': '20px','justifyContent': 'center'}),
                                
                                # Second Row: Feature Comparison and Top Artists
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(
                                            id='community-feature-comparison',style={'height': '350px'},
                                            figure={
                                                'data': [],
                                                'layout': {
                                                    'title': 'Feature Comparison by Genre',
                                                    'plot_bgcolor': SPOTIFY_COLORS['dark_gray'],
                                                    'paper_bgcolor': SPOTIFY_COLORS['dark_gray'],
                                                    'font': {'color': 'white'},
                                                    'margin': {'t': 50, 'b': 30, 'l': 30, 'r': 30}
                                                }
                                            }
                                        ),
                                        width=6,
                                        style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            'marginRight': '5px'
                                        }
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id='community-top-artists', style={'height': '300px'},
                                            figure={
                                                'data': [],
                                                'layout': {
                                                    'title': 'Top Artists in genre',
                                                    'plot_bgcolor': SPOTIFY_COLORS['dark_gray'],
                                                    'paper_bgcolor': SPOTIFY_COLORS['dark_gray'],
                                                    'font': {'color': 'white'},
                                                     'margin': {'t': 50, 'b': 30, 'l': 30, 'r': 30}
                                                }
                                            }
                                        ),
                                        width=5,
                                        style={
                                            'backgroundColor': '#f9f0f1',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'marginRight': '5px'
                                        }
                                    )
                                ], style={'marginTop': '20px', 'marginRight': '5px', 'display': 'flex', 'gap': '30px'}),
                            
                                # Hidden div to store intermediate data
                                html.Div(id='community-data-store', style={'display': 'none'})
                            ]
                        ),
                        
                        # Tab 3: Phân tích rủi ro
                        dcc.Tab(
                        label='Risk Analysis',
                        value='risk',
                        style={
                            'backgroundColor': '#fff',
                            'color': '#191414',
                            'border': f'1px solid {SPOTIFY_COLORS["gray"]}',
                            'padding': '8px 16px',
                            'fontWeight': '600',
                            'fontFamily': 'Montserrat, sans-serif',
                            'fontSize': '12px',
                            'height': '40px',
                            'lineHeight': '24px',
                            'borderRadius': '8px 8px 0 0',
                            'marginRight': '5px'
                        },
                        selected_style={
                                    'backgroundColor': SPOTIFY_COLORS['darker_green'],
                                    'color': SPOTIFY_COLORS['white'],
                                    'border': f'2px solid {SPOTIFY_COLORS["light_green"]}',
                                    'padding': '6px 12px',
                                    'fontWeight': 'bold',
                                    'fontFamily': 'Montserrat',
                                    'fontSize': '10px',
                                    'height': '36px',
                                    'lineHeight': '24px',
                                    'boxShadow': 'none',
                                    'marginBottom': '0px'
                                },
                        children=[
                            # Container chính để căn lề
                            html.Div(
                                style={
                                    'padding': '20px',
                                    'backgroundColor': '#fff',
                                    'borderRadius': '0 8px 8px 8px',
                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}',
                                    'borderTop': 'none'
                                },
                                children=[
                                    # Scenario description - updated to mention songs
                                    dbc.Row([
                                        dbc.Col(html.P(
                                            "🔁 Mô phỏng lan truyền ảnh hưởng khi một bài hát, nghệ sĩ hoặc thể loại bị 'shock' (tẩy chay, scandal, giảm thị hiếu).",
                                            style={
                                                'color': '#191414',
                                                'fontSize': '14px',
                                                'fontFamily': 'Montserrat, sans-serif',
                                                'marginBottom': '20px'
                                            }
                                        ))
                                    ]),

                                    # Dropdowns for song, artist and genre selection - added song dropdown
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='song-risk-dropdown',
                                                options=[{'label': song, 'value': song} for song in sorted(df_songs['track_name'].unique())],
                                                multi=True,
                                                placeholder="Chọn bài hát trung tâm",
                                                style={
                                                    'color': '#191414',
                                                    'fontFamily': 'Montserrat',
                                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                                },
                                                clearable=False
                                            ),
                                            width=3,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='center-artists',
                                                options=[{'label': a, 'value': a} for a in sorted(df_songs['artist_names'].unique())],
                                                multi=True,
                                                placeholder="Chọn nghệ sĩ trung tâm",
                                                style={
                                                    'color': '#191414',
                                                    'fontFamily': 'Montserrat',
                                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                                }
                                            ),
                                            width=3,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='genre-risk-dropdown',
                                                options=[{'label': g, 'value': g} for g in sorted(df_songs['genre'].unique())],
                                                placeholder="Chọn thể loại để mô phỏng rủi ro",
                                                style={
                                                    'color': '#191414',
                                                    'fontFamily': 'Montserrat',
                                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                                }
                                            ),
                                            width=3,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id='risk-severity',
                                                options=[
                                                    {'label': 'Mức độ nhẹ', 'value': 'low'},
                                                    {'label': 'Mức độ trung bình', 'value': 'medium'},
                                                    {'label': 'Mức độ cao', 'value': 'high'}
                                                ],
                                                value='medium',
                                                clearable=False,
                                                style={
                                                    'color': '#191414',
                                                    'fontFamily': 'Montserrat',
                                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                                }
                                            ),
                                            width=2,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)'
                                            }
                                        )
                                    ], style={'marginBottom': '20px'}),

                                    # Contagion network and genre impact graphs
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(
                                                id='sensitivity-network',
                                                style={'height': '400px', 'borderRadius': '8px'}
                                            ),
                                            width=6,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col(
                                            dcc.Graph(
                                                id='community-impact-bar',
                                                style={'height': '400px', 'borderRadius': '8px'}
                                            ),
                                            width=5,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginLeft': '5px'
                                            }
                                        )
                                    ], style={'marginBottom': '20px'}),

                                    # Influence heatmap and risk prediction
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(
                                                id='influence-heatmap',
                                                style={'height': '400px', 'borderRadius': '8px'}
                                            ),
                                            width=6,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginRight': '5px'
                                            }
                                        ),
                                        dbc.Col(
                                            dcc.Graph(
                                                id='risk-prediction',
                                                style={'height': '400px', 'borderRadius': '8px'}
                                            ),
                                            width=5,
                                            style={
                                                'backgroundColor': '#f9f0f1',
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'marginLeft': '5px'
                                            }
                                        )
                                    ], style={'marginTop': '20px'}),

                                    # High risk songs table - updated column names to Vietnamese
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5(
                                                "Top bài hát có nguy cơ cao",
                                                style={
                                                    'color': '#191414',
                                                    'fontFamily': 'Montserrat',
                                                    'marginBottom': '10px',
                                                    'marginTop': '20px'
                                                }
                                            ),
                                            width=12
                                        ),
                                        dbc.Col(
                                            dash_table.DataTable(
                                                id='high-risk-songs',
                                                columns=[
                                                    {'name': 'Bài hát', 'id': 'track_name', 'type': 'text'},
                                                    {'name': 'Nghệ sĩ', 'id': 'artist_names', 'type': 'text'},
                                                    {'name': 'Thể loại', 'id': 'genre', 'type': 'text'},
                                                    {'name': 'Điểm ảnh hưởng', 'id': 'influence', 'type': 'numeric',
                                                    'format': {'specifier': '.2f'}}
                                                ],
                                                style_table={
                                                    'overflowX': 'auto',
                                                    'borderRadius': '8px',
                                                    'border': f'1px solid {SPOTIFY_COLORS["gray"]}',
                                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                                                },
                                                style_cell={
                                                    'textAlign': 'left',
                                                    'backgroundColor': '#fff',
                                                    'color': '#191414',
                                                    'border': f'1px solid {SPOTIFY_COLORS["light_gray"]}',
                                                    'padding': '12px',
                                                    'fontFamily': 'Montserrat',
                                                    'fontSize': '13px'
                                                },
                                                style_header={
                                                    'backgroundColor': SPOTIFY_COLORS['darker_green'],
                                                    'color': '#fff',
                                                    'fontWeight': '600',
                                                    'border': f'1px solid {SPOTIFY_COLORS["light_green"]}'
                                                },
                                                style_data_conditional=[
                                                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                                                    {'if': {'row_index': 'even'}, 'backgroundColor': '#fff'},
                                                    {
                                                        'if': {'state': 'active'},
                                                        'backgroundColor': SPOTIFY_COLORS['light_green'],
                                                        'color': '#fff'
                                                    },
                                                    {
                                                        'if': {'column_id': 'influence', 'filter_query': '{influence} > 0.7'},
                                                        'backgroundColor': '#FFEBEE',
                                                        'color': '#C62828',
                                                        'fontWeight': 'bold'
                                                    }
                                                ],
                                                page_size=10,
                                                sort_action='native',
                                                filter_action='native'
                                            ),
                                            width=12,
                                            style={
                                                'marginBottom': '30px'
                                            }
                                        )
                                    ])
                                ]
                            )
                        ]
                    )],
                    style={
                        'marginBottom': '20px',
                        'borderBottom': f'1px solid {SPOTIFY_COLORS["gray"]}',
                        'boxShadow': '0 4px 10px rgba(0,0,0,0.6)',  
                        'borderRadius': '20px',
                        'overflow': 'hidden' ,
                        'height': '40px',    
                        'minHeight': 'auto',
                        
                    }
                ),
                
                # Footer
                html.Footer(
                    [
                        html.P(
                            "© 2023 Spotify Influence Dashboard | Created with Dash",
                            style={
                                'color': SPOTIFY_COLORS['light_gray'],
                                'textAlign': 'center',
                                'marginTop': '50px'
                            }
                        )
                    ]
                )
            ]
        ),
        dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
    ]
)