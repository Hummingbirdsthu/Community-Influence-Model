from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table
import numpy as np
import networkx as nx
import base64
import requests
import dash_bootstrap_components as dbc
import community.community_louvain as community_louvain

# Màu sắc theo phong cách Spotify
SPOTIFY_COLORS = {
    'green': '#1DB954',
    'light_green': '#1ED760',
    'dark_green': '#1AA34A',
    'darker_green': '#178A3E',
    'black': '#191414',
    'dark_gray': '#212121',
    'gray': '#535353',
    'light_gray': '#B3B3B3',
    'lighter_gray': '#E5E5E5',
    'white': '#FFFFFF',
    'off_white': '#F8F8F8',
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
    df = pd.read_csv("merged_with_lambda.csv")
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
genre_insight = (
    df_songs.groupby('genre')
    .agg(
        count=('track_name', 'count'),
        lambda_mean=('Lambda', 'mean'),
        lambda_std=('Lambda', 'std'),
        popularity_mean=('popularity', 'mean'),
        energy_mean=('energy', 'mean'),
        valence_mean=('valence', 'mean'),
        danceability_mean=('danceability', 'mean')
    )
    .reset_index()
    .sort_values(by='lambda_mean', ascending=False)
)

genre_insight_rounded = genre_insight.copy()
genre_insight_rounded.iloc[:, 1:] = genre_insight_rounded.iloc[:, 1:].round(2)

genre_table_data = genre_insight_rounded[['genre', 'count', 'lambda_mean', 'popularity_mean']].rename(columns={
    'genre': 'Genre',
    'count': 'Số lượng bài hát',
    'lambda_mean': 'Lambda trung bình',
    'popularity_mean': 'Độ phổ biến trung bình'
}).to_dict('records')

# Tạo nội dung cho từng tab
overview_tab_content = dbc.Container([
    # Row: Image (left) + description and cards (right)
    dbc.Row([
        # Image column
        dbc.Col(
            html.Img(src='/assets/dfi_sasa-pekec_music-streaming.jpg',
                     style={
                         'width': '100%',
                         'height': '430px',
                         'borderRadius': '12px'
                     }),
            style={'padding': '0', 'background': 'transparent'},
            width=5
        ),

        # Right column: description + cards
        dbc.Col([
            # Description
            html.Div([
                html.H4("Tổng quan dữ liệu âm nhạc", style={
                    'fontFamily': 'Montserrat',
                    'fontWeight': '700',
                    'color': '#1DB954',
                    'marginBottom': '10px'
                }),
                html.P("Khám phá các chỉ số phổ biến của danh sách phát như mức độ phổ biến trung bình, thể loại nổi bật, thời lượng bài hát và số nghệ sĩ.",
                       style={
                           'fontFamily': 'Montserrat',
                           'fontSize': '14px',
                           'color': '#333'
                       })
            ], style={'marginBottom': '20px'}),

            # Main content row with original cards + new vertical card
            dbc.Row([
                # Original 4 cards (now in 8 columns)
                dbc.Col([
                    # First row of cards
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-fire", style={
                                        'color': '#FF6B6B',
                                        'fontSize': '1.5rem',
                                        'marginBottom': '10px'
                                    }),
                                    html.H6("AVG POPULARITY", style={
                                        'fontSize': '0.8rem',
                                        'color': '#6c757d',
                                        'fontWeight': '600'
                                    }),
                                    html.H4("72.5", style={
                                        'color': '#FF6B6B',
                                        'fontWeight': 'bold'
                                    })
                                ])
                            ], style={
                                'backgroundColor': '#fff',
                                'borderLeft': '4px solid #FF6B6B',
                                'borderRadius': '10px',
                                'padding': '15px',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)'
                            }),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-tag", style={
                                        'color': '#6C5CE7',
                                        'fontSize': '1.5rem',
                                        'marginBottom': '10px'
                                    }),
                                    html.H6("TOP GENRE", style={
                                        'fontSize': '0.8rem',
                                        'color': '#6c757d',
                                        'fontWeight': '600'
                                    }),
                                    html.H4("Pop", style={
                                        'color': '#6C5CE7',
                                        'fontWeight': 'bold'
                                    })
                                ])
                            ], style={
                                'backgroundColor': '#fff',
                                'borderLeft': '4px solid #6C5CE7',
                                'borderRadius': '10px',
                                'padding': '15px',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)'
                            }),
                            width=6
                        )
                    ], style={'marginBottom': '15px'}),

                    # Second row of cards
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-clock", style={
                                        'color': '#00B894',
                                        'fontSize': '1.5rem',
                                        'marginBottom': '10px'
                                    }),
                                    html.H6("AVG DURATION", style={
                                        'fontSize': '0.8rem',
                                        'color': '#6c757d',
                                        'fontWeight': '600'
                                    }),
                                    html.H4("3:12", style={
                                        'color': '#00B894',
                                        'fontWeight': 'bold'
                                    })
                                ])
                            ], style={
                                'backgroundColor': '#fff',
                                'borderLeft': '4px solid #00B894',
                                'borderRadius': '10px',
                                'padding': '15px',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)'
                            }),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.I(className="fas fa-users", style={
                                        'color': '#FD79A8',
                                        'fontSize': '1.5rem',
                                        'marginBottom': '10px'
                                    }),
                                    html.H6("UNIQUE ARTISTS", style={
                                        'fontSize': '0.8rem',
                                        'color': '#6c757d',
                                        'fontWeight': '600'
                                    }),
                                    html.H4("148", style={
                                        'color': '#FD79A8',
                                        'fontWeight': 'bold'
                                    })
                                ])
                            ], style={
                                'backgroundColor': '#fff',
                                'borderLeft': '4px solid #FD79A8',
                                'borderRadius': '10px',
                                'padding': '15px',
                                'boxShadow': '0 2px 6px rgba(0,0,0,0.05)'
                            }),
                            width=6
                        )
                    ])
                ], width=8),  # Reduced from 12 to 8 columns
                
                # New vertical card (4 columns)
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-chart-line", style={
                                    'color': '#1DB954',
                                    'fontSize': '2rem',
                                    'marginBottom': '15px',
                                    'display': 'block',
                                    'textAlign': 'center'
                                }),
                                html.H5("Trending Now", style={
                                    'textAlign': 'center',
                                    'color': '#1DB954',
                                    'fontWeight': '600',
                                    'marginBottom': '20px'
                                }),
                                html.Div([
                                    html.Div([
                                        html.Span("1. ", style={'fontWeight': 'bold'}),
                                        html.Span("Blinding Lights", style={'fontWeight': '500'})
                                    ], className="mb-2"),
                                    dbc.Progress(value="95", max="100", style={
                                        'height': '6px',
                                        'backgroundColor': '#e9ecef',
                                        'marginBottom': '15px'
                                    }, className="bg-success"),
                                    
                                    html.Div([
                                        html.Span("2. ", style={'fontWeight': 'bold'}),
                                        html.Span("Save Your Tears", style={'fontWeight': '500'})
                                    ], className="mb-2"),
                                    dbc.Progress(value="88", max="100", style={
                                        'height': '6px',
                                        'backgroundColor': '#e9ecef',
                                        'marginBottom': '15px'
                                    }, className="bg-success"),
                                    
                                    html.Div([
                                        html.Span("3. ", style={'fontWeight': 'bold'}),
                                        html.Span("Stay", style={'fontWeight': '500'})
                                    ], className="mb-2"),
                                    dbc.Progress(value="82", max="100", style={
                                        'height': '6px',
                                        'backgroundColor': '#e9ecef',
                                        'marginBottom': '15px'
                                    }, className="bg-success"),
                                    
                                    html.Div([
                                        html.Span("4. ", style={'fontWeight': 'bold'}),
                                        html.Span("good 4 u", style={'fontWeight': '500'})
                                    ], className="mb-2"),
                                    dbc.Progress(value="76", max="100", style={
                                        'height': '6px',
                                        'backgroundColor': '#e9ecef'
                                    }, className="bg-success")
                                ], style={'padding': '0 10px'})
                            ])
                        ])
                    ], style={
                        'height': '100%',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 6px rgba(0,0,0,0.05)',
                        'borderTop': '4px solid #1DB954'
                    }),
                    width=4  # Takes 4 columns
                )
            ])
        ], width=7)
    ], style={'marginTop': '20px', 'marginBottom': '30px'}),
    # Charts row 1
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='genre-influence-chart',
                     figure={'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar'}]},
                     style={'height': '300px'},
                     config={'responsive': True}),
            width=4,
            style={
                'backgroundColor': '#ffffff',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                'marginBottom': '15px'
            }
        ),
        dbc.Col(
            dcc.Graph(id='top-songs-chart',
                     figure={'data': [{'x': [1, 2, 3], 'y': [2, 4, 3], 'type': 'bar'}]},
                     style={'height': '300px'},
                     config={'responsive': True}),
            width=4,
            style={
                'backgroundColor': '#ffffff',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                'marginBottom': '15px'
            }
        ),
        dbc.Col(
            dcc.Graph(id='top-artist-chart',
                     figure={'data': [{'x': [1, 2, 3], 'y': [3, 2, 5], 'type': 'bar'}]},
                     style={'height': '300px'},
                     config={'responsive': True}),
            width=4,
            style={
                'backgroundColor': '#ffffff',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                'marginBottom': '15px'
            }
        )
    ], style={'marginBottom': '20px'}),

    # Charts row 2
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='network-song',
                     figure={'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'line'}]},
                     style={'height': '400px'},
                     config={'responsive': True}),
            width=6,
            style={
                'backgroundColor': '#ffffff',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                'marginBottom': '15px'
            }
        ),
        dbc.Col([
            html.Label("Thống kê Lambda theo thể loại", style={
                'fontWeight': 'bold', 'color': "#020C05", 'marginBottom': '6px'
            }),
            dash_table.DataTable(
                id='genre-lambda-table',
                data=genre_table_data,
                columns=[
                    {'name': 'Genre', 'id': 'Genre'},
                    {'name': 'Số lượng bài hát', 'id': 'Số lượng bài hát'},
                    {'name': 'Lambda trung bình', 'id': 'Lambda trung bình'},
                    {'name': 'Độ phổ biến trung bình', 'id': 'Độ phổ biến trung bình'}
                ],
                style_table={'overflowY': 'auto', 'height': '350px'},
                fixed_rows={'headers': True},
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': 'Montserrat',
                    'color': 'black',
                    'fontSize': '13px',
                    'padding': '6px',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Genre'}, 'width': '120px'},
                    {'if': {'column_id': 'Số lượng bài hát'}, 'width': '150px'},
                    {'if': {'column_id': 'Lambda trung bình'}, 'width': '150px'},
                    {'if': {'column_id': 'Độ phổ biến trung bình'}, 'width': '180px'}
                ],
                style_header={
                    'backgroundColor': '#1DB954',
                    'color': 'black',
                    'fontWeight': 'bold'
                },
                row_selectable='single',
                selected_rows=[],
            ),
            html.Div(id='genre-selected-info', style={'marginTop': '10px'})
        ], width=6,
            style={
                'backgroundColor': '#ffffff',
                'borderRadius': '8px',
                'padding': '15px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                'marginBottom': '15px'
            })
    ])
], fluid=True)

community_tab_content = dbc.Row([
# Hidden div to store intermediate data
                                html.Div(id='community-data-store', style={'display': 'none'}),

                                # ✅ Thêm Store để lưu trạng thái chỉ số recommendation
                                dcc.Store(id='recommendation-index-store', data=0),
                                # genre Selection Dropdown
dbc.Row(
    [
        dbc.Col(
            dbc.Row(
                [
                    # Col chứa ảnh + dropdown
                    dbc.Col(
                        html.Div([
                            html.Img(
                                src='/assets/musical-pentagram-sound-waves-notes-background.png',  # thay ảnh tùy bạn
                                style={
                                    'height': '50px',
                                    'width': '250px',
                                    'objectFit': 'cover',
                                    'marginBottom': '10px'
                                }
                            ),
                            dcc.Dropdown(
                                id='community-dropdown',
                                options=[
                                    {'label': f'Genre {i}', 'value': i}
                                    for i in sorted(df_songs['genre'].unique())
                                ],
                                value=sorted(df_songs['genre'].unique())[0],
                                style={'color': 'black','width': '200px',}
                            )
                        ],
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '20px',
                            'padding': '15px',
                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                            'height': '100px',
                            'width': '250px',# Cố định chiều cao để khớp với 2 card bên
                        }),
                        width=3,
                        style={'marginRight': '5px'}
                    ),

                    dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Div(
                                html.I("λ", style={
                                    'color': 'white',
                                    'background': '#1DB954',
                                    'padding': '15px',
                                    'borderRadius': '50%',
                                    'fontSize': '20px',
                                    'boxShadow': '0 4px 8px rgba(29, 185, 84, 0.3)'
                                }),
                                style={'marginRight': '10px', 'textAlign': 'center'}
                            ),
                            html.Div([
                                html.H6("MEAN LAMBDA",
                                        className="card-title",
                                        style={
                                            'fontWeight': '600',
                                            'letterSpacing': '1.5px',
                                            'color': '#6c757d',
                                            'fontSize': '0.9rem',
                                            'marginBottom': '5px'
                                        }),
                                html.H2(id='lambda-mean-display',
                                        className="card-text",
                                        style={
                                            'color': "#000000",
                                            'fontWeight': 'bold',
                                            'fontSize': '1.8rem'
                                        })
                            ])
                        ],
                        style={'display': 'flex', 'alignItems': 'center'})
                    ),
                    width=3,
                    style={
                        'backgroundColor': '#ffffff',
                        'borderRadius': '15px',
                        'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                        'padding': '15px',
                        'height': '120px',
                        'marginRight': '10px',
                        'borderTop': '4px solid #1DB954'  # Thêm viền trên màu xanh
                    }
                ),

                # Card 2: Total Songs (đã chỉnh sửa)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Div(
                                html.I(className="fas fa-music", style={
                                    'color': 'white',
                                    'background': '#1DB954',
                                    'padding': '15px',
                                    'borderRadius': '50%',
                                    'boxShadow': '0 4px 8px rgba(29, 185, 84, 0.3)'
                                }),
                                style={'marginRight': '10px', 'textAlign': 'center'}
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
                                html.H2(id='total-songs-display',
                                        className="card-text",
                                        style={
                                            'color': '#212529',
                                            'fontWeight': 'bold',
                                            'fontSize': '1.8rem'
                                        })
                            ])
                        ],
                        style={'display': 'flex', 'alignItems': 'center'})
                    ),
                    width=3,
                    style={
                        'backgroundColor': '#ffffff',
                        'borderRadius': '15px',
                        'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                        'padding': '15px',
                        'height': '120px',
                        'borderTop': '4px solid #1DB954'  # Thêm viền trên màu xanh
                    }
                )
                ],
                justify="center",
                align="center",
                className="g-3"
            ),
            width=12
        )
    ],
    justify="center",
    align="center",
    style={
        'marginTop': '30px',
        'width': '100%',
        'padding': '5px'
    }
),

                                
                                # First Row: Overview and Genre Distribution
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            dcc.Graph(
                                                id='distribution-by-channel',
                                                style={'height': '600px', 'backgroundColor': '#ffffff', 'borderRadius': '12px'},
                                                config={'responsive': True}
                                            ),

                                            html.Div([
                                                dcc.Dropdown(
                                                    id='attribute-filter',
                                                    options=[{'label': attr, 'value': attr} for attr in sorted(np.array(df_songs.columns)[12:21])],
                                                    value='danceability',
                                                    style={
                                                        'backgroundColor': 'transparent',
                                                        'color': "#000000",
                                                        'fontFamily': 'Montserrat',
                                                        'fontSize': '14px',
                                                        'borderRadius': '6px'
                                                    }
                                                )
                                            ], style={
                                                'position': 'absolute',
                                                'top': '20px',
                                                'right': '20px',
                                                'width': '210px',
                                                'zIndex': '1000'
                                            })
                                        ], style={
                                            'position': 'relative',
                                            #'padding': '1em',
                                            'backgroundColor': 'rgba(255,255,255,0.04)',
                                            'borderRadius': '12px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.05)',
                                            'marginTop': '1em'
                                        }),
                                    ], width=7),

                                    dbc.Col([
                                        dbc.Row([
                                            dcc.Graph(
                                                id='density-chart',
                                                style={'height': '260px', 'backgroundColor': '#ffffff'},
                                                config={'responsive': True}
                                            )
                                        ], style={
                                            'marginBottom': '1em',
                                            'padding': '1em',
                                            'backgroundColor': '#ffffff',
                                            'borderRadius': '12px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'marginTop': '1em',
                                            'width': '550px'
                                        }),

                                        dbc.Row([
                                            html.Div([
                                                dcc.Graph(
                                                    id='popularity-by-genre-chart',
                                                    style={'height': '260px', 'backgroundColor': '#ffffff'},
                                                    config={'responsive': True}
                                                ),

                                                html.Div([
                                                    dcc.Dropdown(
                                                        id='y-attribute-filter',
                                                        options=[{'label': attr, 'value': attr} for attr in sorted(np.array(df_songs.columns)[12:21])],
                                                        value='popularity',
                                                         style={
                                                            'backgroundColor': 'transparent',
                                                            'color': "#000000",
                                                            'fontFamily': 'Montserrat',
                                                            'fontSize': '13px',
                                                            'borderRadius': '6px',
                                                            'height': '42px',         # 🔺 Chiều cao dropdown
                                                            'lineHeight': '42px',     # 🔺 Để text nằm giữa theo chiều dọc
                                                            'paddingLeft': '8px'      # (Tùy chọn) cho dễ đọc
                                                        }
                                                    )
                                                ],style={
                                                    'position': 'absolute',
                                                    'top': '1px',         # 🔺 Cao hơn nữa so với '30px'
                                                    'right': '20px',
                                                    'width': '180px',
                                                    'zIndex': '1000'
                                                })
                                            ], style={'position': 'relative'})
                                        ], style={
                                            'padding': '1em',
                                            'backgroundColor': '#ffffff',
                                            'borderRadius': '12px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'width': '550px'
                                        })
                                    ], width=4)
                                ], style={
                                    'gap': '10px',
                                    'padding': '10px',
                                    #'borderRadius': '10px',
                                    #'border': '2px solid rgba(179,179,179,0.4)',
                                    'boxShadow': '0 4px 6px rgba(255,255,255,0.05)',
                                    'background': 'transparent',
                                    'backdropFilter': 'blur(10px)',
                                    '-webkitBackdropFilter': 'blur(50px)'
                                }),
                                
                                # Second Row: Feature Comparison and Top Artists
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(
                                            id='community-distribution-map',style={'height': '490px'},
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
                                            'backgroundColor': '#ffffff',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            'height': '90%',
                                            #'marginRight': '5px'
                                        }
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id='community-top-artists', style={'height': '390px'},
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
                                        width=3,
                                        style={
                                            'backgroundColor': '#ffffff',  # Nền trắng
                                            'borderRadius': '20px',
                                            'padding': '15px',
                                            'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                            #'marginRight': '5px'
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
                                                'backgroundColor': '#ffffff',  # Nền trắng
                                                'borderRadius': '20px',
                                                'padding': '15px',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.06)',
                                                'height': '480px',
                                                #'marginRight': '5px'
                                            }
                                    )
                                ], style={'marginTop': '20px', 'display': 'flex', 'gap': '30px'}),
                            
                                # Hidden div to store intermediate data
                                html.Div(id='community-data-store', style={'display': 'none'})
])

risk_tab_content = dbc.Row([
    html.Div(
        style={
            'padding': '25px',
            'backgroundColor': '#fff',
            'borderRadius': '0 8px 8px 8px',
            'border': f'1px solid {SPOTIFY_COLORS["gray"]}',
            'borderTop': 'none',
            'boxShadow': '0 2px 15px rgba(0,0,0,0.05)'
        },
        children=[
            # Header Section
            dbc.Row([
                dbc.Col([
                    html.H4(
                        "Phân tích Rủi ro Lan truyền",
                        style={
                            'color': '#191414',
                            'fontSize': '20px',
                            'fontFamily': 'Montserrat, sans-serif',
                            'fontWeight': '600',
                            'marginBottom': '10px'
                        }
                    ),
                    html.P(
                        "🔁 Mô phỏng lan truyền ảnh hưởng khi một bài hát hoặc nghệ sĩ bị 'shock' (tẩy chay, scandal, giảm thị hiếu).",
                        style={
                            'color': '#666',
                            'fontSize': '14px',
                            'fontFamily': 'Montserrat, sans-serif',
                            'marginBottom': '25px'
                        }
                    )
                ], width=12)
            ]),
            
            # Control Panel Section
            dbc.Row([
                dbc.Col([
                    html.Div(
                        style={
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '8px',
                            'padding': '20px',
                            'marginBottom': '25px'
                        },
                        children=[
                            html.H6(
                                "Thiết lập Mô phỏng",
                                style={
                                    'color': '#191414',
                                    'fontFamily': 'Montserrat',
                                    'marginBottom': '15px',
                                    'fontWeight': '600'
                                }
                            ),
                            dbc.Row([
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='song-risk-dropdown',
                                        options=[
                                            {'label': f"{row['track_name']} (λ: {row['Lambda']:.2f})", 
                                             'value': row['track_name']}
                                            for _, row in df_songs.sort_values(by='Lambda', ascending=False).iterrows()
                                        ],
                                        multi=True,
                                        placeholder="Chọn bài hát trung tâm...",
                                        style={
                                            'color': '#191414',
                                            'fontFamily': 'Montserrat',
                                            'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                        },
                                        clearable=False
                                    ),
                                    width=6,
                                    style={'paddingRight': '10px'}
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id='center-artists',
                                        options=[{'label': a, 'value': a} for a in sorted(df_songs['artist_names'].unique())],
                                        multi=True,
                                        placeholder="Chọn nghệ sĩ trung tâm...",
                                        style={
                                            'color': '#191414',
                                            'fontFamily': 'Montserrat',
                                            'border': f'1px solid {SPOTIFY_COLORS["gray"]}'
                                        }
                                    ),
                                    width=6,
                                    style={'paddingLeft': '10px'}
                                )
                            ]),
                            dbc.Row([
                                dbc.Col(
                                    html.Div(
                                        dcc.RadioItems(
                                            id='network-view-toggle',
                                            options=[
                                                {'label': html.Span(['🟢 Mạng gốc (Trước khi xóa)'], 
                                                  style={'fontFamily': 'Montserrat'}),
                                                 'value': 'before'},
                                                {'label': html.Span(['🔴 Mạng sau khi xóa'], 
                                                  style={'fontFamily': 'Montserrat'}),
                                                 'value': 'after'}
                                            ],
                                            value='after',
                                            labelStyle={
                                                'display': 'inline-flex',
                                                'alignItems': 'center',
                                                'marginRight': '20px',
                                                'cursor': 'pointer'
                                            },
                                            inputStyle={'marginRight': '5px'},
                                            style={'marginTop': '15px'}
                                        ),
                                        style={'padding': '10px 0'}
                                    ),
                                    width=12
                                )
                            ])
                        ]
                    )
                ], width=12)
            ]),
            
            # Visualization Section
            dbc.Row([
                # Network Graph
                dbc.Col(
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                            'height': '100%'
                        },
                        children=[
                            html.H6(
                                "Mạng lưới Ảnh hưởng",
                                style={
                                    'fontFamily': 'Montserrat',
                                    'fontWeight': '600',
                                    'color': '#191414',
                                    'marginBottom': '15px'
                                }
                            ),
                            dcc.Graph(
                                id='sensitivity-network',
                                style={'height': '400px', 'borderRadius': '4px'}
                            )
                        ]
                    ),
                    width=6,
                    style={'paddingRight': '10px'}
                ),
                
                # Community Impact
                dbc.Col(
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                            'height': '100%'
                        },
                        children=[
                            html.H6(
                                "Ảnh hưởng đến Cộng đồng",
                                style={
                                    'fontFamily': 'Montserrat',
                                    'fontWeight': '600',
                                    'color': '#191414',
                                    'marginBottom': '15px'
                                }
                            ),
                            dcc.Graph(
                                id='community-impact-bar',
                                style={'height': '400px', 'borderRadius': '4px'}
                            )
                        ]
                    ),
                    width=6,
                    style={'paddingLeft': '10px'}
                )
            ], style={'marginBottom': '25px'}),
            
            dbc.Row([
                # Heatmap
                dbc.Col(
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                            'height': '100%'
                        },
                        children=[
                            html.H6(
                                "Bản đồ Nhiệt Ảnh hưởng",
                                style={
                                    'fontFamily': 'Montserrat',
                                    'fontWeight': '600',
                                    'color': '#191414',
                                    'marginBottom': '15px'
                                }
                            ),
                            dcc.Graph(
                                id='influence-heatmap',
                                style={'height': '400px', 'borderRadius': '4px'}
                            )
                        ]
                    ),
                    width=6,
                    style={'paddingRight': '10px'}
                ),
                
                # Risk Prediction
                dbc.Col(
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '8px',
                            'padding': '15px',
                            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                            'height': '100%'
                        },
                        children=[
                            html.H6(
                                "Dự đoán Rủi ro Lan truyền",
                                style={
                                    'fontFamily': 'Montserrat',
                                    'fontWeight': '600',
                                    'color': '#191414',
                                    'marginBottom': '15px'
                                }
                            ),
                            dcc.Graph(
                                id='risk-prediction',
                                style={'height': '400px', 'borderRadius': '4px'}
                            )
                        ]
                    ),
                    width=6,
                    style={'paddingLeft': '10px'}
                )
            ], style={'marginBottom': '25px'}),
            
            # High Risk Songs Table
            dbc.Row([
                dbc.Col(
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'borderRadius': '8px',
                            'padding': '20px',
                            'boxShadow': '0 4px 12px rgba(0,0,0,0.08)'
                        },
                        children=[
                            html.H6(
                                "Top bài hát có nguy cơ cao",
                                style={
                                    'fontFamily': 'Montserrat',
                                    'fontWeight': '600',
                                    'color': '#191414',
                                    'marginBottom': '15px'
                                }
                            ),
                            dash_table.DataTable(
                                id='high-risk-songs',
                                columns=[
                                    {'name': 'Bài hát', 'id': 'track_name', 'type': 'text'},
                                    {'name': 'Nghệ sĩ', 'id': 'artist_names', 'type': 'text'},
                                    {'name': 'Thể loại', 'id': 'genre', 'type': 'text'},
                                    {'name': 'Điểm ảnh hưởng (λ)', 'id': 'influence', 
                                     'type': 'numeric', 'format': {'specifier': '.2f'}}
                                ],
                                style_table={
                                    'overflowX': 'auto',
                                    'borderRadius': '4px',
                                    'border': 'none'
                                },
                                style_cell={
                                    'fontFamily': 'Montserrat',
                                    'padding': '12px',
                                    'textAlign': 'left',
                                    'border': 'none',
                                    'fontSize': '14px'
                                },
                                style_header={
                                    'backgroundColor': SPOTIFY_COLORS['green'],
                                    'color': 'white',
                                    'fontWeight': '600',
                                    'border': 'none',
                                    'textTransform': 'uppercase',
                                    'letterSpacing': '0.5px',
                                    'fontSize': '13px'
                                },
                                style_data={
                                    'backgroundColor': 'white',
                                    'color': '#191414',
                                    'borderBottom': f'1px solid {SPOTIFY_COLORS["light_gray"]}'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    {
                                        'if': {'column_id': 'influence', 'filter_query': '{influence} > 0.8'},
                                        'backgroundColor': '#ffebee',
                                        'color': '#c62828'
                                    }
                                ],
                                page_size=10,
                                sort_action='native',
                                filter_action='native',
                                sort_by=[{'column_id': 'influence', 'direction': 'desc'}]
                            )
                        ]
                    ),
                    width=12
                )
            ])
        ]
    )
])

layout_spotify = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src=SPOTIFY_LOGO, style={'height': '50px', 'marginRight': '15px'}),
                html.H1("Spotify Network Influence Analysis", 
                       style={
                           'color': 'white',
                           'fontWeight': 'bold',
                           'fontFamily': 'Montserrat',
                           'marginBottom': '0'
                       })
            ], style={'display': 'flex', 'alignItems': 'center', 'padding': '15px 0'})
        ], width=12)
    ], style={
        'backgroundColor': SPOTIFY_COLORS['black'],
        'padding': '0 30px',
        'marginBottom': '20px',
        'borderRadius': '8px'
    }),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Tổng quan", tab_id="overview", children=overview_tab_content),
        dbc.Tab(label="Cộng đồng", tab_id="community", children=community_tab_content),
        dbc.Tab(label="Đánh giá rủi ro", tab_id="risk", children=risk_tab_content)
    ], id="tabs", active_tab="overview"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.P(
                "© 2023 Spotify Network Analysis | Powered by Dash",
                style={
                    'color': SPOTIFY_COLORS['light_gray'],
                    'fontSize': '12px',
                    'textAlign': 'center',
                    'marginTop': '30px',
                    'fontFamily': 'Montserrat'
                }
            )
        ], width=12)
    ])
], fluid=True, style={'fontFamily': 'Montserrat', 'backgroundColor': SPOTIFY_COLORS['off_white']})