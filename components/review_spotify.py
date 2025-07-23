import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go

data = pd.read_csv("merged_spotify_data_clean.csv") 
df = pd.DataFrame(data)
df['track_duration_min'] = df['track_duration_ms'] / 60000
df['release_date'] = pd.to_datetime(df['release_date'], format='mixed', errors='coerce')
df['release_year'] = df['release_date'].dt.year

total_tracks = df['track_id'].nunique()
total_artists = df['artist_id'].nunique()
avg_popularity = df['popularity'].mean()
avg_duration_min = df['track_duration_min'].mean().round(2)
total_playlists = df['playlist_id'].nunique()

# Initialize Dash app with Spotify-like dark theme
app = dash.Dash(__name__,external_stylesheets=[
    dbc.themes.FLATLY,  
    'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'
]
)

# Spotify color scheme
SPOTIFY_COLORS = {
    'background': '#121212',
    'card': '#181818',
    'text': '#FFFFFF',
    'primary': '#1DB954',
    'secondary': '#535353',
    'highlight': '#1ED760'
}

# Custom styles
STYLES = {
    'font-family': 'Montserrat, sans-serif',
    'title': {
        'font-weight': '600',
        'color': SPOTIFY_COLORS['text']
    },
    'card': {
        'border-radius': '12px',
        'background-color': SPOTIFY_COLORS['card'],
        'box-shadow': '0 4px 6px rgba(0,0,0,0.3)',
        'padding': '20px',
        'margin-bottom': '20px',
        'border': 'none'
    },
    'header': {
        'background-color': 'transparent',
        'border-bottom': 'none',
        'color': SPOTIFY_COLORS['text'],
        'font-weight': '600',
        'text-transform': 'uppercase',
        'letter-spacing': '1px',
        'padding-bottom': '10px'
    },
    'metric': {
        'value': {
            'font-size': '28px',
            'font-weight': '700',
            'color': SPOTIFY_COLORS['text'],
            'margin': '10px 0'
        },
        'label': {
            'font-size': '14px',
            'color': SPOTIFY_COLORS['secondary'],
            'margin-bottom': '5px'
        },
        'progress': {
            'height': '6px',
            'border-radius': '3px',
            'margin': '10px 0'
        }
    }
}

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime

data = pd.read_csv("merged_spotify_data_clean.csv") 
df = pd.DataFrame(data)
df['track_duration_min'] = df['track_duration_ms'] / 60000
df['release_date'] = pd.to_datetime(df['release_date'], format='mixed', errors='coerce')
df['release_year'] = df['release_date'].dt.year

total_tracks = df['track_id'].nunique()
total_artists = df['artist_id'].nunique()
avg_popularity = df['popularity'].mean()
avg_duration_min = df['track_duration_min'].mean().round(2)
total_playlists = df['playlist_id'].nunique()

# Initialize Dash app with white theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.FLATLY,  
    'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
])

# Modern color scheme with white background
COLORS = {
    'background': 'transparent',
    'card': '#F8F9FA',
    'text': "#000000",
    'primary': '#1DB954',  # Keeping Spotify green as accent
    'secondary': '#6C757D',
    'highlight': '#1ED760',
    'border': '#E9ECEF',
    'success': '#28A745',
    'info': '#17A2B8',
    'warning': '#FFC107',
    'danger': '#DC3545'
}

# Custom styles for white theme
STYLES = {
    'font-family': 'Montserrat, sans-serif',
    'title': {
        'font-weight': '600',
        'color': COLORS['text']
    },
    'card': {
        'border-radius': '12px',
        'background-color': COLORS['card'],
        'box-shadow': '0 4px 6px rgba(0,0,0,0.05)',
        'padding': '20px',
        'margin-bottom': '20px',
        'border': f"1px solid {COLORS['border']}"
    },
    'header': {
        'background-color': 'transparent',
        'border-bottom': f"1px solid {COLORS['border']}",
        'color': COLORS['text'],
        'font-weight': '600',
        'text-transform': 'uppercase',
        'letter-spacing': '1px',
        'padding-bottom': '10px'
    },
    'metric': {
        'value': {
            'font-size': '28px',
            'font-weight': '700',
            'color': COLORS['text'],
            'margin': '10px 0'
        },
        'label': {
            'font-size': '14px',
            'color': COLORS['secondary'],
            'margin-bottom': '5px'
        },
        'progress': {
            'height': '6px',
            'border-radius': '3px',
            'margin': '10px 0'
        }
    }
}

def create_donut_chart(value, title, color=COLORS['primary'], total=None):
    if value is None:
        value = 0

    if total is None:
        if "Tracks" in title:
            total = max(value * 1.5, 1000)
        elif "Artists" in title:
            total = max(value * 1.5, 500)
        elif "Popularity" in title:
            total = 100
        elif "Duration" in title:
            total = 10
        else:
            total = value * 1.5 if value > 0 else 100

    value = min(value, total)

    fig = go.Figure(go.Pie(
        values=[value, total - value],
        hole=0.7,
        marker_colors=[color, COLORS['border']],
        textinfo='none',
        hoverinfo='none',
        rotation=90
    ))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[
            dict(
                text=f"<b>{round(value, 1)}</b>",
                x=0.5, y=0.5,
                font_size=24,
                showarrow=False,
                font_family="Montserrat",
                font_color=COLORS['text']
            ),
            dict(
                text=title,
                x=0.5, y=1.2,
                showarrow=False,
                font_family="Montserrat",
                font_size=14,
                font_color=COLORS['text']
            )
        ],
        height=180,
        width=180,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig
def create_track_card(track_data):
    if track_data is None:
        return html.Div(
            html.Div([
                html.Img(
                    src="https://cdn-icons-png.flaticon.com/512/727/727209.png",
                    style={
                        'width': '100%',
                        'opacity': '0.3',
                        'max-width': '200px',
                        'margin': '0 auto',
                        'filter': 'grayscale(100%)'
                    }
                ),
                html.P("Select a track to see details", style={
                    'color': '#888',
                    'text-align': 'center',
                    'margin-top': '20px',
                    'font-style': 'italic',
                    'font-size': '1.1rem'
                })
            ], style={
                'padding': '40px 0',
                'text-align': 'center',
                'background': 'rgba(245, 245, 245, 0.5)',
                'border-radius': '12px',
                'border': '2px dashed rgba(200, 200, 200, 0.5)'
            })
        )
    
    # Color palette
    accent_color = '#1DB954'  # Spotify green
    text_color = '#333333'
    secondary_text = '#666666'
    bg_color = '#FFFFFF'
    card_shadow = '0 8px 30px rgba(0,0,0,0.12)'
    
    return html.Div([
        html.Div([
            html.Img(
                src=track_data['spotify_cover_url'],
                style={
                    'width': '100%',
                    'border-radius': '12px',
                    'box-shadow': card_shadow,
                    'margin-bottom': '20px',
                    'transition': 'transform 0.3s ease',
                    'border': '1px solid rgba(0,0,0,0.05)'
                },
                className='hover-zoom'
            )
        ]),
        
        html.Div([
            html.H4(
                track_data['track_name'],
                style={
                    'color': text_color,
                    'margin-bottom': '5px',
                    'white-space': 'nowrap',
                    'overflow': 'hidden',
                    'text-overflow': 'ellipsis',
                    'font-weight': '700',
                    'font-size': '1.4rem'
                }
            ),
            html.P(
                track_data['artist_names'],
                style={
                    'color': accent_color,
                    'margin-bottom': '15px',
                    'font-weight': '600',
                    'font-size': '1.1rem'
                }
            ),
            
            html.Div([
                # Album and release date
                html.Div([
                    html.Div([
                        html.I(className="fas fa-compact-disc", style={
                            'color': secondary_text,
                            'margin-right': '8px',
                            'width': '16px',
                            'text-align': 'center'
                        }),
                        html.Span(track_data['album_name'], style={
                            'color': text_color,
                            'font-weight': '500'
                        })
                    ], style={
                        'display': 'flex',
                        'align-items': 'center',
                        'margin-bottom': '12px'
                    }),
                    html.Div([
                        html.I(className="fas fa-calendar-alt", style={
                            'color': secondary_text,
                            'margin-right': '8px',
                            'width': '16px',
                            'text-align': 'center'
                        }),
                        html.Span(track_data['release_date'].strftime('%B %d, %Y'), 
                                 style={'color': text_color})
                    ], style={
                        'display': 'flex',
                        'align-items': 'center',
                        'margin-bottom': '20px'
                    }),
                ]),
                
                # Popularity meter
                html.Div([
                    html.Div([
                        html.Span("Popularity", style={
                            'font-size': '0.85rem',
                            'color': secondary_text,
                            'text-transform': 'uppercase',
                            'letter-spacing': '0.5px'
                        }),
                        html.Span(f"{track_data['popularity']}", 
                                 style={
                                     'float': 'right',
                                     'color': text_color,
                                     'font-weight': '600'
                                 })
                    ], style={'margin-bottom': '6px'}),
                    dbc.Progress(
                        value=track_data['popularity'],
                        max=100,
                        color=accent_color,
                        style={
                            'height': '6px',
                            'margin-bottom': '20px',
                            'border-radius': '3px',
                            'background': 'rgba(0,0,0,0.05)'
                        }
                    )
                ]),
                
                # Audio features
                *[html.Div([
                    html.Div([
                        html.Div([
                            html.I(className=f"fas {'fa-fire' if feat == 'energy' else 'fa-smile' if feat == 'valence' else 'fa-running'}",
                                  style={
                                      'color': accent_color,
                                      'margin-right': '8px',
                                      'width': '16px'
                                  }),
                            html.Span(feat.capitalize(), style={
                                'font-size': '0.85rem',
                                'color': secondary_text,
                                'text-transform': 'uppercase',
                                'letter-spacing': '0.5px'
                            })
                        ], style={'display': 'flex', 'align-items': 'center'}),
                        html.Span(f"{track_data[feat]*100:.0f}", 
                                 style={
                                     'float': 'right',
                                     'color': text_color,
                                     'font-weight': '600'
                                 })
                    ], style={'margin-bottom': '6px'}),
                    dbc.Progress(
                        value=track_data[feat]*100,
                        max=100,
                        color=accent_color,
                        style={
                            'height': '6px',
                            'margin-bottom': '20px',
                            'border-radius': '3px',
                            'background': 'rgba(0,0,0,0.05)'
                        }
                    )
                ]) for feat in ['danceability', 'energy', 'valence']]
            ], style={'padding': '0 8px'})
        ])
    ], style={
        'background': bg_color,
        'padding': '20px',
        'border-radius': '16px',
        'box-shadow': card_shadow,
        'transition': 'all 0.3s ease',
        'border': '1px solid rgba(0,0,0,0.05)',
        ':hover': {
            'transform': 'translateY(-5px)',
            'box-shadow': '0 12px 40px rgba(0,0,0,0.15)'
        }
    })

def create_radar_chart(track_data):
    if track_data is None:
        return go.Figure()
    
    categories = ['Danceability', 'Energy', 'Happiness', 'Acousticness', 'Instrumentalness', 'Liveness']
    values = [
        track_data['danceability'],
        track_data['energy'],
        track_data['valence'],
        track_data['acousticness'],
        track_data['instrumentalness'],
        track_data['liveness']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Audio Features',
        line=dict(color=COLORS['primary']),
        fillcolor=f"rgba(29, 185, 84, 0.2)"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    
    return fig

# Create the dashboard layout
review_spotify = dbc.Container([
    # Custom CSS
    html.Link(
        href='https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
        rel='stylesheet'
    ),
    
    # Header with logo
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png",
                    style={'height': '40px', 'margin-right': '15px'}
                ),
                html.H1(
                    "Spotify Dashboard",
                    style={
                        'font-weight': '700',
                        'margin': '0',
                        'color': COLORS['primary'],
                        'display': 'inline-block',
                        'vertical-align': 'middle'
                    }
                )
            ], style={
                'display': 'flex', 
                'align-items': 'center', 
                'margin-bottom': '30px',
                'padding': '15px',
                'background': 'transparent',
                'border-radius': '8px',
                'border-bottom': f"2px solid {COLORS['primary']}"
            })
        ], width=12)
    ]),
    # Metrics Row
    dbc.Row([
        # Left Column: Featured Card
        dbc.Col(
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-medal", style={
                            'margin-right': '12px', 
                            'color': COLORS['primary'],
                            'font-size': '20px'
                        }),
                        html.Span("TOP PERFORMANCE", style={
                            'font-weight': 'bold',
                            'letter-spacing': '1px',
                            'color': COLORS['text'],
                            'font-size': '16px'
                        })
                    ], style={
                        'display': 'flex', 
                        'align-items': 'center',
                        'height': '100%'
                    })
                ], style={
                    **STYLES['header'],
                    'background': 'linear-gradient(90deg, rgba(29,185,84,0.1) 0%, rgba(29,185,84,0.05) 100%)',
                    'border-bottom': f"2px solid {COLORS['primary']}",
                    'padding': '15px 20px'
                }),
                dbc.CardBody([
                    html.H4("MUSIC INSIGHTS", style={
                        'color': COLORS['primary'],
                        'margin-bottom': '20px',
                        'font-weight': 'bold',
                        'letter-spacing': '1px',
                        'text-transform': 'uppercase',
                        'font-size': '14px'
                    }),
                    
                    dbc.Button([
                        html.I(className="fas fa-chevron-right", style={'margin-right': '8px'}),
                        "VIEW DETAILS"
                    ], 
                    id="view-details-button",
                    color="primary", 
                    size="sm",
                    className="mb-4",
                    style={
                        'border-radius': '20px',
                        'font-weight': '600',
                        'background-color': COLORS['primary'],
                        'border': 'none',
                        'box-shadow': '0 2px 8px rgba(29,185,84,0.3)',
                        'padding': '8px 16px',
                        'text-transform': 'uppercase',
                        'letter-spacing': '0.5px'
                    }
                    ),                    
                    dbc.Row([
                        # Total Tracks
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-music", style={
                                        'color': COLORS['primary'],
                                        'font-size': '24px',
                                        'margin-bottom': '10px'
                                    }),
                                    dcc.Graph(
                                        figure=create_donut_chart(total_tracks, "Total Tracks"),
                                        config={'displayModeBar': False},
                                        style={'height': '80px'}
                                    )
                                ], style={
                                    'display': 'flex',
                                    'flex-direction': 'column',
                                    'align-items': 'center',
                                    'justify-content': 'center',
                                    'height': '100%'
                                })
                            ], style={
                                'background-color': 'transparent',
                                'border-radius': '10px',
                                'padding': '15px 0'
                            })
                        ], md=3, className="mb-3"),
                        
                        # Unique Artists
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-users", style={
                                        'color': COLORS['info'],
                                        'font-size': '24px',
                                        'margin-bottom': '10px'
                                    }),
                                    dcc.Graph(
                                        figure=create_donut_chart(total_artists, "Unique Artists"),
                                        config={'displayModeBar': False},
                                        style={'height': '80px'}
                                    )
                                ], style={
                                    'display': 'flex',
                                    'flex-direction': 'column',
                                    'align-items': 'center',
                                    'justify-content': 'center',
                                    'height': '100%'
                                })
                            ], style={
                                'background-color': 'transparent',
                                'border-radius': '10px',
                                'padding': '15px 0'
                                
                            })
                        ], md=3, className="mb-3"),
                        
                        # Avg Popularity
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-fire", style={
                                        'color': COLORS['danger'],
                                        'font-size': '24px',
                                        'margin-bottom': '10px'
                                    }),
                                    dcc.Graph(
                                        figure=create_donut_chart(avg_popularity, "Avg Popularity", COLORS['info'], 100),
                                        config={'displayModeBar': False},
                                        style={'height': '80px'}
                                    )
                                ], style={
                                    'display': 'flex',
                                    'flex-direction': 'column',
                                    'align-items': 'center',
                                    'justify-content': 'center',
                                    'height': '100%'
                                })
                            ], style={
                                'background-color': 'transparent',
                                'border-radius': '10px',
                                'padding': '15px 0'
                                
                            })
                        ], md=3, className="mb-3"),
                        
                        # Avg Duration
                        dbc.Col([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-clock", style={
                                        'color': COLORS['warning'],
                                        'font-size': '24px',
                                        'margin-bottom': '10px'
                                    }),
                                    dcc.Graph(
                                        figure=create_donut_chart(avg_duration_min, "Avg Duration", COLORS['warning'], 10),
                                        config={'displayModeBar': False},
                                        style={'height': '80px'}
                                    )
                                ], style={
                                    'display': 'flex',
                                    'flex-direction': 'column',
                                    'align-items': 'center',
                                    'justify-content': 'center',
                                    'height': '100%'
                                })
                            ], style={
                                'background-color': 'transparent',
                                'border-radius': '10px',
                                'padding': '15px 0'
                    
                            })
                        ], md=3, className="mb-3")
                    ], style={'margin-top': '5px'})
                ], style={
                    'padding': '20px',
                    'height': '100%'
                })
            ], style={
                **STYLES['card'],
                'background': COLORS['card'],
                'border': 'none',
                'box-shadow': '0 4px 20px rgba(0,0,0,0.08)',
                'height': '100%',
                'border-radius': '12px'
            }),
            md=9,
            className="pr-2"
        ),

        # Right Column: Image
        dbc.Col(
            html.Div(
                html.Img(
                    src="/assets/t1.png",
                    style={
                        'width': '100%',
                        'height': '85%',
                        'object-fit': 'cover',
                        'border-radius': '12px',
                        'box-shadow': '0 4px 20px rgba(0,0,0,0.1)'
                    }
                ),
                style={
                    'height': '120%',
                    'overflow': 'hidden',
                    'border-radius': '12px'
                }
            ),
            md=3,
            className="pl-2"
        )
    ],className="mb-5", style={'height': '400px', 'margin-bottom': '20px'}),
    dbc.Row([
    dbc.Col([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-music", style={'margin-right': '10px', 'color': COLORS['primary']}),
                            "Track Analysis"
                        ], style={'display': 'flex', 'align-items': 'center'})
                    ], style=STYLES['header']),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id='playlist-filter',
                                    options=[{'label': p, 'value': p} for p in df['playlist_name'].unique()],
                                    multi=True,
                                    placeholder='Select Playlists',
                                    style={
                                        'font-family': 'Montserrat',
                                        'background-color': COLORS['background'],
                                        'color': COLORS['text'],
                                        'border': f"1px solid {COLORS['border']}"
                                    }
                                )
                            ], md=6),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='metric-selector',
                                    options=[
                                        {'label': 'Popularity', 'value': 'popularity'},
                                        {'label': 'Energy', 'value': 'energy'},
                                        {'label': 'Danceability', 'value': 'danceability'},
                                        {'label': 'Valence (Happiness)', 'value': 'valence'},
                                        {'label': 'BPM', 'value': 'BPM'},
                                        {'label': 'Duration (min)', 'value': 'track_duration_min'}
                                    ],
                                    value='popularity',
                                    clearable=False,
                                    style={
                                        'font-family': 'Montserrat',
                                        'background-color': COLORS['background'],
                                        'color': COLORS['text'],
                                        'border': f"1px solid {COLORS['border']}"
                                    }
                                )
                            ], md=6),
                        ], className='mb-3'),

                        dcc.Graph(
                            id='tracks-graph',
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    ])
                ], style=STYLES['card']),
                md=7
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-info-circle", style={'margin-right': '10px', 'color': COLORS['primary']}),
                            "Track Details"
                        ], style={'display': 'flex', 'align-items': 'center'})
                    ], style=STYLES['header']),
                    dbc.CardBody([
                        html.Div(id='selected-track-info', style={
                            'color': COLORS['text'],
                            'height': '400px',
                            'overflow-y': 'auto',
                            'padding': '10px'
                        })
                    ])
                ], style=STYLES['card']),
                md=5
            )
        ], className="mb-3"),

        # Distribution + Radar chart nằm ngang
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-wave-square", style={'margin-right': '10px', 'color': COLORS['primary']}),
                            "Audio Features Distribution"
                        ], style={'display': 'flex', 'align-items': 'center'})
                    ], style=STYLES['header']),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='feature-selector',
                            options=[
                                {'label': 'Danceability', 'value': 'danceability'},
                                {'label': 'Energy', 'value': 'energy'},
                                {'label': 'Valence (Happiness)', 'value': 'valence'},
                                {'label': 'Acousticness', 'value': 'acousticness'},
                                {'label': 'Instrumentalness', 'value': 'instrumentalness'},
                                {'label': 'Liveness', 'value': 'liveness'},
                                {'label': 'Speechiness', 'value': 'speechiness'}
                            ],
                            value='danceability',
                            clearable=False,
                            style={
                                'font-family': 'Montserrat',
                                'background-color': COLORS['background'],
                                'color': COLORS['text'],
                                'border': f"1px solid {COLORS['border']}"
                            },
                            className='mb-2'
                        ),
                        dcc.Graph(
                            id='features-distribution',
                            config={'displayModeBar': False},
                            style={'height': '240px'}
                        )
                    ])
                ], style=STYLES['card']),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.I(className="fas fa-chart-radar", style={'margin-right': '10px', 'color': COLORS['primary']}),
                            "Audio Features Radar"
                        ], style={'display': 'flex', 'align-items': 'center'})
                    ], style=STYLES['header']),
                    dbc.CardBody([
                        dcc.Graph(
                            id='audio-features-radar',
                            config={'displayModeBar': False},
                            style={'height': '290px'}
                        )
                    ])
                ], style=STYLES['card']),
                md=6
            )
        ], className="mb-3"),

        # Release Timeline (đơn lẻ)
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.I(className="fas fa-calendar-alt", style={'margin-right': '10px', 'color': COLORS['primary']}),
                    "Music Release Timeline"
                ], style={'display': 'flex', 'align-items': 'center'})
            ], style=STYLES['header']),
            dbc.CardBody([
                dcc.Graph(
                    id='release-timeline',
                    config={'displayModeBar': False},
                    style={'height': '400px'}
                )
            ])
        ], style=STYLES['card'])
    ], md=12)
]),
    dcc.Store(id='scroll-store'),
    # Bottom Row - Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-table", style={'margin-right': '10px', 'color': COLORS['primary']}),
                        "Playlist Tracks Data"
                    ], style={'display': 'flex', 'align-items': 'center'})
                ], style=STYLES['header']),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='tracks-table',
                        columns=[
                            {"name": "", "id": "spotify_cover_url", "presentation": "markdown"},
                            {"name": "Track", "id": "track_name"},
                            {"name": "Artist", "id": "artist_names"},
                            {"name": "Album", "id": "album_name"},
                            {"name": "Popularity", "id": "popularity"},
                            {"name": "Duration", "id": "track_duration_min", 'format': {'specifier': '.2f'}},
                            {"name": "Release Year", "id": "release_year"}
                        ],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={
                            'overflowX': 'auto',
                            'font-family': 'Montserrat'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'whiteSpace': 'normal',
                            'backgroundColor': COLORS['card'],
                            'color': COLORS['text'],
                            'border': f"1px solid {COLORS['border']}"
                        },
                        style_header={
                            'backgroundColor': COLORS['background'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px',
                            'color': COLORS['text'],
                            'border': f"1px solid {COLORS['border']}"
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'spotify_cover_url'},
                                'width': '60px',
                                'padding': '5px'
                            },
                            {
                                'if': {'column_id': 'popularity', 'filter_query': '{popularity} > 70'},
                                'backgroundColor': 'rgba(40, 167, 69, 0.1)',
                                'color': COLORS['text']
                            },
                            {
                                'if': {'state': 'selected'},
                                'backgroundColor': 'rgba(29, 185, 84, 0.1)',
                                'border': f"1px solid {COLORS['primary']}"
                            }
                        ],
                        markdown_options={'html': True},
                        filter_action='native',
                        sort_action='native',
                        row_selectable='single',
                        selected_rows=[]
                    )
                ])
            ], style=STYLES['card'])
        ], width=12)
    ], className="mt-4")
], fluid=True, style={
    'background-color': COLORS['background'],
    'padding': '20px',
    'min-height': '100vh',
    'font-family': 'Montserrat, sans-serif'
})
