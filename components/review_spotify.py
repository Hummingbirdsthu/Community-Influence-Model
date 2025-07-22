import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.colors
from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State


data = pd.read_csv("data/merged_spotify_data_clean.csv") 
df = pd.DataFrame(data)

total_tracks = df['track_id'].nunique()
total_artists = df['artist_id'].nunique()
avg_popularity = df['popularity'].mean()
avg_duration_min = (df['track_duration_ms'].mean() / 60000).round(2)

# Khởi tạo Dash app với font Montserrat
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'
])

# Custom CSS
custom_css = {
    'external_stylesheets': [
        {
            'href': 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
}

app.css.append_css(custom_css)

# Định nghĩa styles
STYLES = {
    'font-family': 'Montserrat, sans-serif',
    'title': {
        'font-weight': '600',
        'color': '#2c3e50'
    },
    'card': {
        'border-radius': '10px',
        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
        'padding': '10px'
    },
    'positive': {
        'color': '#1DB954'  # Màu xanh Spotify
    },
    'negative': {
        'color': '#e74c3c'
    }
}

def create_donut_chart(value, title, color='#1DB954', total=1000):
    # Giới hạn value không vượt quá total
    value = min(value, total)
    
    fig = go.Figure(go.Pie(
        values=[value, total - value],
        hole=0.7,
        marker_colors=[color, '#f8f9fa'],
        textinfo='none',
        hoverinfo='none',
        rotation=90
    ))
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[
            dict(
                text=f"<b>{value}</b>",
                x=0.5, y=0.5,
                font_size=24,
                showarrow=False,
                font_family="Montserrat"
            ),
            dict(
                text=title,
                x=0.5, y=1.2,
                showarrow=False,
                font_family="Montserrat",
                font_size=14,
                font_color="#010202"
            )
        ],
        height=180,
        width=180,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# App layout
review_spotify = dbc.Container([
    # CSS for Montserrat font
    html.Link(
        href='https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
        rel='stylesheet'
    ),
    
    # Custom style tag
    html.Div([
        dcc.Markdown('''
            <style>
                body {
                    font-family: Montserrat, sans-serif !important;
                }
                .card {
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .card-header {
                    font-weight: 600;
                }
                .spotify-green {
                    color: #1DB954;
                }
                .logo-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 20px;
                }
                .logo-img {
                    height: 60px;
                    margin-right: 15px;
                }
                .title-container {
                    display: flex;
                    flex-direction: column;
                }
            </style>
        ''', dangerously_allow_html=True)
    ]),
    
    # Logo and Title Row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/1024px-Spotify_logo_without_text.svg.png",
                    style={'height': '30px', 'marginRight': '15px'}
                ),
                html.H1(
                    "Spotify Data",
                    style={
                        'fontWeight': '700',
                        'margin': '0',
                        'color': '#1DB954',
                        'textShadow': '2px 2px 0 white, -2px 2px 0 white, 2px -2px 0 white, -2px -2px 0 white'
                    },
                    className="spotify-title"
                ),
            ], style={'display': 'flex', 'alignItems': 'center'})  # flex container nằm ngang, căn giữa dọc
        ], width=12)
    ], className="mb-4"),

    # Row 1: Summary Cards (Donut Charts)
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=total_tracks,
                            title="Total Tracks",
                            color='#1DB954'
                        ),
                        config={'displayModeBar': False}
                    )
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=total_artists,
                            title="Unique Artists",
                            color='#1ED760'
                        ),
                        config={'displayModeBar': False}
                    )
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=avg_popularity.round(1),
                            title="Avg Popularity",
                            color='#1DB954'
                        ),
                        config={'displayModeBar': False}
                    )
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=avg_duration_min,
                            title="Avg Duration (min)",
                            color='#1ED760'
                        ),
                        config={'displayModeBar': False}
                    )
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        )
    ], className="mb-4"),
    
    # Row 2: Main Content
    dbc.Row([
        # Left Column: Filters and Track Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Track Analysis", style={'fontWeight': '600'}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='playlist-filter',
                                options=[{'label': p, 'value': p} for p in df['playlist_name'].unique()],
                                multi=True,
                                placeholder='Select Playlists',
                                style={'fontFamily': 'Montserrat'}
                            )
                        ], md=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'Popularity', 'value': 'popularity'},
                                    {'label': 'Energy', 'value': 'energy'},
                                    {'label': 'Danceability', 'value': 'danceability'},
                                    {'label': 'Happiness', 'value': 'happiness'},
                                    {'label': 'BPM', 'value': 'BPM'}
                                ],
                                value='popularity',
                                clearable=False,
                                style={'fontFamily': 'Montserrat'}
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
            
            # Track Details Card
            dbc.Card([
                dbc.CardHeader("Track Details", style={'fontWeight': '600'}),
                dbc.CardBody(
                    id='selected-track-info',
                    style={'fontFamily': 'Montserrat'}
                )
            ], style=STYLES['card'])
        ], md=6),
        
        # Right Column: Data Table
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Playlist Tracks Data", style={'fontWeight': '600'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='tracks-table',
                        columns=[
                            {"name": "Track", "id": "track_name"},
                            {"name": "Artist", "id": "artist_names"},
                            {"name": "Album", "id": "album_name"},
                            {"name": "Popularity", "id": "popularity"},
                            {"name": "Duration", "id": "track_duration_ms"},
                            {"name": "BPM", "id": "BPM"}
                        ],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={
                            'overflowX': 'auto',
                            'fontFamily': 'Montserrat'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'whiteSpace': 'normal'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'popularity', 'filter_query': '{popularity} > 70'},
                                'backgroundColor': '#d2f0d9',
                                'color': '#1a5331'
                            }
                        ]
                    )
                ])
            ], style=STYLES['card'])
        ], md=6)
    ], className="mb-4")
], fluid=True, style={'fontFamily': 'Montserrat', 'padding': '20px'})

