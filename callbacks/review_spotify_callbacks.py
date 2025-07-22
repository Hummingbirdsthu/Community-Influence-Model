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
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.colors

data = pd.read_csv("data/merged_spotify_data_clean.csv") 
df = pd.DataFrame(data)

total_tracks = df['track_id'].nunique()
total_artists = df['artist_id'].nunique()
avg_popularity = df['popularity'].mean()
avg_duration_min = (df['track_duration_ms'].mean() / 60000).round(2)

def review_spotify_callbacks(app):
    @app.callback(
        Output('tracks-graph', 'figure'),
        [Input('playlist-filter', 'value'),
        Input('metric-selector', 'value')]
    )


    def update_graph(selected_playlists, selected_metric):
        if selected_playlists:
            filtered_df = df[df['playlist_name'].isin(selected_playlists)]
        else:
            filtered_df = df
            
        top_tracks = filtered_df.nlargest(20, selected_metric)
        
        # Tạo gradient màu từ đậm đến nhạt dựa trên số lượng track
        colors = plotly.colors.sample_colorscale('Greens', [i/19 for i in range(20)])  # 20 màu
        
        fig = {
            'data': [
                {
                    'x': top_tracks['track_name'],
                    'y': top_tracks[selected_metric],
                    'type': 'bar',
                    'marker': {'color': colors},
                    'text': top_tracks['artist_names'],
                    'hovertemplate': '<b>%{x}</b><br>Artist: %{text}<br>' + 
                                    f'{selected_metric}: %{{y}}<extra></extra>'
                }
            ],
            'layout': {
                'title': f'Top Tracks by {selected_metric.capitalize()}',
                'xaxis': {'title': 'Track', 'tickangle': -45},
                'yaxis': {'title': selected_metric.capitalize()},
                'hovermode': 'closest',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'family': 'Montserrat'},
                'margin': {'b': 120}
            }
        }
        
        return fig

    @app.callback(
        Output('selected-track-info', 'children'),
        [Input('tracks-graph', 'clickData')]
    )
    def display_track_details(clickData):
        if clickData is None:
            return html.Div(
                "Click on a track in the graph to see details",
                style={'color': '#7f8c8d', 'textAlign': 'center', 'fontStyle': 'italic'}
            )
        
        track_name = clickData['points'][0]['x']
        track_data = df[df['track_name'] == track_name].iloc[0]
        
        # Chuyển đổi duration từ ms sang phút:giây
        duration_ms = track_data['track_duration_ms']
        duration_min = f"{int(duration_ms/60000)}:{int((duration_ms%60000)/1000):02d}"
        
        return html.Div([
            html.H4(track_data['track_name'], style={
                'fontWeight': '600',
                'marginBottom': '15px',
                'color': '#1DB954'
            }),
            
            html.P([
                html.Strong("Artist: "), 
                track_data['artist_names']
            ], style={'marginBottom': '5px'}),
            
            html.P([
                html.Strong("Album: "), 
                track_data['album_name']
            ], style={'marginBottom': '5px'}),
            
            html.P([
                html.Strong("Popularity: "), 
                f"{track_data['popularity']}/100"
            ], style={'marginBottom': '5px'}),
            
            html.P([
                html.Strong("Duration: "), 
                duration_min
            ], style={'marginBottom': '5px'}),
            
            html.P([
                html.Strong("BPM: "), 
                str(track_data['BPM'])
            ], style={'marginBottom': '5px'}),
            
            html.Hr(),
            
            html.Div([
                html.Div([
                    html.Div("Energy", style={'fontSize': '0.8em', 'color': '#7f8c8d'}),
                    html.Progress(
                         value=str(track_data['energy']*100),
                        max=100,
                        style={'width': '100%', 'height': '10px', 'backgroundColor': '#1DB954'}
                    ),
                    html.Span(f"{track_data['energy']:.2f}", style={'float': 'right', 'fontSize': '0.8em'})
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Div("Danceability", style={'fontSize': '0.8em', 'color': '#7f8c8d'}),
                    html.Progress(
                         value=str(track_data['danceability']*100),
                        max=100,
                        style={'width': '100%', 'height': '10px', 'backgroundColor': '#1DB954'}
                    ),
                    html.Span(f"{track_data['danceability']:.2f}", style={'float': 'right', 'fontSize': '0.8em'})
                ], style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Div("Happiness", style={'fontSize': '0.8em', 'color': '#7f8c8d'}),
                    html.Progress(
                         value=str(track_data['happiness']*100),
                        max=100,
                        style={'width': '100%', 'height': '10px', 'backgroundColor': '#1DB954'}
                    ),
                    html.Span(f"{track_data['happiness']:.2f}", style={'float': 'right', 'fontSize': '0.8em'})
                ], style={'marginBottom': '10px'})
            ])
        ])
