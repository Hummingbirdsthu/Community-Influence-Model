from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import networkx as nx
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.colors
from dash.exceptions import PreventUpdate

data = pd.read_csv("merged_spotify_data_clean.csv") 
df = pd.DataFrame(data)

total_tracks = df['track_id'].nunique()
total_artists = df['artist_id'].nunique()
avg_popularity = df['popularity'].mean()
avg_duration_min = (df['track_duration_ms'].mean() / 60000).round(2)

SPOTIFY_COLORS = {
    'background': '#121212',
    'card': '#181818',
    'text': "#000000",
    'primary': '#1DB954',
    'secondary': "#000000",
    'highlight': '#1ED760'
}
COLORS = {
    'background': '#FFFFFF',
    'card': '#F8F9FA',
    'text': "#000000",
    'primary': '#1DB954',  # Keeping Spotify green as accent
    'secondary': "#000000",
    'highlight': '#1ED760',
    'border': '#E9ECEF',
    'success': '#28A745',
    'info': '#17A2B8',
    'warning': '#FFC107',
    'danger': '#DC3545'
}
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
        line=dict(color=COLORS['primary'], width=3),
        fillcolor='rgba(29, 185, 84, 0.25)',  # Màu xanh nhạt mềm hơn
        marker=dict(size=6)
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=True,
                linewidth=1,
                linecolor='lightgray',
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color=COLORS['text'], size=11)
            ),
            angularaxis=dict(
                tickfont=dict(color=COLORS['text'], size=12),
                rotation=90,
                direction="clockwise"
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=300,
        font=dict(
            family="Montserrat",
            size=12,
            color=COLORS['text']
        )
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
        # Album Cover
        html.Div([
            html.Img(
                src=track_data['spotify_cover_url'],
                style={
                    'width': '20%',
                    'border-radius': '12px',
                    'box-shadow': card_shadow,
                    'margin-bottom': '20px',
                    'transition': 'transform 0.3s ease',
                    'border': '1px solid rgba(0,0,0,0.05)'
                },
                className='hover-zoom'
            )
        ]),
        
        # Track Info
        html.Div([
            # Track Title
            html.H4(
                track_data['track_name'],
                style={
                    'color': text_color,
                    'margin-bottom': '5px',
                    'white-space': 'nowrap',
                    'overflow': 'hidden',
                    'text-overflow': 'ellipsis',
                    'font-weight': '700',
                    'font-size': '0.8rem'
                }
            ),
            
            # Artist Name
            html.P(
                track_data['artist_names'],
                style={
                    'color': accent_color,
                    'margin-bottom': '15px',
                    'font-weight': '600',
                    'font-size': '0.8rem'
                }
            ),
            
            # Track Details
            html.Div([
                # Album Info
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
                    
                    # Release Date
                    html.Div([
                        html.I(className="fas fa-calendar-alt", style={
                            'color': secondary_text,
                            'margin-right': '8px',
                            'width': '16px',
                            'text-align': 'center'
                        }),
                        html.Span(pd.to_datetime(track_data['release_date']).strftime('%B %d, %Y'),
                                 style={'color': text_color})
                    ], style={
                        'display': 'flex',
                        'align-items': 'center',
                        'margin-bottom': '20px'
                    }),
                ]),
                
                # Popularity Meter
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
                                     'font-weight': '600',
                                     'background': 'linear-gradient(to right, #1DB954, #1ED760)',
                                     '-webkit-background-clip': 'text',
                                     '-webkit-text-fill-color': 'transparent'
                                 })
                    ], style={'margin-bottom': '8px'}),
                    
                    # Progress Bar
                    html.Div(
                        style={
                            'height': '8px',
                            'margin-bottom': '20px',
                            'border-radius': '4px',
                            'background': 'rgba(0,0,0,0.05)',
                            'overflow': 'hidden',
                            'box-shadow': 'inset 0 1px 2px rgba(0,0,0,0.1)'
                        },
                        children=[
                            html.Div(
                                style={
                                    'height': '100%',
                                    'width': f"{track_data['popularity']}%",
                                    'background': 'linear-gradient(90deg, #1DB954, #1ED760)',
                                    'border-radius': '4px',
                                    'box-shadow': '0 2px 4px rgba(29, 185, 84, 0.3)',
                                    'position': 'relative',
                                    'overflow': 'hidden'
                                },
                                children=[
                                    html.Div(
                                        style={
                                            'position': 'absolute',
                                            'top': '0',
                                            'left': '0',
                                            'right': '0',
                                            'bottom': '0',
                                            'background': 'linear-gradient(90deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 50%, rgba(255,255,255,0.3) 100%)',
                                            'animation': 'shine 2s infinite'
                                        }
                                    )
                                ]
                            )
                        ]
                    )
                ]),
                
                # Audio Features
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
                                'font-size': '0.6rem',
                                'color': secondary_text,
                                'text-transform': 'uppercase',
                                'letter-spacing': '0.5px'
                            })
                        ], style={'display': 'flex', 'align-items': 'center'}),
                        html.Span(f"{track_data[feat]}", 
                                 style={
                                     'float': 'right',
                                     'color': text_color,
                                     'font-weight': '600',
                                     'background': 'linear-gradient(to right, #1DB954, #1ED760)',
                                     '-webkit-background-clip': 'text',
                                     '-webkit-text-fill-color': 'transparent'
                                 })
                    ], style={'margin-bottom': '8px'}),
                    
                    # Progress Bar for each feature
                    html.Div(
                        style={
                            'height': '8px',
                            'margin-bottom': '20px',
                            'border-radius': '4px',
                            'background': 'rgba(0,0,0,0.05)',
                            'overflow': 'hidden',
                            'box-shadow': 'inset 0 1px 2px rgba(0,0,0,0.1)'
                        },
                        children=[
                            html.Div(
                                style={
                                    'height': '100%',
                                    'width': f"{track_data[feat] * 100:.1f}%",
                                    'background': f"linear-gradient(90deg, {accent_color}, {lighten_color(accent_color, 20)})",
                                    'border-radius': '4px',
                                    'box-shadow': f'0 2px 4px {opacify_color(accent_color, 0.3)}',
                                    'position': 'relative',
                                    'overflow': 'hidden',
                                    'transition': 'width 1s ease-out'
                                },
                                children=[
                                    html.Div(
                                        style={
                                            'position': 'absolute',
                                            'top': '0',
                                            'left': '0',
                                            'right': '0',
                                            'bottom': '0',
                                            'background': 'linear-gradient(90deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 50%, rgba(255,255,255,0.3) 100%)',
                                            'animation': 'shine 2s infinite'
                                        }
                                    )
                                ]
                            )
                        ]
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


# Helper functions
def lighten_color(hex_color, percent):
    """Lighten HEX color by specified percentage"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[hex_color]
    except:
        c = hex_color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return mc.to_hex(colorsys.hls_to_rgb(c[0], min(1, c[1] + percent/100), c[2]))

def opacify_color(hex_color, opacity):
    """Add opacity to HEX color"""
    from matplotlib.colors import to_rgb
    rgb = [int(x*255) for x in to_rgb(hex_color)]
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"
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
            
        top_tracks = filtered_df.nlargest(10, selected_metric)
        
        # Tạo gradient màu từ đậm đến nhạt
        colors = plotly.colors.sample_colorscale('Greens', [i/19 for i in range(20)])
        
        fig = go.Figure(
            data=[go.Bar(
                x=top_tracks['track_name'],
                y=top_tracks[selected_metric],
                marker_color=colors,
                text=top_tracks['artist_names'],
                hovertemplate='<b>%{x}</b><br>Artist: %{text}<br>' + 
                            f'{selected_metric}: %{{y}}<extra></extra>'
            )],
            layout=go.Layout(
                title=f'Top Tracks by {selected_metric.capitalize()}',
                xaxis={'title': 'Track', 'tickangle': -45},
                yaxis={'title': selected_metric.capitalize()},
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Montserrat'},
                margin={'b': 120}
            )
        )
        
        return fig

    @app.callback(
        [Output('selected-track-info', 'children'),
         Output('audio-features-radar', 'figure')],
        [Input('tracks-table', 'selected_rows'),
         Input('tracks-graph', 'clickData')]
    )
    def display_track_details(selected_rows, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'tracks-table' and selected_rows:
            track_data = df.iloc[selected_rows[0]]
        elif trigger_id == 'tracks-graph' and clickData:
            track_name = clickData['points'][0]['x']
            track_data = df[df['track_name'] == track_name].iloc[0]
        else:
            return create_track_card(None), go.Figure()
        
        return create_track_card(track_data), create_radar_chart(track_data)

    @app.callback(
        Output('features-distribution', 'figure'),
        [Input('feature-selector', 'value'),
         Input('playlist-filter', 'value')]
    )
    def update_features_distribution(selected_feature, selected_playlists):
        if selected_playlists:
            filtered_df = df[df['playlist_name'].isin(selected_playlists)]
        else:
            filtered_df = df

        fig = px.histogram(
            filtered_df,
            x=selected_feature,
            nbins=30,
            color_discrete_sequence=[SPOTIFY_COLORS['primary']],
            marginal='box',
            opacity=0.85  # 👈 tăng độ đậm cột
        )

        fig.update_traces(
            marker_line_width=1.5,  # 👈 viền rõ cột
            marker_line_color='rgba(255,255,255,0.1)',  # 👈 viền sáng nhẹ
            selector=dict(type='histogram')
        )

        fig.update_layout(
            title=f'<b>Distribution of {selected_feature.capitalize()}</b>',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Montserrat', size=12, color=SPOTIFY_COLORS['text']),
            xaxis=dict(
                title=f'<b>{selected_feature.capitalize()}</b>',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis=dict(
                title='<b>Count</b>',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)'
            ),
            margin=dict(l=40, r=20, t=60, b=50),
            hovermode='x unified',
            bargap=0.05
        )

        return fig

    @app.callback(
        Output('release-timeline', 'figure'),
        [Input('playlist-filter', 'value')]
    )
    def update_release_timeline(selected_playlists):
        if selected_playlists:
            filtered_df = df[df['playlist_name'].isin(selected_playlists)]
        else:
            filtered_df = df

        # Tạo bins theo thập kỷ để nhóm dữ liệu tốt hơn
        filtered_df['release_decade'] = (filtered_df['release_year'] // 10) * 10
        
        # Chuẩn bị dữ liệu timeline
        timeline_df = (
            filtered_df
            .groupby(['release_year', 'genre'])
            .size()
            .reset_index(name='count')
            .sort_values('release_year')
        )

        # Tạo màu sắc tùy chỉnh cho từng thể loại
        custom_colors = {
            'Pop': '#FF6B6B',
            'Rock': '#4ECDC4',
            'Hip-Hop': '#FFA07A',
            'Electronic': '#A2D5F2',
            'R&B': '#C7CEEA',
            'Jazz': '#FFD166',
            'Classical': '#B8E0D2',
            'Country': '#D4A5A5'
        }
        
        # Tạo figure với Plotly Express
        fig = px.area(
            timeline_df,
            x='release_year',
            y='count',
            color='genre',
            line_group='genre',
            color_discrete_map=custom_colors,
            template='plotly_white',
            hover_data={
                'genre': True,
                'count': ':.0f',
                'release_year': ':.0f'
            },
            category_orders={
                "genre": filtered_df['genre'].value_counts().index.tolist()
            }
        )
        
        # Cải tiến layout
        fig.update_layout(
            title={
                'text': '<b>LỊCH SỬ PHÁT TRIỂN ÂM NHẠC</b><br><span style="font-size:0.9em">Số lượng bài hát phát hành theo năm và thể loại</span>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            font={
                'family': 'Montserrat, sans-serif',
                'color': '#333333',
                'size': 12
            },
            xaxis={
                'title': '<b>NĂM PHÁT HÀNH</b>',
                'gridcolor': 'rgba(0,0,0,0.05)',
                'tickvals': list(range(
                    int(timeline_df['release_year'].min()),
                    int(timeline_df['release_year'].max()) + 1,
                    5
                )),
                'showline': True,
                'linecolor': '#dddddd',
                'linewidth': 2,
                'mirror': True
            },
            yaxis={
                'title': '<b>SỐ LƯỢNG BÀI HÁT</b>',
                'gridcolor': 'rgba(0,0,0,0.05)',
                'showline': True,
                'linecolor': '#dddddd',
                'linewidth': 2,
                'mirror': True
            },
            hovermode='x unified',
            legend={
                'title': '<b>THỂ LOẠI</b>',
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -1.0,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 11},
                'itemwidth': 30
            },
            margin={'t': 100, 'b': 120, 'l': 80, 'r': 40},
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': "Montserrat",
                'bordercolor': '#eeeeee'
            },
            transition={'duration': 300}
        )
        
        # Cải tiến đường và điểm
        fig.update_traces(
            mode='lines+markers',
            line={'width': 2.5},
            marker={
                'size': 7,
                'opacity': 0.8,
                'line': {'width': 1, 'color': 'white'}
            },
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Năm: <b>%{x}</b><br>"
                "Số bài hát: <b>%{y:,}</b><extra></extra>"
            )
        )
        
        # Thêm annotation cho các điểm đặc biệt
        if not timeline_df.empty:
            # Tìm năm có nhiều bài hát nhất
            max_year = timeline_df.loc[timeline_df['count'].idxmax()]
            fig.add_annotation(
                x=max_year['release_year'],
                y=max_year['count'],
                text=f"Đỉnh điểm: {max_year['count']} bài hát",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                ax=0,
                ay=-40,
                bgcolor='white',
                bordercolor='#2a3f8f',
                borderwidth=1,
                font={'color': '#2a3f8f', 'size': 11}
            )
            
            # Thêm đường highlight cho năm đỉnh điểm
            fig.add_vline(
                x=max_year['release_year'],
                line_width=1,
                line_dash="dot",
                line_color="#2a3f8f",
                opacity=0.5
            )
        # Thêm thanh trượt và nút chọn phạm vi năm
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.08,
            rangeselector={
                'buttons': [
                    {'count': 5, 'label': "5 năm", 'step': "year", 'stepmode': "backward"},
                    {'count': 10, 'label': "10 năm", 'step': "year", 'stepmode': "backward"},
                    {'count': 20, 'label': "20 năm", 'step': "year", 'stepmode': "backward"},
                    {'step': "all", 'label': "Tất cả"}
                ],
                'bgcolor': 'rgba(255,255,255,0.8)',
                'activecolor': '#2a3f8f',
                'bordercolor': '#eeeeee',
                'font': {'size': 10}
            }
        )
        fig.update_xaxes(range=[1997, timeline_df['release_year'].max()])
        fig.update_layout(
    xaxis_rangeslider=dict(
        bgcolor='rgba(245,245,245,0.9)',
        borderwidth=1,
        bordercolor='rgba(220,220,220,0.8)',
        thickness=0.08
    )
)

        return fig

    @app.callback(
        Output('top-artists', 'figure'),
        [Input('playlist-filter', 'value')]
    )
    def update_top_artists(selected_playlists):
        if selected_playlists:
            filtered_df = df[df['playlist_name'].isin(selected_playlists)]
        else:
            filtered_df = df
            
        top_artists = filtered_df['artist_names'].value_counts().nlargest(10).reset_index()
        top_artists.columns = ['artist', 'count']
        
        fig = px.bar(
            top_artists,
            x='count',
            y='artist',
            orientation='h',
            color='count',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            title='Top Artists by Track Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Montserrat'},
            xaxis={'title': 'Number of Tracks'},
            yaxis={'title': 'Artist', 'autorange': 'reversed'},
            hovermode='y',
            coloraxis_showscale=False
        )
        
        return fig

    @app.callback(
        Output('tracks-table', 'data'),
        [Input('playlist-filter', 'value')]
    )
    def update_table(selected_playlists):
        if not selected_playlists:
            # Chuyển đổi URL hình ảnh thành thẻ HTML img
            df_copy = df.copy()
            df_copy['spotify_cover_url'] = df_copy['spotify_cover_url'].apply(
                lambda x: f'<img src="{x}" style="height:40px; border-radius:3px;"/>' if x else ''
            )
            return df_copy.to_dict('records')
        
        filtered_df = df[df['playlist_name'].isin(selected_playlists)]
        
        # Chuyển đổi URL hình ảnh thành thẻ HTML img
        filtered_df = filtered_df.copy()
        filtered_df['spotify_cover_url'] = filtered_df['spotify_cover_url'].apply(
            lambda x: f'<img src="{x}" style="height:40px; border-radius:3px;"/>' if x else ''
        )
        
        return filtered_df.to_dict('records')

    # Thêm callback cho chức năng cuộn trang
    @app.callback(
        Output('scroll-to-table-target', 'children'),
        [Input('view-details-button', 'n_clicks')]
    )
    def scroll_to_table(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        
        # Tạo một component dcc.Location với hash trỏ đến bảng
        return dcc.Location(id='scroll-location', pathname='', hash='tracks-table')

    # Hoặc sử dụng cách cuộn mượt hơn với JavaScript
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks) {
                const element = document.getElementById('tracks-table');
                if (element) {
                    element.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('scroll-store', 'data'),
        Input('view-details-button', 'n_clicks')
    )