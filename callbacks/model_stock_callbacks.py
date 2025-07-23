from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import pickle
import pandas as pd

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

def highlight_style_stock(df):
    max_bic = df['bic_score'].max()
    styles = []
    hover_style = {
        'if': {'state': 'active'},  # ← hovered row
        'backgroundColor': '#85c1e9',  # soft green or whatever matches your theme
        'color': 'black'  # readable text
    }
    styles.append(hover_style)

    for i, row in df.iterrows():
        style = {
            "if": {"row_index": i},
            "backgroundColor": 'rgb(255,255,255,0.07)',
            "color": '#222b3a'
        }

        if row['bic_score'] == max_bic:
            style.update({
                "backgroundColor": '#3B6E8D',
                "color": 'white',
                "fontWeight": 'bold'
            })

        styles.append(style)

    return styles

def cell_style_stock(header=False, row_header=False):
    base = {
        'border': '1px solid #ccc',
        'padding': '0.6em',
        'textAlign': 'center'
    }
    if header:
        base.update({
            'fontWeight': 'bold',
            'backgroundColor': '#F1F1F1'
        })
    if row_header:
        base.update({
            'fontWeight': 'bold',
            'backgroundColor': '#222b3a',
            'textAlign': 'left'
        })
    return base

# Load mô hình đã huấn luyện
with open('cim_model_stock.pkl', 'rb') as f:
    saved_model = pickle.load(f)

# Load the CSV with all CIM params combinations
grid_search_results = pd.read_csv("grid_search_results_stock.csv")

def model_stock_callbacks(app):
    # Filter results based on user input ranges
    @app.callback(
        [Output('combo-display_stock', 'children'),
        Output('res-table-name_stock', 'children'),
        Output('results-table-container_stock', 'children'),
        Output('filtered-grid-results_stock', 'data'),
        Output('theta-dropdown_stock', 'options'),
        Output('theta-dropdown_stock', 'value')],
        Input('grid-search-button_stock', 'n_clicks'),
        [State('theta-from_stock', 'value'), State('theta-to_stock', 'value'),
        State('gamma-from_stock', 'value'), State('gamma-to_stock', 'value')],
    )
    def filter_results(n_clicks, theta1, theta2, gamma1, gamma2):
        if n_clicks == 0:
            return "", "", "", None, [], None

        df = grid_search_results.copy()

        df_filtered = df[
            (df['theta'].between(theta1, theta2)) &
            (df['gamma'].between(gamma1, gamma2))
        ]

        if df_filtered.empty:
            return "No matching combinations found.", "", "", None, [], None

        # combo_msg = f"Grid Search combinations: {len(df_filtered)}"
        combo_msg = html.Span( 
            f"Grid Search combinations: {len(df_filtered)}",
            style={'color': '#222b3a', 'fontWeight': 'bold'})
        title_msg = html.Span( 
            f"Top Models by BIC Score",
            style={'color': '#222b3a', 'fontWeight': 'bold'})

        results_table = dash_table.DataTable(
            id='results-table_stock',
            columns=[{"name": col, "id": col} for col in ['theta','gamma','k_hat','bic_score','silhouette_score']],
            data=df_filtered.to_dict("records"),
            row_selectable='single',
            selected_rows=[],
            style_cell={'textAlign': 'center', 'fontFamily': 'Montserrat'},
            style_table={'overflowY': 'auto', 'maxHeight': '434px', 'width': '910px', 'overflowX': 'hidden'},
            style_header={
                'backgroundColor': '#154360',
                'color': 'white',
                'fontWeight': 'bold',
                'fontFamily': 'Montserrat'
            },
            style_data_conditional=highlight_style_stock(df_filtered)
        )

        theta_options = [{'label': str(t), 'value': t} for t in sorted(df_filtered['theta'].unique())]
        theta_value = df_filtered['theta'].unique()[0]

        return combo_msg, title_msg, results_table, df_filtered.to_dict('records'), theta_options, theta_value
    
    # Callback
    @app.callback(
        Output('grid-chart-title-container_stock', 'style'),
        Output('bic-combined-chart_stock', 'figure'),
        Output('combine-chart-container_stock', 'style'),
        Output('silhouette-chart_stock', 'figure'),
        Output('silhouette-chart-container_stock', 'style'),
        Output('tunning-title_stock', 'style'),
        Input('grid-search-button_stock', 'n_clicks'),
        Input('theta-dropdown_stock', 'value'),
        State('filtered-grid-results_stock', 'data')
    )
    def update_chart(n_clicks, selected_theta, stored_data):
        if n_clicks == 0 or not stored_data:
            return {'display': 'none'}, go.Figure(), {'display': 'none'}, go.Figure(), {'display': 'none'}, {
                    'marginBottom': '0.7em',
                    'padding': '0em',
                    'fontFamily': "Montserrat",
                    'fontSize': '26px'
                }

        df = pd.DataFrame(stored_data)

        # Filter by selected theta and fixed tau = 3
        df_filtered = df[
            (df['theta'] == selected_theta)
        ].sort_values(by='gamma')

        gamma_values = df_filtered['gamma'].astype(str)  # Ensure gamma is treated as categorical

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for BIC score
        fig.add_trace(
            go.Bar(
                x=gamma_values,
                y=df_filtered['bic_score'],
                name='BIC Score',
                marker=dict(color='#3B6E8D'),
                width=0.4  # Fixed width for neat spacing
            ),
            secondary_y=False
        )

        # Line chart for k_hat
        fig.add_trace(
            go.Scatter(
                x=gamma_values,
                y=df_filtered['k_hat'],
                mode='lines+markers',
                name='K̂ (Estimated Communities)',
                line=dict(color='#222b3a', width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=f"BIC Score & K̂ vs γ (θ = {selected_theta}, τ = 3)",
            xaxis_title="γ (gamma)",
            yaxis_title="BIC Score",
            yaxis2_title="K̂ (Estimated Communities)",
            bargap=0.2,  # Space between bars
            template="plotly_white",
            height=400,
            width=822,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Heatmap
        # Pivot for proper grid
        heatmap_data = df.pivot(index='theta', columns='gamma', values='silhouette_score')
        heatmap_data = heatmap_data.sort_index()  # Optional: sort theta
        heatmap_data.columns = [str(g) for g in heatmap_data.columns]  # Treat gamma as string for labeling

        fig2 = px.imshow(
            heatmap_data,
            labels=dict(x='γ (gamma)', y='θ (theta)', color='Silhouette Score'),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=[
                [0.0, "#85c1e9"],   # low score → light green
                [0.25, "#5499c7"],
                [0.5, "#2980b9"],
                [0.75, "#2471a3"],
                [1.0, "#154360"]    # high score → dark green
            ],
            text_auto=True,
            aspect="auto",  # Let Plotly adjust shape nicely
            title='Heatmap: Gamma vs Theta → Silhouette Score'
        )

        fig2.update_layout(
            template='plotly_white',
            font=dict(size=14),
            margin=dict(t=60, r=20, l=20, b=40),
            xaxis=dict(tickangle=0),
            yaxis=dict(type='category'),
            height=438,
            width=600
        )

        tunning_title = {
            'marginBottom': '0.7em',
            'backgroundColor': '#3B6E8D',
            'color': SPOTIFY_COLORS['white'],
            'fontFamily': "Montserrat",
            'fontSize': '26px',
            'border': '1px solid #888',
            'borderRadius': '6px',
            'width': '820px',
            'padding': '0.3em',
            "backdrop-filter": "blur(20px)",
            "-webkit-backdrop-filter": "blur(50px)"
        }

        return {}, fig, {'marginBottom': '2em'}, fig2, {}, tunning_title
    
    @app.callback(
        Output('statistic-table_stock', 'children'),
        Output('statistic-table_stock', 'style'),
        Input('results-table_stock', 'selected_rows'),
        State('results-table_stock', 'data')
    )
    def show_row_plot(selected_rows, stored_data):
        if not selected_rows or not stored_data:
            return None, {'display': 'none'}

        data = {
            'λ': ['(0.476, 0.690)', '(0.230, 0.693)', '(0.038, 0.319)'],
            'σ²': ['0.247', '0.070', '< 10⁻³'],
            'Intercept': ['-1.005', '0.485', '0.038'],
            'energy': ['0.046', '0.021', '0.031'],
            # 'danceability': ['0.001', '0.032', '0.965'],
            # 'happiness': ['0.034', '0.038', '0.370'],
            # 'acousticness': ['-0.002', '0.038', '0.966']
        }

        df = pd.DataFrame(data, index=['est', 's.e.', 'p-value']).reset_index().rename(columns={'index': ''})

        html_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th(col, style={
                        'border': '1px solid #ccc',
                        'padding': '0.6em',
                        'backgroundColor': '#3B6E8D',
                        'color': '#ffffff',
                        'fontWeight': 'bold',
                        'fontFamily': 'Montserrat',
                        'fontSize': '15px',
                        'textAlign': 'center'
                    }) for col in df.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(cell, style={
                        'border': '1px solid #ccc',
                        'padding': '0.6em',
                        'fontFamily': 'Montserrat',
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'color': '#222b3a'
                    }) for cell in row.values
                ]) for _, row in df.iterrows()
            ])
        ], style={
            'borderCollapse': 'collapse',
            'width': '100%',
            'marginTop': '1em'
        })

        statistic_style={
            'width': '96%',      # or '600px', '80%', etc.
            'maxHeight': '434px', # for vertical scrolling
            'overflowY': 'auto',  # enables scrolling if table overflows
            'border': '0px solid #ccc',
            'overflowX': 'hidden', # prevent scroll
            'padding': '0em'
        }

        return html_table, statistic_style