import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
def review_stock_callbacks(app):
    @app.callback(
        Output('stock-graph', 'figure'),
        [Input('sector-filter', 'value'),
         Input('metric-selector', 'value'),
         Input('top-n-selector', 'value')]
    )
    def update_graph(selected_sectors, selected_metric, top_n):
        # Lọc theo sector nếu có
        if selected_sectors:
            filtered_df = df[df['Sector'].isin(selected_sectors)].copy()
        else:
            filtered_df = df.copy()
        
        # Sắp xếp và lấy top N theo metric
        if selected_metric == 'Change':
            top_df = filtered_df.nlargest(top_n // 2, columns=[selected_metric])
            bottom_df = filtered_df.nsmallest(top_n // 2, columns=[selected_metric])
            filtered_df = pd.concat([top_df, bottom_df]).sort_values(by=selected_metric, ascending=False)
        else:
            filtered_df = filtered_df.nlargest(top_n, columns=[selected_metric])
        
        # Xử lý màu sắc
        if selected_metric == 'Change':
            colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in filtered_df[selected_metric]]
            marker = {'color': colors}
        else:
            marker = {
                'color': filtered_df[selected_metric],
                'colorscale': [[0, '#1e87cd'], [1, '#ff99cc']],  # Gradient: xanh → hồng
                'cmin': filtered_df[selected_metric].min(),
                'cmax': filtered_df[selected_metric].max(),
                'colorbar': {'title': selected_metric}
            }

        fig = {
            'data': [
                {
                    'x': filtered_df['Name'],
                    'y': filtered_df[selected_metric],
                    'type': 'bar',
                    'text': filtered_df['Symbol'],
                    'marker': marker,
                    'hovertemplate': '<b>%{x}</b><br>' +
                                    f'{selected_metric}: %{{y}}<br>' +
                                    'Symbol: %{text}<extra></extra>'
                }
            ],
            'layout': {
                'title': {
                    'text': f'Top {top_n} {selected_metric} by Company',
                    'font': {'family': 'Montserrat', 'size': 18}
                },
                'xaxis': {
                    'title': 'Company',
                    'tickfont': {'family': 'Montserrat'},
                    'tickangle': -45
                },
                'yaxis': {
                    'title': selected_metric,
                    'tickfont': {'family': 'Montserrat'}
                },
                'hovermode': 'closest',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'family': 'Montserrat'},
                'margin': {'b': 120}
            }
        }

        return fig


    @app.callback(
        Output('selected-stock-info', 'children'),
        [Input('stock-graph', 'clickData')]
    )
    def display_click_data(clickData):
        if clickData is None:
            return html.Div("Click on a bar in the graph to see detailed stock information",
                           style={'color': "#182122", 'textAlign': 'center'})
        
        symbol = clickData['points'][0]['text']
        stock_data = df[df['Symbol'] == symbol].iloc[0]
        
        change_class = "positive-change" if stock_data['Change'] >= 0 else "negative-change"
        
        return dbc.Row([
            dbc.Col([
                html.H4(stock_data['Name'], 
                       style={'fontWeight': '600', 'marginBottom': '20px'}),
                html.P(f"Symbol: {stock_data['Symbol']}"),
                html.P(f"Exchange: {stock_data['Exchange']}"),
                html.P(f"Sector: {stock_data['Sector']}")
            ], width=3),
            
            dbc.Col([
                html.H5("Pricing", style={'fontWeight': '600', 'marginBottom': '15px'}),
                html.P(f"Close: ${stock_data['Close']:.2f}"),
                html.P(f"Change: {stock_data['Change']:.2f}%", 
                      className=change_class),
                html.P(f"Volume: {stock_data['Volume']:,}")
            ], width=3),
            
            dbc.Col([
                html.H5("Valuation", style={'fontWeight': '600', 'marginBottom': '15px'}),
                html.P(f"Market Cap: ${stock_data['Market Cap']/1e12:.2f}T"),
                html.P(f"P/E Ratio: {stock_data['P/E Ratio']:.2f}"),
                html.P(f"EPS: {stock_data['EPS']:.2f}"),
                html.P(f"Beta (1Y): {stock_data['Beta (1Y)']:.2f}")
            ], width=3),
            
            dbc.Col([
                html.H5("Performance Chart", style={'fontWeight': '600', 'marginBottom': '15px'}),
                dcc.Graph(
                    figure={
                        'data': [
                            {
                                'values': [abs(stock_data['Close']), abs(stock_data['Change']), abs(stock_data['P/E Ratio'])],
                                'labels': ['Close Price', 'Daily Change', 'P/E Ratio'],
                                'type': 'pie',
                                'hole': 0.5,
                                'marker': {'colors': ["#8fc5e9", "#b9fbd4", "#edcb94"]}
                            }
                        ],
                        'layout': {
                            'height': 200,
                            'margin': {'l': 20, 'r': 20, 't': 20, 'b': 20},
                            'showlegend': False,
                            'font': {'family': 'Montserrat'}
                        }
                    },
                    config={'displayModeBar': False}
                )
            ], width=3)
        ])
