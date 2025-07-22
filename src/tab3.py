from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd

df = pd.read_csv('data/filtered_lambda_output.csv')
df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])



tab3_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(
                id='trend-increasing-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            ),
        ], md=4, className="h-100"),
        dbc.Col([
            html.Div(
                id='pie-chart-sector',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            ),
        ], md=5, className="h-100"),
        dbc.Col([
            html.Div(
                id='top-stocks-table',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            ),
        ], md=3, className="h-100"),
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='stock-filter',
                options=[{'label': symbol, 'value': symbol} for symbol in df['Id'].unique()],
                multi=False,
                value='GSRT',  # Default selected stock
                placeholder='Select Stock',
                style={
                    'fontFamily': 'Montserrat',
                    'color': '#fff',
                    'backgroundColor': '#222b3a',
                    'borderRadius': '10px',
                    'marginBottom': '10px'
                }
            ),
            html.Div(
                id='candlestick-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            )
            
        ], md=8, className="h-100"),
        dbc.Col([
            html.Div(
                id='volume-by-month-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            )
        ], md=4, className="h-100"),
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    dbc.Row([
        dash_table.DataTable(
            id='top10-strategy-table',
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '400px',
                'borderRadius': '10px', 
                'backgroundColor': '#222b3a', 
                'border': '2px solid #222b3a'
            },
            style_cell={
                'fontFamily': 'Montserrat',
                'textAlign': 'center',
                'padding': '5px',
                'backgroundColor': '#fff',
                'color': '#111',
                'border': '1px solid #222b3a'
            },
            style_header={
                'backgroundColor': "#253552",
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[]  # sẽ cập nhật trong callback
        )
    ], className="gx-3 gy-3 mb-3 align-items-stretch")
])