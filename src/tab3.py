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
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
            ),
        ], md=4, className="h-100"),
        dbc.Col([
            html.Div(
                id='pie-chart-sector',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
            ),
        ], md=5, className="h-100"),
        dbc.Col([
            html.Div(
                id='top-stocks-table',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
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
                    'color': 'rgb(0,0,0,1)',
                    'backgroundColor': "#f4f3f3",
                    'borderRadius': '10px',
                    'marginBottom': '10px',
                    'box-shadow': '0 4px 8px rgba(0,0,0,0.08)'
                }
            ),
            html.Div(
                id='candlestick-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
            )
            
        ], md=8, className="h-100"),
        dbc.Col([
            html.Div(
                id='volume-by-month-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
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
                'border': '1px solid #222b3a',
                'box-shadow': '0 4px 20px 10px rgba(0,0,0,0.08)'
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