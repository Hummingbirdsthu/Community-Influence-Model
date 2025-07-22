from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_table





tab1_layout = html.Div([
    # 1. Summary cards 
    dbc.Row([
        dbc.Col(html.Div(id='total-stock-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='total-sector-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='total-community-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='max-change-card'), md=3, className='h-100'),
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    # 2. Network + sunburst chart
    dbc.Row([
        dbc.Col([
            html.Div(
                id='network-graph',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            )
        ], md=8, className="h-100"),
        dbc.Col([
            dcc.Graph(
                id='sunburst-sector-cluster',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            ),
        ], md=4, className="h-100"),
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    dbc.Row([dbc.Col([
            html.Div(
                id='lambda-distribution-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat'}
            ),
        ], md=4, className="h-100"),
        dbc.Col([
            dash_table.DataTable(
                id='network-table',
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
        ], md=8, className="h-100")
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),
])