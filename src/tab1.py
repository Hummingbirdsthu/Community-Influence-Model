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
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
            )
        ], md=8, className="h-100"),
        dbc.Col([
            dcc.Graph(
                id='sunburst-sector-cluster',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'}
            ),
        ], md=4, className="h-100"),
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    # 3. Barchart + data table
    dbc.Row(
        [dbc.Col([
            html.Div(
                id='lambda-distribution-chart',
                style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'font-family': 'Montserrat', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)', 'height': '445px'}
            ),
        ], md=4, className="h-100"),
        dbc.Col([

            html.H5("Stocks by EPS", style={
                'fontFamily': 'Montserrat',
                'fontWeight': 'bold',
                'color': '#222b3a',
                'marginTop': '10px',
                'marginBottom': '10px',
                'textAlign': 'left'
            }),

            dash_table.DataTable(
                id='network-table',
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
        ], md=8, className="h-100")
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),
])