from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


# define color scheme
COLOR_SCHEME = {
    "lightblue": "#14FFEF",
    "darkblue": "#222b3a",
    "white": "#FFFFFF",
    "highlight": "#00FFFF",
    "background": "rgba(0,0,0,0.7)"
}

tab_style = {
    'idle':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'backgroundColor': "#7691f2",
        'border':'none',
        'fontSize': '18px'  ,
        'margin': '0px'
    },
    'active':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'border':'none',
        'textDecoration': 'underline',
        'backgroundColor': '#7691f2',
        'fontSize': '18px' ,
        'margin': '0px'
    }
}



# Add CSS for styling (you might want to put this in an external CSS file in a real app)
styles = {
    'container': {
        'padding': '20px'
    },
    'row': {
        'margin-top': '-1px',
        'margin-bottom': '1px'
    },
    'card': {
        'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
        'transition': '0.3s',
        'border-radius': '5px',
        'padding': '15px',
        'margin-bottom': '20px'
    },
    'graph': {
        'height': '430px',
        'borderRadius': '10px', 
        'border': '10px solid #fff', 
        'padding': '15px',
        'backgroundColor': '#fff' # Adjust height as needed
    },
    'filter_container': {
        'margin-top': '20px',
        'padding': '15px',
        'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
        'border-radius': '5px',
    }
}

MAX_OPTIONS_DISPLAY = 3300


tab2_layout = html.Div([

    # 1. Summary cards 
    dbc.Row([
        dbc.Col(html.Div(id='total-stock-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='total-sector-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='total-community-card'), md=3, className='h-100'),
        dbc.Col(html.Div(id='max-change-card'), md=3, className='h-100'),
    ], className='gx-3 gy-3 mb-3'),

    # 2. Stacked bar + sankey 
    dbc.Row([
        # stacked bar chart 
        dbc.Col([
            html.Div([
                dcc.Graph(id='stacked-bar-chart', style=styles['graph']),
            ]),
        ], md=7, className='h-100'),
        # sankey diagram
        dbc.Col([
            html.Div([
                dcc.Graph(id='sankey-diagram', style=styles['graph']),
            ]),
        ], md=5, className='h-100'),
    ], className='gx-3 gy-3 mb-3 align-items-stretch'),

    # 3. Tree map + bubble chart
    dbc.Row([
        # tree map
        dbc.Col([
            html.Div([
                dcc.Graph(id='tree-map', style=styles['graph']),
            ]),
        ], md=6, className='h-100'),
        # bubble chart
        dbc.Col([
            html.Div([
                dcc.Graph(id='bubble-chart', style=styles['graph']),
            ]),
        ], md=6, className='h-100'),
    ], className='gx-3 gy-3 mb-3 align-items-stretch'),
    

])


