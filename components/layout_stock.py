from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd


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
        'fontSize': '18px'  
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
        'backgroundColor': "#2a3f8f",
        'fontSize': '20px',
        'color': '#ffffff'
    }
}

MAX_OPTIONS_DISPLAY = 3300

# load file 
df = pd.read_csv('data/filtered_lambda_output.csv')
df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])

# Initialize Dash app with Montserrat font
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'])
# server = app.server
# load_figure_template('slate')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'], title='STOCK MARKET DASHBOARD')
load_figure_template('slate')

custom_css = {
    'external_stylesheets': [
        {
            'href': 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
}

# Thêm style cho body và html để chiếm hết trang giấy
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #_dash-app-content {
                height: 100%;
                margin: 0;
                padding: 0;
                background: #222b3a;
            }
            .full-page-container {
                min-height: 100vh;
                min-width: 100vw;
                height: 100vh;
                width: 100vw;
                background: #222b3a;
                padding: 0;
                margin: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''



# Layout
layout_stock = dbc.Container([
    # 1. Header
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Img(
                    src="/assets/img/logo.png",
                    style={'height': '45px', 'width': '100px', 'marginRight': '15px'
                }),
                html.H1(
                    "Stock Market Influence Analysis",
                    style={
                        'fontFamily': 'Montserrat',
                        'color': '#2a3f8f',
                        'display': 'inline-block',
                        'margin': '0',
                        'font-weight': '700',
                        'vertical-align': 'middle'
                    }
                )
            ], style={
                'display': 'flex', 
                'align-items': 'center', 
                'margin-bottom': '20px',
                'margin-top': '20px',
                'padding': '15px',
                'background': 'transparent',
                'border-radius': '8px',
                'border-bottom': f"2px solid #2a3f8f"
            }),
            md=9, className="mb-2", width=12
        ),
    ]),
    dcc.Interval(id='refresh', interval=300 * 1000),

    # 2. Tabs + Filter
    dbc.Row([
        # a. 3 Tabs bên trái
        dbc.Col(
            dcc.Tabs(
                id='graph-tabs',
                value='network_visualize',
                children=[
                    dcc.Tab(label='Network Visualization', value='network_visualize', style=tab_style['idle'], selected_style=tab_style['active']),
                    dcc.Tab(label='Community Analysis', value='community_analysis', style=tab_style['idle'], selected_style=tab_style['active']),
                    dcc.Tab(label='Porfolio Recommendation', value='stock_analysis', style=tab_style['idle'], selected_style=tab_style['active'])
                ],
                style={'marginTop': '15px', 
                       'height': '50px', 
                       'width': '900px',
                       'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'
                       } #  bỏ 'width': '900px' để nội dung không cố định trong một kích thước
            ),
            md=8,  
            style={'display': 'flex', 'justifyContent': 'flex-start', 'alignItems': 'center', 'margin': '0px'},
            className='gx-0 gy-0'
        ),
        
        # b. 2 Filter bên phải
        dbc.Col(
            dcc.Dropdown(
                id='sector-filter',
                options=[{'label': s, 'value': s} for s in sorted(df['Sector'].unique())],
                multi=False,
                placeholder='Select Sector',
                style={
                    'fontFamily': 'Montserrat',
                    'color': '#2a3f8f',
                    'backgroundColor': 'rgba(0,0,0,0.08)',
                    'borderRadius': '10px',
                    'box-shadow': '0 4px 20px rgba(0,0,0,0.02)'
                    #'display': 'block'  # Mặc định ẩn
                }
            ),
            md=2, # style={'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
        ),
        dbc.Col(
            dcc.Dropdown(
                id='community-filter',
                options=[{'label': f'Community {c+1}', 'value': c} for c in sorted(df['Cluster'].unique())],
                placeholder='Select Community',
                style={
                    'fontFamily': 'Montserrat',
                    'color': '#2a3f8f',
                    'backgroundColor': 'rgba(0,0,0,0.08)',
                    'borderRadius': '10px',
                    'box-shadow': '0 4px 20px rgba(0,0,0,0.02)'
                    #'display': 'block'  # Mặc định ẩn
                }
            ),
            md=2, style={'justifyContent': 'flex-end', 'alignItems': 'flex-end'}
        ),
        
    ], style={'justifyContent': 'flex-end', 'alignItems': 'flex-end', 'marginLeft': '0px'}, className="mb-4"),

        # 3. Summary cart
        # summary cart gồm: total stock, total sector, total community, max_change có call back filter theo 'Cluster' và 'Sector'
    # dbc.Row([
    #     dbc.Col(html.Div(id='total-stock-card'), md=3, className='h-100'),
    #     dbc.Col(html.Div(id='total-sector-card'), md=3, className='h-100'),
    #     dbc.Col(html.Div(id='total-community-card'), md=3, className='h-100'),
    #     dbc.Col(html.Div(id='max-change-card'), md=3, className='h-100'),
    # ], className="gx-0 gy-1"),

    # 4. Content theo từng tab
    html.Div(id='tab-content', className="gx-3 gy-3 mb-3 align-items-stretch"),




], fluid=True,  className="review-stock-container") #className="full-page-container"

