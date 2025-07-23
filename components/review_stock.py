import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update, callback
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
import dash_bootstrap_components as dbc
# import dash_table
import pandas as pd
from dash_bootstrap_templates import load_figure_template
from src.const import get_constants


# Sample data
df = pd.read_csv("data/us_stocks_data.csv")
df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])

# Calculate summary metrics
num_stocks, num_sectors, max_market_cap, max_market_cap_stock, max_eps, max_eps_stock = get_constants(df)


# Initialize Dash app with Montserrat font
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



# =========================================================================================
# Function to generate summary cards
def generate_stats_card (title, subtitle, value, image_path):
    return html.Div(
        dbc.Card([
            dbc.CardImg(src=image_path, top=True, style={'width': '50px','alignSelf': 'center'}),
            dbc.CardBody([
                html.P(value, className="card-value", style={'margin': '0px','fontSize': '39px','fontWeight': 'bold'}),
                html.H4(title, className="card-title", style={'margin': '0px','fontSize': '18px','fontWeight': 'bold'}),
                html.P(subtitle, className="card-subtitle", style={'margin': '0px', 'fontSize': '14px', 'color': '#333'})
            ], style={'textAlign': 'center'}),
        ], style={'width': '97%', 'height': '100%', 'margin': '10px', 'padding': '20px 10px',"backgroundColor":'#ffffff','border':'none','borderRadius':'10px'})
    )

# define color scheme
COLOR_SCHEME = {
    "lightblue": "#14FFEF",
    "darkblue": "#567396",
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
        'backgroundColor': '#8197e3',
        'border':'none'
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
        'backgroundColor': '#8197e3'
    }
}

MAX_OPTIONS_DISPLAY = 3300


# =========================================================================================
# Define the layout of the 
review_stock = dbc.Container([

    # Custom CSS
    html.Link(
        href='https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
        rel='stylesheet'
    ),
    
    # 1. Title + sector filter
    dbc.Row([
        # Title
        dbc.Col(
            html.Div([
                html.Img(
                    src="/assets/img/logo.png",
                    style={'height': '45px', 'width': '100px', 'marginRight': '15px'
                }),
                html.H1(
                    "Stock Market Dashboard",
                    style={
                        'fontFamily': 'Montserrat',
                        'color': '#2a3f8f',
                        'display': 'inline-block',
                        'margin': '0',
                        'font-weight': '700',
                        'vertical-align': 'middle'
                    }
                )
            ],
            style={
                'display': 'flex', 
                'align-items': 'center', 
                'margin-bottom': '20px',
                'margin-top': '20px',
                'padding': '15px',
                'background': 'transparent',
                'border-radius': '8px',
                'border-bottom': f"2px solid #2a3f8f"
            }),
            md=9, className="mb-2"
        ),
        # Sector filter
        dbc.Col(
            dcc.Dropdown(
                id='sector-filter',
                options=[{'label': s, 'value': s} for s in df['Sector'].unique()],
                # multi=True,
                placeholder='Select Sectors',
                style={
                    'fontFamily': 'Montserrat',
                    'font-weight': '700',
                    'padding': '4px',
                    'border': '2px solid #fff',
                    'color': 'rgb(0,0,0,1)',
                    'fontColor': '#111',
                    'border-radius': '8px',
                    'backgroundColor': "#c0d2ff",
                    'align-items': 'center', 
                }
            ),
            md=3, className="align-items-end",
            style={'alignItems': 'center', 'justifyContent': 'flex-end'}
        )
    ], align="center", className="mb-3"),


    # 2. Summary cards
    # Thêm Store để lưu giá trị summary
    dcc.Store(id='summary-values'),

    dbc.Row([
        dbc.Col([
            html.Div(id='total-stocks-card')
        ], md=3, className="h-100"),
        dbc.Col([
            html.Div(id='total-sectors-card')
        ], md=3, className="h-100"),
        dbc.Col([
            html.Div(id='max-marketcap-card')
        ], md=3, className="h-100"),
        dbc.Col([
            html.Div(id='max-eps-card')
        ], md=3, className="h-100"),
    ], className='gx-2 gy-3 mb-3 align-items-stretch'),
    # className="gx-0 gy-1"),


    # 3. Numerical distribution + sector distribution
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Dropdown(
                    id='quantitative-dropdown',
                    options=[
                        {'label': col, 'value': col}
                        for col in df.select_dtypes(include='number').columns
                    ],
                    value=df.select_dtypes(include='number').columns[0],
                    clearable=False,
                    style={'marginBottom': '10px', 'fontFamily': 'Montserrat', 'box-shadow': '0 4px 6px rgba(0,0,0,0.08)', 'color': '#000000'}
                ),
                dcc.Graph(id='bar-chart',
                          style={'borderRadius': '10px', 'border': '10px solid #fff',
                                 'backgroundColor': '#fff', 'font-family': 'Montserrat',
                                 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'})
            ])
        ], md=7, className="h-100"),
        dbc.Col([
            dcc.Graph(id='pie-chart',
                      style={'borderRadius': '10px', 'border': '10px solid #fff',
                             'backgroundColor': '#fff', 'font-family': 'Montserrat',
                             'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'})
        ], md=5, className="h-100")
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    # 4. Heatmap + scatter plot 
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap-chart',
                      style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'})
        ], md=6, className="h-100"),
        dbc.Col([
            dcc.Dropdown(
                id='scatter-pair-dropdown',
                options=[
                    {'label': 'Close price vs. Volume', 'value': 'close_volume'},
                    {'label': 'Market Cap vs. Volume', 'value': 'marketcap_volume'},
                    {'label': 'Market Cap vs. Close price', 'value': 'marketcap_close'}
                ],
                value='close_volume',
                clearable=False,
                style={'marginBottom': '10px', 'fontFamily': 'Montserrat', 'box-shadow': '0 4px 8px rgba(0,0,0,0.08)', 'color': '#000000'}
            ),
            dcc.Graph(id='scatter-chart',
                      style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'})
        ], md=6, className="h-100")
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),

    # 5. Horizontal top 5 'symbol' highest + data table: top 10 
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='top5-num-dropdown',
                options=[
                    {'label': col, 'value': col}
                    for col in df.select_dtypes(include='number').columns if col not in ['Change', 'Market Cap']
                ],
                value=df.select_dtypes(include='number').columns[0],
                clearable=False,
                style={'marginBottom': '10px', 'fontFamily': 'Montserrat', 'box-shadow': '0 4px 8px rgba(0,0,0,0.08)', 'color': '#000000'}
            ),
            dcc.Graph(id='top5-bar-chart',
                      style={'borderRadius': '10px', 'border': '10px solid #fff', 'backgroundColor': '#fff', 'box-shadow': '0 4px 20px rgba(0,0,0,0.08)'})
        ], md=3, className="h-100"),
        
        dbc.Col([
            html.H5("Top 10 Stocks by EPS", style={
                'fontFamily': 'Montserrat',
                'fontWeight': 'bold',
                'color': '#222b3a',
                'marginTop': '10px',
                'marginBottom': '10px',
                'textAlign': 'left'
            }),

            dash_table.DataTable(
                id='top10-change-table',
                style_table={
                    'overflowX': 'auto',
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
        ], md=9, className="h-100")
    ], className="gx-3 gy-3 mb-3 align-items-stretch"),





], fluid=True,  className="review-stock-container")#className="full-page-container",

