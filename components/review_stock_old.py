import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output

# Sample data
data = pd.read_csv("us_stocks_data.csv")
df = pd.DataFrame(data)

# Calculate summary metrics
total_market_cap = df['Market Cap'].sum() / 1e12  # Convert to trillion
avg_pe_ratio = df['P/E Ratio'].mean()
total_volume = df['Volume'].sum() / 1e6  
sector_counts = df['Sector'].value_counts().to_dict()
# Initialize Dash app with Montserrat font
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'
])

custom_css = {
    'external_stylesheets': [
        {
            'href': 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
            'rel': 'stylesheet'
        }
    ]
}

app.css.append_css(custom_css)

# Define global styles
STYLES = {
    'font-family': 'Montserrat, sans-serif',
    'title': {
        'font-weight': '600',
        'color': '#2c3e50'
    },
    'card': {
        'border-radius': '10px',
        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
        'padding': '10px'
    },
    'positive': {
        'color': '#27ae60'
    },
    'negative': {
        'color': '#e74c3c'
    }
}

def create_donut_chart(value, title, color, center_text=None):
    fig = go.Figure(go.Pie(
        values=[value, 100-value],
        hole=0.7,
        marker_colors=[color, '#f8f9fa'],
        textinfo='none',
        hoverinfo='none',
        rotation=90
    ))
    
    # Nếu không truyền center_text thì hiển thị giá trị số như trước
    if center_text is None:
        center_text = f"<b>{value}</b>"
    else:
        # Nếu là danh sách thì nối thành chuỗi nhiều dòng (HTML)
        if isinstance(center_text, list):
            center_text = "<br>".join(center_text)
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[
            dict(
                text=center_text,
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False,
                font_family="Montserrat",
                align="center"
            ),
            dict(
                text=title,
                x=0.5, y=1.2,
                showarrow=False,
                font_family="Montserrat",
                font_size=14,
                font_color="#000000"
            )
        ],
        height=180,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# App layout
review_stock = dbc.Container([
    # CSS for Montserrat font
    html.Link(
        href='https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap',
        rel='stylesheet'
    ),
    
    # Custom style tag
    html.Div([
        dcc.Markdown('''
            <style>
                body {
                    font-family: Montserrat, sans-serif !important;
                }
                .card {
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .card-header {
                    font-weight: 600;
                }
                .positive-change {
                    color: #27ae60;
                }
                .negative-change {
                    color: #e74c3c;
                }
                .logo-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 20px;
                }
                .logo-img {
                    height: 60px;
                    margin-right: 15px;
                }
                .title-container {
                    display: flex;
                    flex-direction: column;
                }
                .donut-card {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                }
            </style>
        ''', dangerously_allow_html=True)
    ]),
    
    # Logo and Title Row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src="https://cdn-icons-png.flaticon.com/128/12140/12140258.png",
                    className="logo-img",
                    style={'height': '40px', 'marginRight': '10px'}
                ),
                html.Div([
                    html.H1("Stock Market Dashboard", 
                            style={
                                'fontWeight': '700', 
                                'margin': '0',
                                'background': 'linear-gradient(90deg, #968efa, #faa2f8)',  # gradient xanh Spotify
                                '-webkit-background-clip': 'text',
                                '-webkit-text-fill-color': 'transparent',
                                'background-clip': 'text',
                                'text-fill-color': 'transparent',
                            })
                ])
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'gap': '10px'  # khoảng cách giữa logo và text
            })
        ], width=12)
    ], className="mb-4"),

    
    # Row 1: Summary Cards (Donut Charts)
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=round(total_market_cap, 1),
                            title="Total Market Cap ($T)",
                            color='#3498db'
                        ),
                        config={'displayModeBar': False}
                    ),
                    html.P("Sum of all companies", 
                          style={'textAlign': 'center', 'color': "#000000",'marginTop': '10px', 'marginBottom': '0px'})
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=round(avg_pe_ratio, 1),
                            title="Avg P/E Ratio",
                            color='#2ecc71'
                        ),
                        config={'displayModeBar': False}
                    ),
                    html.P("Industry average", 
                          style={'textAlign': 'center', 'color': "#000000", 'marginBottom': '0px'})
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=round(total_volume, 0),
                            title="Total Volume (M)",
                            color='#e74c3c'
                        ),
                        config={'displayModeBar': False}
                    ),
                    html.P("Shares traded", 
                          style={'textAlign': 'center', 'color': "#000000", 'marginBottom': '0px'})
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        ),
        
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_donut_chart(
                            value=len(sector_counts),
                            title="Sectors",
                            color='#f39c12'
                        ),
                        config={'displayModeBar': False}
                    ),
                    html.P(", ".join(sector_counts.keys()), 
                          style={'textAlign': 'center', 'color': "#000000", 'fontSize': '12px', 'marginTop': '-20px'})
                ])
            ], style=STYLES['card']),
            width=3, className="mb-4"
        )
    ], className="mb-4"),
    
    # Rest of your layout remains the same...
    dbc.Row([
        # Cột trái: Stock Performance
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Performance", style={'fontWeight': '600'}),
                dbc.CardBody([
                    # Filters nằm trong CardBody
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='sector-filter',
                                options=[{'label': s, 'value': s} for s in df['Sector'].unique()],
                                multi=True,
                                placeholder='Select Sectors',
                                style={'fontFamily': 'Montserrat'}
                            )
                        ], md=4),
                        dbc.Col([
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'Close Price', 'value': 'Close'},
                                    {'label': 'Volume', 'value': 'Volume'},
                                    {'label': 'Market Cap', 'value': 'Market Cap'},
                                    {'label': 'P/E Ratio', 'value': 'P/E Ratio'},
                                    {'label': 'Change %', 'value': 'Change'}
                                ],
                                value='Close',
                                clearable=False,
                                style={'fontFamily': 'Montserrat'}
                            )
                        ], md=4),
                        dbc.Col([
                            dcc.Dropdown(
                                id='top-n-selector',
                                options=[
                                    {'label': 'Top 5', 'value': 5},
                                    {'label': 'Top 10', 'value': 10},
                                    {'label': 'Top 20', 'value': 20}
                                ],
                                value=10,
                                clearable=False,
                                style={'fontFamily': 'Montserrat'}
                            )
                        ], md=4),
                    ], className='mb-3'),
                    dcc.Graph(
                        id='stock-graph',
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                ])
            ], style=STYLES['card']),
            dbc.Card([
            dbc.CardHeader("Stock Details", style={'fontWeight': '600'}),
            dbc.CardBody(
                id='selected-stock-info',
                style={'fontFamily': 'Montserrat'}
            )
        ], style=STYLES['card'])
        ], md=5),  # Bạn có thể điều chỉnh md=6 thành md=7 nếu muốn biểu đồ to hơn

        # Cột phải: Data Table
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detailed Stock Data", style={'fontWeight': '600'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='stock-table',
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict('records'),
                        page_size=10,
                        style_table={
                            'overflowX': 'auto',
                            'fontFamily': 'Montserrat'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'minWidth': '100px', 
                            'width': '100px', 
                            'maxWidth': '200px',
                            'whiteSpace': 'normal',
                            'fontFamily': 'Montserrat',
                            'padding': '10px'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.5px'
                        },
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{Change} > 0',
                                    'column_id': 'Change'
                                },
                                'color': STYLES['positive']['color']
                            },
                            {
                                'if': {
                                    'filter_query': '{Change} < 0',
                                    'column_id': 'Change'
                                },
                                'color': STYLES['negative']['color']
                            }
                        ]
                    )
                ])
            ], style=STYLES['card'])
        ], md=7)  # Hoặc md=5 nếu bạn mở rộng biểu đồ ở cột trái
    ], className="mb-4"),

], fluid=True, style={'fontFamily': 'Montserrat', 'padding': '20px'})
