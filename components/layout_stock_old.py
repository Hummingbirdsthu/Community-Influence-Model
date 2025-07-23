from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
from datetime import datetime
import pytz
import base64
COLOR_SCHEME = {
    "lightblue": "#14FFEF",
    "darkblue": "#567396",
    "white": "#FFFFFF",
    "highlight": "#00FFFF",
    "background": "rgba(0,0,0,0.7)"
}
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap'])
server = app.server
server = app.server
load_figure_template('slate')
df = pd.read_csv('filtered_lambda_output.csv')
def generate_sample_network():
    G = nx.erdos_renyi_graph(n=50, p=0.1)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(1, 10)
    for node in G.nodes():
        G.nodes[node]['size'] = np.random.randint(5, 20)
        G.nodes[node]['group'] = np.random.randint(1, 5)

    return G

def analyze_communities(G):
    partition = community_louvain.best_partition(G, weight='weight')

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

    nodes = []
    for node in G.nodes():
        nodes.append({
            'Node': node,
            'Community': partition[node],
            'Degree': degree_centrality[node],
            'Betweenness': betweenness_centrality[node],
            'Closeness': closeness_centrality[node],
            'Eigenvector': eigenvector_centrality[node],
            'Connections': G.degree(node)
        })

    return pd.DataFrame(nodes), partition

# Tạo dữ liệu mẫu
sample_network = generate_sample_network()
df_nodes, community_partition = analyze_communities(sample_network)

layout_stock = dbc.Container(
    fluid=True,
    style={
        'backgroundColor': "#8197e3",
        'minHeight': '100vh',
        'margin': '0',
        'padding': '0',
        'color': COLOR_SCHEME['white'],
        'fontFamily': 'Montserrat',
        'maxWidth': '2500px',
        'marginLeft': 'auto',
        'overflow': 'hidden'  # Giữ hidden cho container chính
    },
    children=[
        # Thêm wrapper div với scroll
        html.Div(
            style={
                'height': 'calc(100vh - 20px)',
                'overflowY': 'auto',
                'padding': '0px 15px'
            },
            children=[
                dcc.Interval(id='refresh', interval=300 * 1000),
                
                # Header section
                dbc.Row(
                    dbc.Col(
                        html.H2(
                            "Network Influence Analyst Dashboard",
                            style={
                                'font-weight': 'bold',
                                'font-size': '32px',
                                'font-family': 'Montserrat',
                                'color': "#d0fdff",
                                'textShadow': '2px 2px 8px #000000',
                                'textAlign': 'center',
                                'marginTop': '20px',
                                'marginBottom': '30px'
                            }
                        ),
                        width=12
                    )
                ),
                
                # Main tabs
                dcc.Tabs(
                    id='tabs',
                    value='tab-1',
                    children=[
                        # Tab 1: Network Visualization
                        dcc.Tab(
                            label='Network Visualization',
                            value='tab-1',
                            style={
                                'backgroundColor': '#fff',
                                'color': '#191414',
                                'border': '2px solid #ddd',
                                'padding': '6px 12px',
                                'fontWeight': 'bold',
                                'fontFamily': 'Montserrat',
                                'fontSize': '14px',
                                'height': '40px',
                                'lineHeight': '24px'
                            },
                            selected_style={
                                'backgroundColor': '#517DD6',
                                'color': 'white',
                                'border': '2px solid #00f2ff',
                                'padding': '6px 12px',
                                'fontWeight': 'bold'
                            },
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label('Select Sectors', 
                                                           style={'color': 'white', 'font-size': 15, 'font-family': 'Montserrat'}),
                                                dcc.Dropdown(
                                                    id='sector-filter',
                                                    options=[{'label': sector, 'value': sector} for sector in sorted(df['Sector'].unique())],
                                                    multi=True,
                                                    placeholder='All Sectors',
                                                    style={'font-size': 15, 'font-family': 'Montserrat', 'color': 'black'}
                                                )
                                            ], width=7),
                                            dbc.Col([
                                                html.Label('Community Filter', 
                                                          style={'color': 'white', 'font-size': 15, 'font-family': 'Montserrat'}),
                                                dcc.Dropdown(
                                                    id='relationship-filter',
                                                    options=[{'label': f'Community {cluster}', 'value': cluster} 
                                                             for cluster in df['Cluster'].value_counts().index],
                                                    value=None,
                                                    placeholder='Select Community',
                                                    style={'font-size': 15, 'font-family': 'Montserrat', 'color': 'black'}
                                                )
                                            ], width=5)
                                        ]), 
                                        html.Div(
                                            id='network-graph',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'marginTop': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '400px'
                                            }
                                        )
                                    ], width=6, style={'height': '100%'}),
                                    
                                    dbc.Col(
                                        html.Div(
                                            id='pie-chart-sector',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '100%',
                                                'minHeight': '400px'
                                            }
                                        ),
                                        width=3
                                    ),
                                    dbc.Col(
                                        dbc.Card([
                                            dbc.CardHeader(
                                               html.H4(
                                                    "Highest Lambda Stocks by Cluster",
                                                    style={
                                                        'font-weight': 'bold',
                                                        'font-family': 'Montserrat',
                                                        'color': '#00CED1',
                                                        'text-align': 'center',
                                                        'font-size': '20px'
                                                    }
                                                )
                                            ),
                                            dbc.CardBody(
                                                html.Div(id='lambda-bar-chart')
                                            )
                                        ], style={
                                            'backgroundColor': 'white',
                                            'borderRadius': '15px',
                                            'border': '1px solid #00f2ff',
                                            'height': '100%',
                                            'minHeight': '400px'
                                        }),
                                        width=3
                                    ),
                                ], style={'marginBottom': '20px', 'alignItems': 'stretch'}),
                                
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Card([
                                            dbc.CardHeader(
                                                html.H4("Sector Sensitivity Analysis", 
                                                       style={
                                                        'font-weight': 'bold',
                                                        'font-family': 'Montserrat',
                                                        'color': '#00CED1',
                                                        'text-align': 'center',
                                                        'font-size': '20px'
                                                    })
                                            ),
                                            dbc.CardBody([
                                                html.Div(id='beta-sector-table'),
                                                html.P(
                                                    "* ※: Sector with stocks having negative Beta (1Y)",
                                                    style={'color': 'white', 'fontSize': '12px', 'textAlign': 'right'}
                                                )
                                            ])
                                        ], style={
                                            'backgroundColor': "#fcfdff",
                                            'borderRadius': '15px',
                                            'border': '1px solid #00f2ff',
                                            'height': '100%',
                                            'minHeight': '350px'
                                        }),
                                        width=4
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id='lambda-histogram',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '100%',
                                                'minHeight': '350px'
                                            }
                                        ),
                                        width=4
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id='beta-histogram',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '100%',
                                                'minHeight': '350px'
                                            }
                                        ),
                                        width=4
                                    )
                                ], style={'marginBottom': '20px', 'alignItems': 'stretch'}),
                                
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(
                                            id='correlation-heatmap',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '100%',
                                                'minHeight': '300px'
                                            }
                                        ),
                                        width=4
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id='sector-cluster-bar',
                                            style={
                                                'backgroundColor': "#fcfdff",
                                                'borderRadius': '15px',
                                                'padding': '15px',
                                                'border': '1px solid #00f2ff',
                                                'height': '100%',
                                                'minHeight': '300px'
                                            }
                                        ),
                                        width=4
                                    )
                                ], style={'marginBottom': '20px', 'alignItems': 'stretch'})
                            ]
                        ),
                        
                        # Tab 2: Community Analysis
                        dcc.Tab(
                            label='Community Analysis',
                            value='tab-2',
                            style={
                                'backgroundColor': '#fff',
                                'color': '#191414',
                                'border': '2px solid #ddd',
                                'padding': '6px 12px',
                                'fontWeight': 'bold',
                                'fontFamily': 'Montserrat',
                                'fontSize': '14px',
                                'height': '40px',
                                'lineHeight': '24px'
                            },
                            selected_style={
                                'backgroundColor': '#517DD6',
                                'color': 'white',
                                'border': '2px solid #00f2ff',
                                'padding': '6px 12px',
                                'fontWeight': 'bold'
                            },
                            children=[
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id='community-selector',
                                            options=[{'label': f'Cluster {i}', 'value': i} 
                                                     for i in sorted(df['Cluster'].unique())],
                                            placeholder='Select Cluster',
                                            style={
                                                'color': 'black',
                                                'borderRadius': '10px',
                                                'width': '350px',
                                                'margin': '20px auto'
                                            }
                                        ),
                                        width=12,
                                        style={'textAlign': 'center'}
                                    )
                                ),
                                
                                dcc.Loading(
                                    dbc.Row([
                                        dbc.Col(
                                            html.Div(
                                                id='community-stats',
                                                style={
                                                    'backgroundColor': "#fcfdff",
                                                    'borderRadius': '15px',
                                                    'padding': '20px',
                                                    'margin': '10px',
                                                    'border': '1px solid #00f2ff',
                                                    'height': '400px'
                                                }
                                            ),
                                            width=6
                                        ),
                                        dbc.Col(
                                            html.Div(
                                                id='influence-graph',
                                                style={
                                                    'backgroundColor': "#fcfdff",
                                                    'borderRadius': '15px',
                                                    'padding': '20px',
                                                    'margin': '10px',
                                                    'border': '1px solid #00f2ff',
                                                    'height': '400px'
                                                }
                                            ),
                                            width=6
                                        )
                                    ]),
                                    type='graph'
                                )
                            ]
                        ),
                        
                        # Tab 3: Systemic Risk Analysis
                        dcc.Tab(
                            label='Systemic Risk Analysis',
                            value='tab-3',
                            style={
                                'backgroundColor': '#fff',
                                'color': '#191414',
                                'border': '2px solid #ddd',
                                'padding': '6px 12px',
                                'fontWeight': 'bold',
                                'fontFamily': 'Montserrat',
                                'fontSize': '14px',
                                'height': '40px',
                                'lineHeight': '24px'
                            },
                            selected_style={
                                'backgroundColor': "#fcfdff",
                                'color': 'white',
                                'border': '2px solid #00f2ff',
                                'padding': '6px 12px',
                                'fontWeight': 'bold',
                                'height': '300px'
                            },
                            children=[
                                # Content for Tab 3 can be added here
                            ]
                        )
                    ],
                    style={
                        'marginBottom': '20px',
                        'justifyContent': 'center'
                    }
                )
            ]
        )
    ]
)