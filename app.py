from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from components.layout_stock import layout_stock
from components.layout_spotify import layout_spotify
from components.review_stock import review_stock
from components.review_spotify import review_spotify
from callbacks.register_stock_callbacks import register_stock_callbacks
from callbacks.review_stock_callbacks import review_stock_callbacks
from callbacks.review_spotify_callbacks import review_spotify_callbacks
from callbacks.register_spotify_callbacks import register_spotify_callbacks
from scipy import stats
import base64

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ],
    suppress_callback_exceptions=True
)
# Thêm CSS tùy chỉnh để cố định biểu đồ
app.css.append_css({
    'external_url': '/assets/custom.css'
})
app.css.append_css({
    'external_url': '/assets/style.css'
})

sidebar = html.Div(
    [
        html.Div(
            [
                html.Div([
                    html.Button(
                        html.I(className="fas fa-chevron-left"), 
                        id="sidebar-toggle",
                        style={
                            "background": "none",
                            "border": "none",
                            "color": "#2a3f8f",
                            "cursor": "pointer",
                            "fontSize": "1.2rem",
                            "marginRight": "10px",
                            "transition": "transform 0.3s ease"
                        }
                    ),
                    html.H2("Menu", id="menu-title", style={
                        "color": "#2a3f8f",
                        "fontFamily": "Montserrat, sans-serif",
                        "fontWeight": "600",
                        "marginBottom": "20px",
                        "display": "inline-block",
                        "transition": "opacity 0.3s ease"
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
                
                html.Hr(style={"borderTop": "1px solid rgba(42,63,143,0.3)"}),
                dbc.Nav(
                    [
                        # Menu Home
                        dbc.NavLink(
                            [html.I(className="fas fa-home me-2"), html.Span("Home", className="nav-link-text")],
                            href="/",
                            active="exact",
                            id="home-link",
                            style={
                                "color": "#2a3f8f",
                                "fontFamily": "Montserrat, sans-serif",
                                "padding": "10px 15px",
                                "margin": "5px 0",
                                "borderRadius": "5px",
                                "transition": "all 0.3s ease",
                                "whiteSpace": "nowrap"
                            }
                        ),
                        
                        # Menu Stock với submenu
                        html.Div([
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-chart-line me-2"), 
                                    html.Span("Stock", className="nav-link-text"),
                                    html.I(className="fas fa-chevron-down ml-auto", id="stock-arrow")
                                ],
                                href="#",
                                id="stock-collapse-link",
                                style={
                                    "color": "#2a3f8f",
                                    "fontFamily": "Montserrat, sans-serif",
                                    "padding": "10px 15px",
                                    "margin": "5px 0",
                                    "borderRadius": "5px",
                                    "transition": "all 0.3s ease",
                                    "whiteSpace": "nowrap",
                                    "display": "flex",
                                    "alignItems": "center"
                                }
                            ),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavLink(
                                            [html.I(className="fas fa-database me-2"), html.Span("Data", className="nav-link-text")],
                                            href="/stock/data",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                        dbc.NavLink(
                                            [html.I(className="fas fa-tachometer-alt me-2"), html.Span("Dashboard", className="nav-link-text")],
                                            href="/stock/dashboard",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                        dbc.NavLink(
                                            [html.I(className="fas fa-chart-bar me-2"), html.Span("Model Performance", className="nav-link-text")],
                                            href="/stock/performance",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                    ],
                                    vertical=True,
                                ),
                                id="stock-collapse",
                                is_open=False,
                            )
                        ], className="submenu-container"),
                        
                        # Menu Spotify với submenu
                        html.Div([
                            dbc.NavLink(
                                [
                                    html.I(className="fas fa-music me-2"), 
                                    html.Span("Spotify", className="nav-link-text"),
                                    html.I(className="fas fa-chevron-down ml-auto", id="spotify-arrow")
                                ],
                                href="#",
                                id="spotify-collapse-link",
                                style={
                                    "color": "#2a3f8f",
                                    "fontFamily": "Montserrat, sans-serif",
                                    "padding": "10px 15px",
                                    "margin": "5px 0",
                                    "borderRadius": "5px",
                                    "transition": "all 0.3s ease",
                                    "whiteSpace": "nowrap",
                                    "display": "flex",
                                    "alignItems": "center"
                                }
                            ),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavLink(
                                            [html.I(className="fas fa-database me-2"), html.Span("Data", className="nav-link-text")],
                                            href="/spotify/data",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                        dbc.NavLink(
                                            [html.I(className="fas fa-tachometer-alt me-2"), html.Span("Dashboard", className="nav-link-text")],
                                            href="/spotify/dashboard",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                        dbc.NavLink(
                                            [html.I(className="fas fa-chart-bar me-2"), html.Span("Model Performance", className="nav-link-text")],
                                            href="/spotify/performance",
                                            active="exact",
                                            style={
                                                "color": "#2a3f8f",
                                                "fontFamily": "Montserrat, sans-serif",
                                                "padding": "8px 15px 8px 30px",
                                                "margin": "2px 0",
                                                "borderRadius": "5px",
                                                "transition": "all 0.3s ease",
                                                "whiteSpace": "nowrap"
                                            }
                                        ),
                                    ],
                                    vertical=True,
                                ),
                                id="spotify-collapse",
                                is_open=False,
                            )
                        ], className="submenu-container"),
                    ],
                    vertical=True,
                    pills=True,
                    id="nav-links"
                ),
            ],
            style={"padding": "20px 15px"}
        )
    ],
    id="sidebar",
    style={
        "background": "linear-gradient(135deg, #708be0 0%, #fef1fb 100%)",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "250px",
        "fontFamily": "Montserrat, sans-serif",
        "boxShadow": "2px 0 15px rgba(0,0,0,0.1)",
        "zIndex": 1000,
        "transition": "all 0.3s ease",
        "overflow": "hidden",
    }
)

# Tạo header với account và thông báo
header = html.Div(
    [
        html.Div(
            [
                # Notification dropdown
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("New stock data available", header=True),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("System update scheduled"),
                        dbc.DropdownMenuItem("New messages (3)"),
                    ],
                    label=html.I(className="fas fa-bell"),
                    align_end=True,
                    in_navbar=True,
                    className="notification-dropdown",
                    toggle_style={
                        "color": "#2a3f8f",
                        "background": "none",
                        "border": "none",
                        "fontSize": "1rem",
                        "cursor": "pointer"
                    },
                    style={"marginRight": "15px"}
                ),
                
                # Account dropdown
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("Profile", href="#"),
                        dbc.DropdownMenuItem("Settings", href="#"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Log Out", href="#"),
                    ],
                    label=[
                        html.I(className="fas fa-user-circle", style={"marginRight": "8px"}),
                        "Admin User"
                    ],
                    align_end=True,
                    in_navbar=True,
                    className="account-dropdown",
                    toggle_style={
                        "color": "#2a3f8f",
                        "background": "none",
                        "border": "none",
                        "fontSize": "1rem",
                        "cursor": "pointer",
                        "fontFamily": "Montserrat, sans-serif"
                    }
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginLeft": "auto",
                "paddingRight": "20px"
            }
        )
    ],
style={
    "position": "fixed",
    "top": 0,
    "right": 0,
    "height": "50px",
    "backgroundColor": "rgba(255, 255, 255, 0.7)",  # 👈 trắng trong suốt
    "backdropFilter": "blur(8px)",                 # 👈 hiệu ứng nền mờ
    "WebkitBackdropFilter": "blur(8px)",           # 👈 cho Safari
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "flex-end",
    "zIndex": 999,
    "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
    "width": "calc(100% - 250px)",
    "transition": "all 0.3s ease"
},
    id="header"
)

# Layout chính
app.layout = html.Div(
    [
        dcc.Location(id='url'),
        dcc.Store(id='sidebar-state', data={'is_open': True}),
        sidebar,
        header,
        html.Div(
            id='page-content',
            style={
                "marginLeft": "250px",
                "marginTop": "20px",
                "padding": "2rem",
                "minHeight": "calc(100vh - 60px)",
                "backgroundImage": 'url("/assets/sl_122221_47450_06.jpg")',
                "backgroundSize": "cover",
                "backgroundPosition": "center",
                "backgroundRepeat": "no-repeat",
                "backgroundAttachment": "fixed",
                "color": "white",
                "fontFamily": "Montserrat, sans-serif",
                "transition": "all 0.3s ease"
            }
        ),
        # Thêm loading spinner toàn màn hình
       dcc.Loading(
            id="loading-spinner",
            type="circle",
            fullscreen=True,
            color="#2a3f8f",
            style={
                "background": "rgba(255, 255, 255, 0.7)",
                "backdropFilter": "blur(5px)" # Hiệu ứng blur hiện đại
            },
            children=[
                html.Div(
                    [
                        html.Div(className="loading-spinner"),
                        html.P("Đang tải...", style={"marginTop": "20px"})
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "height": "100%"
                    }
                )
            ]
        )
    ],
    style={"fontFamily": "Montserrat, sans-serif"},
    id="main-container"
)

# Callback để xử lý đóng/mở sidebar
app.clientside_callback(
    """
    function(n_clicks, current_state) {
        if (n_clicks === null) {
            return window.dash_clientside.no_update;
        }

        const is_open = !current_state.is_open;
        const sidebar = document.getElementById('sidebar');
        const content = document.getElementById('page-content');
        const header = document.getElementById('header');
        const toggleBtn = document.getElementById('sidebar-toggle');
        const menuTitle = document.getElementById('menu-title');
        const navTexts = document.querySelectorAll('.nav-link-text');
        const submenuArrows = document.querySelectorAll('.fa-chevron-down, .fa-chevron-up');

        if (is_open) {
            sidebar.style.width = "250px";
            sidebar.classList.remove('sidebar-collapsed'); // Bỏ lớp khi mở
            content.style.marginLeft = "250px";
            header.style.width = "calc(100% - 250px)";
            menuTitle.style.opacity = "1";
            navTexts.forEach(el => el.style.display = "inline");
            submenuArrows.forEach(el => el.style.display = "inline");
            toggleBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
        } else {
            sidebar.style.width = "60px";
            sidebar.classList.add('sidebar-collapsed'); // Thêm lớp khi thu gọn
            content.style.marginLeft = "60px";
            header.style.width = "calc(100% - 60px)";
            menuTitle.style.opacity = "0";
            navTexts.forEach(el => el.style.display = "none");
            submenuArrows.forEach(el => el.style.display = "none");
            toggleBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
        }

        return { is_open: is_open };
    },
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 300);
    """,
    Output("sidebar-state", "data"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-state", "data")
);

app.clientside_callback(
    """
    function(pathname) {
        const stockCollapse = document.getElementById('stock-collapse');
        const stockArrow = document.getElementById('stock-arrow');
        const spotifyCollapse = document.getElementById('spotify-collapse');
        const spotifyArrow = document.getElementById('spotify-arrow');

        if (!pathname) return window.dash_clientside.no_update;

        if (pathname.startsWith("/stock")) {
            if (stockCollapse && !stockCollapse.classList.contains("show")) {
                stockCollapse.classList.add("show");
                if (stockArrow) {
                    stockArrow.classList.remove("fa-chevron-down");
                    stockArrow.classList.add("fa-chevron-up");
                }
            }
        } else {
            if (stockCollapse && stockCollapse.classList.contains("show")) {
                stockCollapse.classList.remove("show");
                if (stockArrow) {
                    stockArrow.classList.add("fa-chevron-down");
                    stockArrow.classList.remove("fa-chevron-up");
                }
            }
        }

        if (pathname.startsWith("/spotify")) {
            if (spotifyCollapse && !spotifyCollapse.classList.contains("show")) {
                spotifyCollapse.classList.add("show");
                if (spotifyArrow) {
                    spotifyArrow.classList.remove("fa-chevron-down");
                    spotifyArrow.classList.add("fa-chevron-up");
                }
            }
        } else {
            if (spotifyCollapse && spotifyCollapse.classList.contains("show")) {
                spotifyCollapse.classList.remove("show");
                if (spotifyArrow) {
                    spotifyArrow.classList.add("fa-chevron-down");
                    spotifyArrow.classList.remove("fa-chevron-up");
                }
            }
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("nav-links", "children"),  # dummy output
    Input("url", "pathname")
)

# Callback để xử lý mở/đóng submenu Stock
@app.callback(
    [
        Output("stock-collapse", "is_open"),
        Output("stock-arrow", "className")
    ],
    [Input("stock-collapse-link", "n_clicks")],
    [State("stock-collapse", "is_open")],
)
def toggle_stock_collapse(n, is_open):
    if n:
        arrow_class = "fas fa-chevron-up ml-auto" if not is_open else "fas fa-chevron-down ml-auto"
        return not is_open, arrow_class
    return is_open, "fas fa-chevron-down ml-auto"

# Callback để xử lý mở/đóng submenu Spotify
@app.callback(
    [
        Output("spotify-collapse", "is_open"),
        Output("spotify-arrow", "className")
    ],
    [Input("spotify-collapse-link", "n_clicks")],
    [State("spotify-collapse", "is_open")]
)
def toggle_spotify_collapse(n, is_open):
    if n:
        new_is_open = not is_open
        arrow_class = "fas fa-chevron-up ml-auto" if new_is_open else "fas fa-chevron-down ml-auto"
        return new_is_open, arrow_class
    return is_open, "fas fa-chevron-down ml-auto"

# Callback để xử lý active menu
app.clientside_callback(
    """
    function(pathname) {
        const links = document.querySelectorAll('.nav-link');
        links.forEach(link => {
            if (link.getAttribute('href') === pathname) {
                link.style.backgroundColor = 'rgba(112, 139, 224, 0.3)';
                link.style.fontWeight = '500';
                link.style.color = '#2a3f8f';
            } else {
                link.style.backgroundColor = 'transparent';
                link.style.fontWeight = '400';
                link.style.color = '#2a3f8f';
            }
        });
        return null;
    }
    """,
    Output('null-output', 'children'),
    Input('url', 'pathname')
)

# Callback để gọi dash 
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return html.Div([
            html.Div(
                id="typing-header",
                style={
                    'fontFamily': 'Montserrat',
                    'textAlign': 'center',
                    'color': '#2a3f8f',
                    'fontWeight': 'bold',
                    'fontSize': '48px',
                    'textShadow': '2px 2px 4px rgba(0, 0, 0, 0.4)',
                    'height': '60px',
                    'marginBottom': '20px'
                }
            ),
            
            html.P(
                id="fade-subtitle",
                style={
                    'fontFamily': 'Montserrat',
                    'textAlign': 'center',
                    'color': "#2a3e8fb6",
                    'opacity': 0,
                    'transition': 'opacity 1.5s ease-in',
                    'marginBottom': '40px'
                }
            ),
            
            html.Div(
                id="slide-up-thesis",
                style={
                    'fontFamily': 'Montserrat',
                    'textAlign': 'center',
                    'color': 'white',
                    'fontSize': '24px',
                    'minHeight': '100px',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'backgroundColor': 'rgba(42, 63, 143, 0.7)',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'border': '2px solid rgba(255,255,255,0.3)',
                    'transform': 'translateY(50px)',
                    'opacity': 0,
                    'transition': 'all 1s ease-out',
                    'maxWidth': '900px',
                    'margin': '0 auto',
                    'lineHeight': '1.5'
                }
            ),
            
            dcc.Interval(id='typing-interval', interval=100, n_intervals=0),
            dcc.Interval(id='fade-interval', interval=50, n_intervals=0, max_intervals=1),
            dcc.Interval(id='slide-interval', interval=50, n_intervals=0, max_intervals=1)
        ],
        style={
            'padding': '40px 20px',
            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 16px rgba(0,0,0,0.2)',
            'maxWidth': '1200px',
            'margin': '0 auto',
            "animationDuration": "0.5s"
        },
        className="fade-in"
    )
    
    elif pathname == '/stock/data':
        return review_stock
    elif pathname == '/stock/dashboard':
        return layout_stock
    elif pathname == '/stock/performance':
        return html.Div([
            html.H1("Stock Model Performance", style={'color': '#2a3f8f'}),
            html.P("This section will display stock model performance metrics and visualizations.")
        ])
    elif pathname == '/spotify/data':
        return review_spotify
    elif pathname == '/spotify/dashboard':
        return layout_spotify
    elif pathname == '/spotify/performance':
        return html.Div([
            html.H1("Spotify Model Performance", style={'color': '#2a3f8f'}),
            html.P("This section will display Spotify model performance metrics and visualizations.")
        ])
    else:
        return html.Div([
            html.H1("404: Not found", style={'textAlign': 'center', 'color': '#2a3f8f'}),
            html.P(f"The pathname {pathname} was not recognised...", 
                  style={'textAlign': 'center', 'color': '#2a3f8f'})
        ],
        style={
            'padding': '40px',
            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
            'borderRadius': '15px',
            'maxWidth': '800px',
            'margin': '0 auto'
        })

# Thêm callback để điều khiển các hiệu ứng animation
@app.callback(
    [Output('typing-header', 'children'),
     Output('fade-subtitle', 'style'),
     Output('slide-up-thesis', 'style')],
    [Input('typing-interval', 'n_intervals'),
     Input('fade-interval', 'n_intervals'),
     Input('slide-interval', 'n_intervals')]
)
def update_animations(typing_n, fade_n, slide_n):
    ctx = callback_context

    if not ctx.triggered:
        trigger_id = None
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    full_header = "Welcome to Dashboard"
    if typing_n is None:
        typing_text = ""
    else:
        typing_text = full_header[:min(typing_n, len(full_header))]

    fade_style = {
        'fontFamily': 'Montserrat',
        'textAlign': 'center',
        'color': "#2a3e8fb6",
        'opacity': 1 if typing_text == full_header else 0,
        'transition': 'opacity 1.5s ease-in',
        'marginBottom': '40px'
    }

    slide_style = {
        'fontFamily': 'Montserrat',
        'textAlign': 'center',
        'color': 'white',
        'fontSize': '24px',
        'minHeight': '100px',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'backgroundColor': 'rgba(42, 63, 143, 0.7)',
        'borderRadius': '10px',
        'padding': '20px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
        'border': '2px solid rgba(255,255,255,0.3)',
        'transform': 'translateY(0)' if typing_text == full_header else 'translateY(50px)',
        'opacity': 1 if typing_text == full_header else 0,
        'transition': 'all 1s ease-out',
        'maxWidth': '900px',
        'margin': '0 auto',
        'lineHeight': '1.5'
    }
    
    return typing_text, fade_style, slide_style

# Callback để cập nhật nội dung phụ đề và đề tài sau khi hiệu ứng hoàn thành
@app.callback(
    [Output('fade-subtitle', 'children'),
     Output('slide-up-thesis', 'children')],
    [Input('typing-header', 'children')]
)
def update_content(typing_text):
    if typing_text == "Welcome to Dashboard":
        subtitle = "Select a menu option to view specific dashboards"
        thesis = "ĐỒ ÁN TỐT NGHIỆP - PHÂN TÍCH MẠNG ẢNH HƯỞNG CỘNG ĐỒNG TRONG DỮ LIỆU MẠNG LƯỚI"
        return subtitle, thesis
    return no_update, no_update

review_stock_callbacks(app)
review_spotify_callbacks(app)
register_stock_callbacks(app)
register_spotify_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)