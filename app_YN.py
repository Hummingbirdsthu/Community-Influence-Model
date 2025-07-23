from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from components.layout_stock import layout_stock
from components.layout_spotify import layout_spotify
from components.review_stock import review_stock
from components.review_spotify import review_spotify
from components.model_spotify import model_spotify
from components.model_stock import model_stock
from callbacks.register_stock_callbacks import register_stock_callbacks
from callbacks.review_stock_callbacks import review_stock_callbacks
from callbacks.review_spotify_callbacks import review_spotify_callbacks
from callbacks.register_spotify_callbacks import register_spotify_callbacks
from callbacks.model_spotify_callbacks import model_spotify_callbacks
from callbacks.model_stock_callbacks import model_stock_callbacks
from scipy import stats
import base64
from dash import dcc
from datetime import datetime
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ],
    suppress_callback_exceptions=True,
    assets_folder='assets',  # Chỉ định thư mục assets
    assets_url_path='assets'  # Đường dẫn URL cho assets
)
app.css.append_css({
    'external_url': '/assets/custom.css'
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
                            "position": "absolute",
                            "right": "10px",
                            "top": "15px",
                            "zIndex": 1001,
                            "transition": "all 0.3s ease"
                        }
                    ),
                    html.H2("Menu", id="menu-title", style={
                        "color": "#2a3f8f",
                        "fontFamily": "Montserrat",
                        "fontWeight": "600",
                        "marginBottom": "20px",
                        "display": "inline-block",
                        "transition": "opacity 0.3s ease"
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
                
                html.Hr(style={"borderTop": "1px solid rgba(42,63,143,0.3)"}),
                dbc.Nav(
                    [
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-home me-2"),
                                html.Span("Home", className="nav-link-text")
                            ],
                            href="/",
                            active="exact",
                            id="home-link",
                            style={
                                "color": "#2a3f8f",
                                "fontFamily": "Montserrat",
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
                                    "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
                                    "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
                                                "fontFamily": "Montserrat",
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
        "width": "200px",
        "fontFamily": "Montserrat",
        "boxShadow": "2px 0 15px rgba(0,0,0,0.1)",
        "zIndex": 1000,
        "transition": "all 0.3s ease",
        "overflowY": "auto",  # Thêm thanh cuộn nếu nội dung dài
    }
)

header = html.Div(
    [
        # Logo and App Name
        html.Div(
            [
                html.Img(src="assets/8121733.jpg", style={"height": "30px", "marginRight": "10px"}),
                html.H4("Network Community Analytics", style={"margin": "0", "color": "#2a3f8f", "fontFamily": "'Montserrat", "fontWeight": "600"})
            ],
            style={"display": "flex", "alignItems": "center", "marginLeft": "20px"}
        ),
        
        # Right-side controls
        html.Div(
            [
                # Calendar Date Picker
                html.Div(
                    [
                        html.I(className="fas fa-calendar-alt", style={
                            "color": "#2a3f8f",
                            "marginRight": "8px",
                            "cursor": "pointer"
                        }),
                        dcc.DatePickerSingle(
                            id='date-picker',
                            date=datetime.now().date(),
                            display_format='DD/MM/YYYY',
                            style={
                                'border': 'none',
                                'background': 'transparent',
                                'color': '#2a3f8f',
                                'fontFamily': "Montserrat",
                                'fontWeight': '500',
                                'cursor': 'pointer',
                                'width': '120px'
                            },
                            calendar_orientation='vertical',  # Fixed to use valid value
                            clearable=False,
                            with_portal=True  # Makes calendar appear in center of screen
                        )
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginRight": "20px",
                        "padding": "6px 12px",
                        "borderRadius": "20px",
                        "background": "rgba(255,255,255,0.7)"
                    }
                ),
                
                # Search bar
                dbc.InputGroup(
                    [
                        dbc.Input(placeholder="Search...", style={"borderRadius": "20px", "border": "1px solid #e0e0e0"}),
                        dbc.InputGroupText(
                            html.I(className="fas fa-search"),
                            style={"backgroundColor": "transparent", "border": "none", "cursor": "pointer"}
                        ),
                    ],
                    style={"width": "250px", "marginRight": "20px"}
                ),
                
                # Notification dropdown
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("New stock data available", header=True),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("System update scheduled"),
                        dbc.DropdownMenuItem("New messages (3)"),
                    ],
                    label=html.Span(
                        [
                            html.I(className="fas fa-bell"),
                            html.Span("3", className="notification-badge")
                        ]
                    ),
                    align_end=True,
                    in_navbar=True,
                    className="notification-dropdown",
                    toggle_style={
                        "color": "#2a3f8f",
                        "background": "none",
                        "border": "none",
                        "fontSize": "1rem",
                        "cursor": "pointer",
                        "position": "relative"
                    },
                    style={"marginRight": "15px"}
                ),
                
                # Account dropdown
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem([
                            html.I(className="fas fa-user", style={"marginRight": "8px"}),
                            "Profile"
                        ], href="#"),

                        dbc.DropdownMenuItem([
                            html.I(className="fas fa-cog", style={"marginRight": "8px"}),
                            "Settings"
                        ], href="#"),

                        dbc.DropdownMenuItem(divider=True),

                        dbc.DropdownMenuItem([
                            html.I(className="fas fa-sign-out-alt", style={"marginRight": "8px"}),
                            "Log Out"
                        ], href="#"),
                    ],
                    label=[
                        html.I(className="fas fa-user-circle", style={"marginRight": "8px"}),
                        "Admin User",
                        html.I(className="fas fa-caret-down", style={"marginLeft": "8px"})
                    ],
                    align_end=True,
                    in_navbar=True,
                    className="account-dropdown",
                    toggle_style={
                        "color": "#2a3f8f",
                        "background": "none",
                        "border": "none",
                        "fontSize": "0.9rem",
                        "cursor": "pointer",
                        "fontFamily": "'Montserrat",
                        "fontWeight": "500",
                        "display": "flex",
                        "alignItems": "center"
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
        "left": "0px",
        "right": 0,
        "height": "60px",
        "backgroundColor": "rgba(255, 255, 255, 0.95)",
        "backdropFilter": "blur(8px)",
        "WebkitBackdropFilter": "blur(8px)",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "zIndex": 999,
        "boxShadow": "0 2px 15px rgba(0,0,0,0.1)",
        "transition": "all 0.3s ease",
        "padding": "0 10px",
        "borderBottom": "1px solid rgba(0,0,0,0.05)"
    },
    id="header"
)
app.layout = html.Div(
    [
        dcc.Location(id='url'),
        dcc.Store(id='sidebar-state', data={'is_open': True}),
        sidebar,
        header,
        html.Div(
            id='page-content',
            style={
                            "position": "absolute",
                            "top": "50px",
                            "left": "250px",  # Khi sidebar mở
                            "right": "0",
                            "bottom": "0",
                            "padding": "2rem",
                            "overflowY": "auto",
                            "backgroundImage": 'url("/assets/sl_122221_47450_06.jpg")',
                            "backgroundSize": "cover",
                            "backgroundPosition": "center",
                            "backgroundRepeat": "no-repeat",
                            "backgroundAttachment": "fixed",
                            "color": "white",
                            "fontFamily": "Montserrat",
                            "transition": "all 0.3s ease",
                            "zIndex": 1
                        }
        ),
      
    ],
    style={
    "fontFamily": "Montserrat",
    "minHeight": "100vh",
    "position": "relative",
    "backgroundColor": "#f4f4fc"
   },
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
            // Khi mở sidebar
            sidebar.style.width = "250px";
            content.style.left = "250px";  // Sát lề sidebar
            header.style.left = "250px";
            menuTitle.style.opacity = "1";
            navTexts.forEach(el => el.style.display = "inline");
            submenuArrows.forEach(el => el.style.display = "inline");
            toggleBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
            toggleBtn.style.left = "235px";  // Điều chỉnh vị trí nút toggle
        } else {
            // Khi đóng sidebar
            sidebar.style.width = "30px";  // Rất nhỏ, chỉ đủ chứa nút toggle
            content.style.left = "30px";   // Content sát vào sidebar
            header.style.left = "30px";
            menuTitle.style.opacity = "0";
            navTexts.forEach(el => el.style.display = "none");
            submenuArrows.forEach(el => el.style.display = "none");
            toggleBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
            toggleBtn.style.left = "5px";  // Điều chỉnh vị trí nút toggle
        }

        return { is_open: is_open };
    }
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
            # Background with subtle pattern
            dbc.Container(
                [
                    dbc.Row(
                        [
                            # Left Column - Text Content (7/12 for better text emphasis)
                            dbc.Col(
                                html.Div(
                                    [
                                        # Typing header with improved animation
                                    html.Div(
                                        id="typing-header",
                                        style={
                                            'fontFamily': '"Montserrat", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',  # Fallback fonts
                                            'color': 'transparent',
                                            'fontWeight': '900', 
                                            'fontSize': 'clamp(2.5rem, 6vw, 4rem)',  # Tăng kích thước động
                                            'textShadow': '1px 1px 3px rgba(0, 0, 0, 0.15)',
                                            'marginBottom': '25px',
                                            'textAlign': 'left',
                                            'lineHeight': '1.15',  # Giảm lineHeight cho tiêu đề ngắn
                                            'background': 'linear-gradient(95deg, #2a3f8f 0%, #3b5bff 70%, #6a8eff 100%)',  # Gradient mượt hơn
                                            'backgroundClip': 'text',
                                            'WebkitBackgroundClip': 'text',
                                            'letterSpacing': '-0.5px',  # Giãn cách chữ tinh tế
                                            'paddingBottom': '8px',  # Thêm khoảng cách dưới
                                            'borderBottom': '2px solid rgba(42, 63, 143, 0.2)',  # Gạch chân trang trí
                                            'display': 'inline-block',  # Giữ cho borderBottom chỉ dài bằng chữ
                                            'transition': 'all 0.3s ease-out'  # Hiệu ứng hover
                                        }
                                    ),
                                        
                                        # Subtitle with improved fade effect
                                        html.P(
                                            id="fade-subtitle",
                                            style={
                                                'fontFamily': 'Montserrat',
                                                'color': "#ffffff",
                                                'fontSize': 'clamp(1rem, 2vw, 1.25rem)',
                                                'opacity': 0,
                                                'transition': 'opacity 1s ease-in, transform 1s ease-out',
                                                'marginBottom': '30px',
                                                'textAlign': 'left',
                                                'transform': 'translateY(20px)',
                                                'lineHeight': '1.6'
                                            }
                                        ),
                                        
                                        # Thesis description with enhanced card design
                                        html.Div(
                                            id="slide-up-thesis",
                                            style={
                                                'fontFamily': 'Montserrat, sans-serif',
                                                'color': 'white',
                                                'fontSize': 'clamp(0.9rem, 1.5vw, 1.1rem)',
                                                'background': 'linear-gradient(145deg, #2a3f8f 0%, #3b4db1 100%)',
                                                'borderRadius': '12px',
                                                'padding': '30px',
                                                'boxShadow': '0 10px 30px rgba(42, 63, 143, 0.3)',
                                                'border': 'none',
                                                'transform': 'translateY(50px)',
                                                'opacity': 0,
                                                'transition': 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
                                                'lineHeight': '1.8',
                                                'textAlign': 'left',
                                                'marginBottom': '40px',
                                                'position': 'relative',
                                                'overflow': 'hidden'
                                            },
                                            children=[
                                                html.Div(style={
                                                    'position': 'absolute',
                                                    'top': '-50px',
                                                    'right': '-50px',
                                                    'width': '200px',
                                                    'height': '200px',
                                                    'background': 'rgba(255,255,255,0.1)',
                                                    'borderRadius': '50%'
                                                }),
                                                html.Div("This dashboard presents my thesis work analyzing...", 
                                                         style={'position': 'relative', 'zIndex': 1})
                                            ]
                                        ),
                                        
                                        # CTA Buttons with hover effects
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Explore Dashboards →",
                                                    href="/stock/dashboard",
                                                    color="primary",
                                                    className="me-3",
                                                    style={
                                                        'background': 'linear-gradient(90deg, #2a3f8f 0%, #4a6bff 100%)',
                                                        'border': 'none',
                                                        'fontFamily': 'Montserrat, sans-serif',
                                                        'fontWeight': '600',
                                                        'padding': '12px 30px',
                                                        'fontSize': '1rem',
                                                        'boxShadow': '0 4px 15px rgba(42, 63, 143, 0.4)',
                                                        'transition': 'all 0.3s ease',
                                                        'borderRadius': '8px'
                                                    },
                                                    id="cta-button"
                                                ),
                                                dbc.Button(
                                                "View Thesis PDF",
                                                id="thesis-pdf-button",
                                                color="secondary",
                                                outline=True,
                                                href="/assets/do_an_tot_nghiep.pdf",  # Thêm href trực tiếp
                                                target="_blank",           # Mở tab mới
                                                external_link=True,       # Bắt buộc với Dash
                                                style={
                                                    'border': '2px solid #2a3f8f',
                                                    'color': '#2a3f8f',
                                                    'fontFamily': 'Montserrat, sans-serif',
                                                    'fontWeight': '600',
                                                    'padding': '12px 30px',
                                                    'fontSize': '1rem',
                                                    'transition': 'all 0.3s ease',
                                                    'borderRadius': '8px',
                                                    'background': 'transparent',
                                                    'cursor': 'pointer'   # Thêm con trỏ chuột
                                                },
                                                className="me-1"
                                            )
                                            ],
                                            style={'display': 'flex', 'gap': '15px', 'flexWrap': 'wrap'}
                                        )
                                    ],
                                    style={
                                        'padding': '50px 40px',
                                        'backgroundColor': 'transparent',
                                        'borderRadius': '16px',
                                        'backdropFilter': 'blur(5px)'
                                    }
                                ),
                                md=7,
                                lg=7,
                                style={'padding': '20px'}
                            ),
                            
                            # Right Column - Image (5/12)
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(
                                            style={
                                                'position': 'relative',
                                                'width': '100%',
                                                'height': '100%',
                                                'minHeight': '400px',
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'alignItems': 'center'
                                            },
                                            children=[
                                                html.Img(
                                                    src="/assets/2708754.png",  
                                                    style={
                                                        'width': '100%',
                                                        'height': 'auto',
                                                        'maxWidth': '600px',
                                                        'filter': 'drop-shadow(0 15px 30px rgba(42, 63, 143, 0.2))',
                                                        'transition': 'transform 0.5s ease',
                                                        'transform': 'scale(0.95)'
                                                    },
                                                    id="hero-image"
                                                ),
                                                html.Div(
                                                    style={
                                                        'position': 'absolute',
                                                        'bottom': '0',
                                                        'left': '0',
                                                        'width': '100%',
                                                        'height': '40%',
                                                        'background': 'linear-gradient(transparent, rgba(42, 63, 143, 0.05))',
                                                        'zIndex': 0
                                                    }
                                                )
                                            ]
                                        )
                                    ],
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center',
                                        'alignItems': 'center',
                                        'height': '100%',
                                        'padding': '20px'
                                    }
                                ),
                                md=5,
                                lg=5,
                                style={'padding': '20px'}
                            )
                        ],
                        style={
                            'alignItems': 'center',
                            'minHeight': 'calc(100vh - 80px)'
                        },
                        className="g-4"
                    )
                ],
                fluid=True,
                style={
                    'maxWidth': '1400px',
                    'padding': '40px 20px',
                    'position': 'relative',
                    'zIndex': 1
                }
            ),
            
            # Footer
            html.Footer(
                style={
                    'width': '100%',
                    'textAlign': 'center',
                    'padding': '20px',
                    'color': '#4a5568',
                    'fontFamily': 'Montserrat, sans-serif',
                    'fontSize': '0.9rem',
                    'background': 'rgba(255,255,255,0.7)',
                    'borderTop': '1px solid rgba(0,0,0,0.05)'
                },
                children=[
                    "© 2023 Thesis Dashboard | ",
                    html.A("GitHub Repository", href="#", style={'color': '#2a3f8f', 'textDecoration': 'none'}),
                    " | ",
                    html.A("Contact", href="#", style={'color': '#2a3f8f', 'textDecoration': 'none'})
                ]
            ),
            
            # Animation intervals
            dcc.Interval(id='typing-interval', interval=100, n_intervals=0),
            dcc.Interval(id='fade-interval', interval=50, n_intervals=0, max_intervals=1),
            dcc.Interval(id='slide-interval', interval=50, n_intervals=0, max_intervals=1),
            
            # Hover effects
            dcc.Store(id='hover-effects-store'),
        ],
        style={
            'width': '100%',
            'minHeight': '100vh',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between'
        })
    
    # Rest of your pathname conditions remain the same...
    elif pathname == '/stock/data':
        return review_stock
    elif pathname == '/stock/dashboard':
        return layout_stock
    elif pathname == '/stock/performance':
        return model_stock
    elif pathname == '/spotify/data':
        return review_spotify
    elif pathname == '/spotify/dashboard':
        return layout_spotify
    elif pathname == '/spotify/performance':
        return model_spotify
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

    # Cải tiến fade_style với icon và animation
    fade_style = {
        'fontFamily': 'Montserrat, sans-serif',
        'textAlign': 'center',
        'color': "#2a3f8f",
        'opacity': 1 if typing_text == full_header else 0,
        'transition': 'opacity 0.8s ease-in, transform 0.8s ease-out',
        'marginBottom': '30px',
        'transform': 'translateY(0)' if typing_text == full_header else 'translateY(10px)',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'gap': '10px'
    }

    # Nâng cấp slide_style với icon và hiệu ứng
    slide_style = {
        'fontFamily': 'Montserrat, sans-serif',
        'textAlign': 'center',
        'color': 'white',
        'fontSize': '26px',
        'fontWeight': '700',
        'minHeight': '140px',  # Tăng chiều cao để chứa icon
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'background': 'linear-gradient(135deg, #2a3f8f 0%, #4a6bff 100%)',
        'borderRadius': '16px',
        'padding': '30px',
        'boxShadow': '0 12px 24px rgba(42, 63, 143, 0.3)',
        'border': '1px solid rgba(255,255,255,0.3)',
        'transform': 'translateY(0) scale(1)' if typing_text == full_header else 'translateY(50px) scale(0.95)',
        'opacity': 1 if typing_text == full_header else 0,
        'transition': 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
        'maxWidth': '900px',
        'margin': '0 auto 40px auto',
        'lineHeight': '1.7',
        'letterSpacing': '0.8px',
        'textShadow': '0 2px 6px rgba(0,0,0,0.2)',
        'position': 'relative',
        'overflow': 'hidden',
        'zIndex': '1'
    }

    # Thêm icon vào nội dung (sẽ được hiển thị qua callback khác)
    icon_style = {
        'fontSize': '48px',
        'marginBottom': '20px',
        'color': 'rgba(255,255,255,0.9)',
        'textShadow': '0 2px 8px rgba(0,0,0,0.2)',
        'transition': 'all 0.5s ease-out'
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
@app.callback(
    Output("thesis-pdf-button", "n_clicks"),
    Input("thesis-pdf-button", "n_clicks"),
    prevent_initial_call=True
)
def open_pdf(n_clicks):
    if n_clicks:
        pdf_path = "/assets/ĐỒ_ÁN_TỐT_NGHIỆP.pdf"
        # Mở trong tab mới
        return dcc.Location(pathname=pdf_path, id="pdf-redirect", refresh=True)
    return None

review_stock_callbacks(app)
review_spotify_callbacks(app)
register_stock_callbacks(app)
register_spotify_callbacks(app)
model_spotify_callbacks(app)
model_stock_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)