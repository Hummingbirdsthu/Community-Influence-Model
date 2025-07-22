import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.express as px

CUSTOM_BLUES = [
    "#0d1a26", "#102840", "#15395b", "#1a4a75", "#205b90",
    "#266dab", "#2c7fc6", "#3391e0", "#4fa3f7", "#6db7ff",
    "#3b5998", "#1e90ff", "#4682b4", "#5dade2", "#2874a6",
    "#154360", "#2471a3", "#2980b9", "#5499c7", "#85c1e9"
]

df = pd.read_csv('data/us_stocks_data.csv')
df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])

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
        ], style={'width': '97%', 'height': '100%', 'margin': '10px', 'padding': '20px 10px',"backgroundColor":'#f8f9fa','border':'none','borderRadius':'10px'})
    )

def review_stock_callbacks(app):

    # chart 2: summary card
    #  a. cập nhật giá trị summary theo sector-filter
    @app.callback(
        Output('summary-values', 'data'),
        Input('sector-filter', 'value')
    )
    def update_summary_values(selected_sector):
        dff = df.copy()
        if selected_sector:
            dff = dff[dff['Sector'] == selected_sector]
        num_stocks = len(dff)
        num_sectors = dff['Sector'].nunique()
        max_market_cap = round(dff['Market Cap'].max() / 1e9, 2)
        max_market_cap_stock = dff.loc[dff['Market Cap'].idxmax(), 'Id'] if not dff.empty else ''
        max_eps = round(dff['EPS'].max(), 2)
        max_eps_stock = dff.loc[dff['EPS'].idxmax(), 'Id'] if not dff.empty else ''
        return {
            'num_stocks': num_stocks,
            'num_sectors': num_sectors,
            'max_market_cap': max_market_cap,
            'max_market_cap_stock': max_market_cap_stock,
            'max_eps': max_eps,
            'max_eps_stock': max_eps_stock
        }
    # b. callback theo từng card
    @app.callback(
        Output('total-stocks-card', 'children'),
        Input('summary-values', 'data')
    )
    def render_total_stocks_card(data):
        return generate_stats_card(
            title="Total Stocks",
            subtitle=".",
            value=data['num_stocks'],
            image_path="/assets/img/num_stocks.png"
        )

    @app.callback(
        Output('total-sectors-card', 'children'),
        Input('summary-values', 'data')
    )
    def render_total_sectors_card(data):
        return generate_stats_card(
            title="Total Sectors",
            subtitle=".",
            value=data['num_sectors'],
            image_path="/assets/img/num_sectors.png"
        )

    @app.callback(
        Output('max-marketcap-card', 'children'),
        Input('summary-values', 'data')
    )
    def render_max_marketcap_card(data):
        return generate_stats_card(
            title="Max Market Cap",
            subtitle=f"Stock: {data['max_market_cap_stock']}",
            value=f"$ {data['max_market_cap']} (B)",
            image_path="/assets/img/max_marketcap.png"
        )

    @app.callback(
        Output('max-eps-card', 'children'),
        Input('summary-values', 'data')
    )
    def render_max_eps_card(data):
        return generate_stats_card(
            title="Max EPS",
            subtitle=f"Stock: {data['max_eps_stock']}",
            value=data['max_eps'],
            image_path="/assets/img/max_eps.png"
        )



    # chart 3: numerical distribution + sector distribution
    @app.callback(
        [Output('bar-chart', 'figure'),
        Output('pie-chart', 'figure')],
        [Input('sector-filter', 'value'),
        Input('quantitative-dropdown', 'value')]
    )
    def update_charts_3(selected_sector, selected_quant):
        dff = df.copy()
        if selected_sector:
            dff = dff[dff['Sector'] == selected_sector]
        # Bar chart
        bar_fig = px.histogram(
            dff,
            x=selected_quant,
            nbins=5000,
            title=f'Distribution of {selected_quant}',
            template='simple_white',
            color_discrete_sequence=["#0c2146"]
        )
        bar_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            yaxis_title='Count',
            height=400
        )
        
        # Pie char
        sector_counts = dff['Sector'].value_counts()
        top5 = sector_counts.nlargest(5)
        dff['Sector_Pie'] = dff['Sector'].apply(lambda x: x if x in top5.index else 'Khác')
        pie_fig = px.pie(
            dff,
            names='Sector_Pie',
            title='Distribution by Sector (Top 5)',
            template='simple_white',
            color_discrete_sequence=px.colors.sequential.Blues[-1::-1]
        )
        pie_fig.update_traces(
            # sort=False,           # giữ đúng thứ tự đã sắp xếp
            direction='clockwise',
            rotation=-0           # mốc 12h
)
        pie_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=445
        )
        return bar_fig, pie_fig

    # chart 4: heatmap + scatter plot
    @app.callback(
        [Output('heatmap-chart', 'figure'),
        Output('scatter-chart', 'figure')],
        [Input('sector-filter', 'value'),
        Input('scatter-pair-dropdown', 'value')])
    def update_heatmap_scatter(selected_sector, selected_pair):
        dff = df.copy()
        if selected_sector:
            dff = dff[dff['Sector'] == selected_sector]

        # Heatmap cho các biến định lượng
        num_df = dff.select_dtypes(include='number')
        heatmap_fig = px.imshow(
            num_df.corr(),
            text_auto=True,
            color_continuous_scale='Blues',
            title='Correlation Heatmap'
        )
        heatmap_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=545,
            margin=dict(t=40, b=40, l=40, r=40)
        )

        # Scatter plot cho các cặp biến
        scatter_map = {
            'close_volume': ('Volume', 'Close', 'Close vs. Volume'),
            'marketcap_volume': ('Market Cap', 'Volume', 'Market Cap vs. Volume'),
            'marketcap_close': ('Market Cap', 'Close', 'Market Cap vs. Close price')
        }
        x_col, y_col, title = scatter_map[selected_pair]
        scatter_fig = px.scatter(
            dff, x=x_col, y=y_col,
            color='Sector',
            color_discrete_sequence=CUSTOM_BLUES,
            title=title,
            template='simple_white'
        )
        scatter_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=500,
            margin=dict(t=40, b=40, l=40, r=40)
        )

        return heatmap_fig, scatter_fig


    # chart 5: horizontal barchart (top 5 stocks) + data table
    @app.callback(
    [Output('top5-bar-chart', 'figure'),
     Output('top10-change-table', 'data'),
     Output('top10-change-table', 'columns'),
     Output('top10-change-table', 'style_data_conditional')],
    [Input('sector-filter', 'value'),
     Input('top5-num-dropdown', 'value')]
    )
    def update_top5_and_table(selected_sector, selected_num_col):
        dff = df.copy()
        if selected_sector:
            dff = dff[dff['Sector'] == selected_sector]

        # Horizontal barchart: top 5 stocks by selected_num_col
        top5 = dff.nlargest(5, selected_num_col)
        bar_fig = px.bar(
            top5.sort_values(selected_num_col),
            x=selected_num_col,
            y='Id',
            orientation='h',
            color='Id',
            color_discrete_sequence=px.colors.sequential.Blues[-1::-1],
            title=f"Top 5 Stocks by {selected_num_col}"
        )
        bar_fig.update_layout(
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font_color='#111',
            height=415,
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )

        # DataTable
        cols = ['Id', 'Name', 'Sector', 'Close', 'Change', 'Volume', 'Market Cap', 'EPS', 'P/E Ratio', 'Beta (1Y)']
        dff_table = dff.sort_values('EPS', ascending=False).head(12).copy()
        dff_table = dff_table[cols]

        # Tạo cột MarketCap% để tô nền
        max_cap = dff_table['Market Cap'].max()
        dff_table['MarketCap%'] = (dff_table['Market Cap'] / max_cap * 100).round(1)

        # Định nghĩa các cột cho DataTable
        def change_marketcap(val):
            if val > 1e12:
                return f"${val / 1e12:.2f}T"
            elif val > 1e9:
                return f"${val / 1e9:.2f}B"
            elif val > 1e6:
                return f"${val / 1e6:.2f}M"
            else:
                return f"${val:.2f}"
        dff_table['Market Cap'] = dff_table['Market Cap'].apply(change_marketcap)

        def round_2(val):
            return round(val, 2)
        dff_table['EPS'] = dff_table['EPS'].apply(round_2)
        dff_table['P/E Ratio'] = dff_table['P/E Ratio'].apply(round_2)
        dff_table['Beta (1Y)'] = dff_table['Beta (1Y)'].apply(round_2)
        dff_table['Change'] = dff_table['Change'].apply(round_2)

        # Thêm icon cho cột Change
        def change_icon(val):
            if val > 0:
                return f"▲ {val}"
            elif val < 0:
                return f"▼ {val}"
            else:
                return f"▬ {val}"

        dff_table['Change'] = dff_table['Change'].apply(change_icon)


        columns = [
            {'name': 'Name', 'id': 'Id'},
            {'name': 'Sector', 'id': 'Sector'},
            {'name': 'Change', 'id': 'Change', 'presentation': 'markdown'},
            {'name': 'Close', 'id': 'Close'},
            {'name': 'Volume', 'id': 'Volume'},
            {'name': 'Market Cap', 'id': 'Market Cap'},
            {'name': 'EPS', 'id': 'EPS'},
            {'name': 'P/E Ratio', 'id': 'P/E Ratio'},
            {'name': 'Beta (1Y)', 'id': 'Beta (1Y)'}
        ]

        dash_table.DataTable(
            id='top10-change-table',
            style_table={'overflowX': 'auto', 'borderRadius': '10px', 'backgroundColor': '#222b3a', 'border': '20px solid #222b3a'},
            style_cell={
                'fontFamily': 'Montserrat',
                'textAlign': 'center',
                'padding': '5px',
                'backgroundColor': '#222b3a',   # background trắng cho cell
                'color': '#222b3a'              # chữ đen cho cell
            },
            style_header={
                'backgroundColor': '#222b3a',   # header xanh
                'color': 'white',            # chữ trắng
                'fontWeight': 'bold'
            },
            style_data_conditional=[]  # giữ nguyên phần này để callback cập nhật
        )

        style_data_conditional = [
            {
                'if': {'column_id': 'Change'},
                'textAlign': 'center'
            },
            # Change > 0: xanh lá
            {
                'if': {'filter_query': '{Change} contains "▲"', 'column_id': 'Change'},
                'color': 'green',
                'fontWeight': 'bold'
            },
            # Change < 0: đỏ
            {
                'if': {'filter_query': '{Change} contains "▼"', 'column_id': 'Change'},
                'color': 'red',
                'fontWeight': 'bold'
            },
            # Change == 0: xám
            {
                'if': {'filter_query': '{Change} contains "▬"', 'column_id': 'Change'},
                'color': 'gray',
                'fontWeight': 'bold'
            },
            
            # Market Cap: tô nền xanh dương theo %
            {
                'if': {'column_id': 'Market Cap'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {MarketCap%}%, transparent {MarketCap%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            # Tô nền cho các cột khác nếu cần
            {
                'if': {'column_id': 'Volume'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Volume%}%, transparent {Volume%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'EPS'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {EPS%}%, transparent {EPS%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'P/E Ratio'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {P/E Ratio%}%, transparent {P/E Ratio%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Beta (1Y)'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Beta (1Y)%}%, transparent {Beta (1Y)%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Id'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Id%}%, transparent {Id%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Sector'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Sector%}%, transparent {Sector%}%, transparent 100%)"
                ),
                'color': '#111'
            },
            {
                'if': {'column_id': 'Close'},
                'background': (
                    "linear-gradient(90deg, #4fa3f7 0%, #4fa3f7 {Close%}%, transparent {Close%}%, transparent 100%)"
                ),
                'color': '#111'
            },
        ]

        return bar_fig, dff_table.to_dict('records'), columns, style_data_conditional









    @app.callback(
        Output('stock-graph', 'figure'),
        [Input('sector-filter', 'value'),
         Input('metric-selector', 'value'),
         Input('top-n-selector', 'value')]
    )
    def update_graph(selected_sectors, selected_metric, top_n):
        # Lọc theo sector nếu có
        if selected_sectors:
            filtered_df = df[df['Sector'].isin(selected_sectors)].copy()
        else:
            filtered_df = df.copy()
        
        # Sắp xếp và lấy top N theo metric
        if selected_metric == 'Change':
            top_df = filtered_df.nlargest(top_n // 2, columns=[selected_metric])
            bottom_df = filtered_df.nsmallest(top_n // 2, columns=[selected_metric])
            filtered_df = pd.concat([top_df, bottom_df]).sort_values(by=selected_metric, ascending=False)
        else:
            filtered_df = filtered_df.nlargest(top_n, columns=[selected_metric])
        
        # Xử lý màu sắc
        if selected_metric == 'Change':
            colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in filtered_df[selected_metric]]
            marker = {'color': colors}
        else:
            marker = {
                'color': filtered_df[selected_metric],
                'colorscale': [[0, '#1e87cd'], [1, '#ff99cc']],  # Gradient: xanh → hồng
                'cmin': filtered_df[selected_metric].min(),
                'cmax': filtered_df[selected_metric].max(),
                'colorbar': {'title': selected_metric}
            }

        fig = {
            'data': [
                {
                    'x': filtered_df['Name'],
                    'y': filtered_df[selected_metric],
                    'type': 'bar',
                    'text': filtered_df['Symbol'],
                    'marker': marker,
                    'hovertemplate': '<b>%{x}</b><br>' +
                                    f'{selected_metric}: %{{y}}<br>' +
                                    'Symbol: %{text}<extra></extra>'
                }
            ],
            'layout': {
                'title': {
                    'text': f'Top {top_n} {selected_metric} by Company',
                    'font': {'family': 'Montserrat', 'size': 18}
                },
                'xaxis': {
                    'title': 'Company',
                    'tickfont': {'family': 'Montserrat'},
                    'tickangle': -45
                },
                'yaxis': {
                    'title': selected_metric,
                    'tickfont': {'family': 'Montserrat'}
                },
                'hovermode': 'closest',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'font': {'family': 'Montserrat'},
                'margin': {'b': 120}
            }
        }

        return fig





    # WILL BE DELETED

    # @app.callback(
    #     Output('selected-stock-info', 'children'),
    #     [Input('stock-graph', 'clickData')]
    # )
    # def display_click_data(clickData):
    #     if clickData is None:
    #         return html.Div("Click on a bar in the graph to see detailed stock information",
    #                        style={'color': "#182122", 'textAlign': 'center'})
        
    #     symbol = clickData['points'][0]['text']
    #     stock_data = df[df['Symbol'] == symbol].iloc[0]
        
    #     change_class = "positive-change" if stock_data['Change'] >= 0 else "negative-change"
        
    #     return dbc.Row([
    #         dbc.Col([
    #             html.H4(stock_data['Name'], 
    #                    style={'fontWeight': '600', 'marginBottom': '20px'}),
    #             html.P(f"Symbol: {stock_data['Symbol']}"),
    #             html.P(f"Exchange: {stock_data['Exchange']}"),
    #             html.P(f"Sector: {stock_data['Sector']}")
    #         ], width=3),
            
    #         dbc.Col([
    #             html.H5("Pricing", style={'fontWeight': '600', 'marginBottom': '15px'}),
    #             html.P(f"Close: ${stock_data['Close']:.2f}"),
    #             html.P(f"Change: {stock_data['Change']:.2f}%", 
    #                   className=change_class),
    #             html.P(f"Volume: {stock_data['Volume']:,}")
    #         ], width=3),
            
    #         dbc.Col([
    #             html.H5("Valuation", style={'fontWeight': '600', 'marginBottom': '15px'}),
    #             html.P(f"Market Cap: ${stock_data['Market Cap']/1e12:.2f}T"),
    #             html.P(f"P/E Ratio: {stock_data['P/E Ratio']:.2f}"),
    #             html.P(f"EPS: {stock_data['EPS']:.2f}"),
    #             html.P(f"Beta (1Y): {stock_data['Beta (1Y)']:.2f}")
    #         ], width=3),
            
    #         dbc.Col([
    #             html.H5("Performance Chart", style={'fontWeight': '600', 'marginBottom': '15px'}),
    #             dcc.Graph(
    #                 figure={
    #                     'data': [
    #                         {
    #                             'values': [abs(stock_data['Close']), abs(stock_data['Change']), abs(stock_data['P/E Ratio'])],
    #                             'labels': ['Close Price', 'Daily Change', 'P/E Ratio'],
    #                             'type': 'pie',
    #                             'hole': 0.5,
    #                             'marker': {'colors': ["#8fc5e9", "#b9fbd4", "#edcb94"]}
    #                         }
    #                     ],
    #                     'layout': {
    #                         'height': 200,
    #                         'margin': {'l': 20, 'r': 20, 't': 20, 'b': 20},
    #                         'showlegend': False,
    #                         'font': {'family': 'Montserrat'}
    #                     }
    #                 },
    #                 config={'displayModeBar': False}
    #             )
    #         ], width=3)
    #     ])
