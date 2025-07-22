import pandas as pd

def get_constants(df):
    # Calculate summary metrics
    numerical_cols = ['Close', 'Change', 'Volume', 'Market Cap', 'EPS', 'P/E Ratio', 'Beta']
    # num of stock 
    num_stocks = df['Symbol'].nunique()
    # mun of sector
    num_sectors = df['Sector'].nunique()
    # max_market_cap 
    max_market_cap = df['Market Cap'].max() / 1e9 # convert to billions (B)
    # max_eps 
    max_eps = df['EPS'].max()
    
    # stock with max market cap
    df['Id'] = df['Symbol'].apply(lambda x: str(x)[7:])
    max_market_cap_stock = df.loc[df['Market Cap'] == max_market_cap*1e9, 'Id'].values[0]
    # stock with max eps
    max_eps_stock = df.loc[df['EPS'] == max_eps, 'Id'].values[0]

    return num_stocks, num_sectors, int(max_market_cap), max_market_cap_stock, max_eps, max_eps_stock

    # max_volume = df['Volume'].max() 
    # name_max_volume = df.loc[df['Volume'] == max_volume, 'Id'].values
    # max_volume = df['Volume'].max() / 1e6  # Convert to millions
    # sector_counts = df['Sector'].value_counts().to_dict()
    # stock_count = df['Symbol'].nunique()
    # yesterday_count = stock_count - 25  # ví dụ giả lập hôm qua ít hơn 25 mã
    # delta_stock = f"new: {stock_count - yesterday_count:+,} " + \
    #             f"({round((stock_count - yesterday_count) / stock_count * 100, 2)}%)"