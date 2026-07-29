import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import talib
import ta
from sklearn.linear_model import LinearRegression
last_valid_date = None
import warnings
import glob
warnings.filterwarnings("ignore")
#today = date.today()
#today_str = datetime.strftime(today,'%Y-%m-%d')

def price_data(filename):
    df = pd.read_csv(f'D:/CmoneyFactor/{filename}.csv', index_col = 0, parse_dates = True)
    df.fillna(method = 'ffill', inplace = True)
    df = df[~df.index.duplicated(keep='last')]
    return df

def price_index():
    df = pd.read_excel('D:/CmoneyFactor/Index.xlsx', skiprows = [1])
    df = df.iloc[:,4:]
    df = df[~pd.isna(df['股票代號'])]
    df.fillna(0, inplace = True)
    df.股票代號 = df.股票代號.apply(lambda x: x[:8])
    df.股票代號 = df.股票代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df.rename(columns={'股票代號':'日期'}, inplace = True)
    df.set_index('日期', inplace = True)
    return df

def label(data, period=20):
    target = data.shift(-1).rolling(period).apply(np.nanmean, raw=True).dropna().values
    target = np.append(target, np.zeros(data.shape[0] - len(target)) + np.nan) if len(target) < data.shape[0] else target
    global last_valid_date
    if last_valid_date is None:
        last_valid_date = pd.Series(target, index=data.index).last_valid_index()
    return (target > data * 1.006).astype(int)

def data_pre_macro(df):
    df.名稱 = df.名稱.apply(lambda x: x[:8])
    df.名稱 = df.名稱.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df.rename(columns={'名稱':'日期'}, inplace = True)
    df.set_index('日期', inplace = True)
    return df

def dataframe_concat(df1, df2):
    df = pd.concat([df1, df2])
    df.sort_index(inplace = True)
    df = df[~df.index.duplicated(keep='last')]
    return df

def get_data(ticker, filename):
    df = pd.read_csv(f'D:/CmoneyFactor/{filename}.csv', index_col = 0, parse_dates = True)
    df = df.loc[:,ticker]
    return df

def get_trade_data(filename):
    df = pd.read_csv(f'D:/CmoneyFactor/{filename}.csv', index_col = 0, parse_dates = True)
    df = df.loc[df.index.drop_duplicates(keep = 'last')]
    return df

def data_pre():
    df = pd.read_excel('D:/CmoneyFactor/Trade1.xlsx', skiprows = [1], sheet_name = 'Fore_cost')
    df = df.iloc[:,4:]
    df = df[~pd.isna(df['股票代號'])]
    df.fillna(0, inplace = True)
    df.股票代號 = df.股票代號.apply(lambda x: x[:8])
    df.股票代號 = df.股票代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df.rename(columns={'股票代號':'日期'}, inplace = True)
    df.set_index('日期', inplace = True)
    return df

def ins_nbd(d1,d2,d3, period=20):     
    nb = d1 + d2 + d3
    nb_average = nb.rolling(period).mean().fillna(0)
    ins_nbd = ((nb / nb_average).abs()).replace([np.inf, -np.inf], np.nan).fillna(0)
    return ins_nbd

def ins_nbv(d1, d2, d3, d4): 
    nb = d1 + d2+ d3
    return nb / d4

def bias(data, period_f=5, period_s=20): # tt
    return (data / data.rolling(period_f, min_periods=1).mean()).fillna(0) - (data / data.rolling(period_s, min_periods=1).mean()).fillna(0)

def hullma(data, period=60):
    period_2 = period // 2
    period_sqrt = np.floor(np.sqrt(period))
    wma1 = talib.EMA(data, timeperiod=period_2).fillna(0)
    wma2 = talib.EMA(data, timeperiod=period).fillna(0)
    return ((data / talib.EMA(wma1 * 2 - wma2, timeperiod=period_sqrt).fillna(0) - 1) * 100).fillna(0).replace([np.inf, -np.inf], 0)

def mmi(data, period):
    median = data.rolling(period).median()
    return ((data > median) & (data.shift() > median)).rolling(period).mean().fillna(0)
def sma(data, period=20):
    return ((data / talib.SMA(data, timeperiod=period) - 1) * 100).fillna(0)
def macd(data, n_fast=20, n_slow=50, n_sign=20):
    return ta.trend.macd_diff(data, window_fast=n_fast, window_slow=n_slow, window_sign=n_sign, fillna=False).fillna(0)

def bb(data, period=20, times=2):
    up_bb, down_bb = ta.volatility.bollinger_hband(data, window=period, window_dev=times, fillna=False), \
                    ta.volatility.bollinger_lband(data, window=period, window_dev=times, fillna=False)
    return ((data - down_bb) / (up_bb - down_bb)).fillna(0)
def aroon_osc(d1, d2, period=14):
    return talib.AROONOSC(d1, d2, timeperiod=period).fillna(0)
def stoch(close, high, low, fastk=20, slowk=10, slowd=10):
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=fastk, slowk_period=slowk, slowd_period=slowd)
    return slowk, slowd
def wr(close, high, low, period = 20):
    return talib.WILLR(high, low, close, timeperiod=period).iloc[period - 1:].fillna(0)
def rsi(data, period=20):
    return talib.RSI(data, timeperiod=period).iloc[period:]
def cci(close, high, low, period=14): # moment
    return talib.CCI(high, low, close, timeperiod=period).fillna(0)

def acc(data, period=20): # moment
    return (data.shift(period) / (data.shift(2 * period) + data) * 2).fillna(0)

def adx(close, high, low, period=14):
    return talib.ADX(high, low, close, timeperiod=period).fillna(0)
def vpt(close, volume):
    _vpt = ta.volume.volume_price_trend(close, volume)
    return _vpt

def alpha_beta(close,TWA00, rolling = 90):
    
    #index = set(close.index).intersection(set(TWA00.index))
    #TWA00 = TWA00.loc[index,:]
    #TWA00.sort_index(inplace = True)
    #close = close.loc[index,:]
    #close.sort_index(inplace = True)
    
    TWA00_return = TWA00.pct_change()
    TWA00_log_return = np.log(TWA00) - np.log(TWA00.shift(1))
    
    TWA00_return = TWA00_return.replace([np.inf, -np.inf], np.nan).fillna(0)
    TWA00_log_return = TWA00_log_return.replace([np.inf, -np.inf], np.nan).fillna(0)
    close_return = close.pct_change()
    close_log_return= np.log(close) - np.log(close.shift(1))
  
    close_return = close_return.replace([np.inf, -np.inf], np.nan).fillna(0)

    close_log_return = close_log_return.replace([np.inf, -np.inf], np.nan).fillna(0)
    """X = The independent variable which is the Stock
    Y = The dependent variable which is the market
    rolling = The length of the Window
    
    It returns the alphas and the betas of
    the rolling regression
    """
    
    # all the observations
    obs = len(TWA00_return)
    
    # initiate the betas with null values
    betas = np.full(obs, np.nan)
    
    # initiate the alphas with null values
    alphas = np.full(obs, np.nan)
    
    for i in range((obs - rolling)):
        regressor = LinearRegression()
        regressor.fit(TWA00_return.to_numpy()[i : i + rolling+1].reshape(-1,1), close_return.to_numpy()[i : i + rolling+1])
        
        betas[i+rolling]  = round(regressor.coef_[0],6)
        alphas[i+rolling]  = round(regressor.intercept_*100,6)      # scale 100
        
        # To covert to dataframe
        temp = (alphas, betas)
        results = pd.DataFrame(list(zip(*temp)), columns = ['alpha', 'beta'])
        results.index = close.index
        results.sort_index(inplace = True)
        results.fillna(method = 'bfill', inplace = True)
        
    return results

temp = pd.read_excel('D:/CmoneyFactor/Rating.xlsx', skiprows = 4, sheet_name = '2330')
split = temp.columns[0].split()
stock_ticker = split[0]
stock_name = split[1]
data = {'日期':temp['日期'], '股票代號':stock_ticker, '股票名稱':stock_name, '目標價':temp['目標價']}
df = pd.DataFrame(data = data)
df.set_index('日期', inplace = True)

def ADL_industry():
    industry_ADL = pd.DataFrame()
    industry_ADL_MA5 = pd.DataFrame()
    industry_ADL_MA60 = pd.DataFrame()
    industry_ADL_MA_DIFF = pd.DataFrame()
    
    
    df_up = pd.read_excel('D:/CmoneyFactor/UpDown.xlsx', skiprows = [1], sheet_name = 'Up').iloc[:,4:]
    df_down = pd.read_excel('D:/CmoneyFactor/UpDown.xlsx', skiprows = [1], sheet_name = 'Down').iloc[:,4:]
    
    df_up.fillna(0, inplace = True)
    df_up.股票代號 = df_up.股票代號.apply(lambda x: x[:8])
    df_up.股票代號 = df_up.股票代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df_up.rename(columns={'股票代號':'日期'}, inplace = True)
    df_up.set_index('日期', inplace = True)
    
    df_down.fillna(0, inplace = True)
    df_down.股票代號 = df_down.股票代號.apply(lambda x: x[:8])
    df_down.股票代號 = df_down.股票代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df_down.rename(columns={'股票代號':'日期'}, inplace = True)
    df_down.set_index('日期', inplace = True)
    
    industry_ADL = (df_up - df_down).cumsum()
    industry_ADL_MA5 = industry_ADL.rolling(5, min_periods=1).mean()
    industry_ADL_MA60 = industry_ADL.rolling(60, min_periods=1).mean()
    industry_ADL_MA_DIFF = industry_ADL_MA5 - industry_ADL_MA60
    return industry_ADL_MA_DIFF

def ADL_stock(ticker):
    df_ADL = ADL_industry()
    df_industry_code = pd.read_excel('D:/CmoneyFactor/UpDown.xlsx', skiprows = 4, sheet_name = 'Industry_code', dtype = {'股票代號':str})
    industry_code = df_industry_code.loc[df_industry_code['股票代號'] == ticker,'2025產業指數代號']
    df_ADL_stock = df_ADL.loc[:,industry_code]
    return df_ADL_stock

df_open = price_data('Open')
df_close = price_data('Close')
df_volume = price_data('Volume')
df_high = price_data('High')
df_low = price_data('Low')
df_macro = price_data('MacroFactor')
TWA00 = price_data('Index')['TWA00']
df_commodity = price_data('Commodity')
TWA00 = TWA00.loc[df_close.index]

ticker = '2454' #2

def macro(close,processed,ticker):

 # 'y_tb_10', 'y_tb_20', 'y_tb_40', 'y_tb_60', 'y_tbv_10', 'y_tbv_20', 'y_tbv_40', 'y_tbv_60'
    close = close.loc[:,ticker]
    processed = processed.reindex(close.index)
    processed.fillna(method = 'ffill', inplace = True)
    processed = processed.fillna(0)
    processed['y_10'] = label(close, 10)
    processed['y_20'] = label(close, 20)
    processed['y_40'] = label(close, 40)
    processed['y_60'] = label(close, 60)
    return processed
macro = macro(df_close, df_macro, ticker).replace([np.inf, -np.inf], 0)
macro.to_csv(f'D:/Feature_new/macro_{ticker}.csv')

def tech_trend_fn(close, high, low, ticker):
    processed = pd.DataFrame({
        'sma_5': sma(close.loc[:,ticker], 5),
        'sma_10': sma(close.loc[:,ticker], 10),
        'sma_20': sma(close.loc[:,ticker], 20),
        'sma_60': sma(close.loc[:,ticker] ,60),
        'sma_120': sma(close.loc[:,ticker], 120),
        'hullma_20': hullma(close.loc[:,ticker], 20),
        'hullma_60': hullma(close.loc[:,ticker], 60), 
        'hullma_120': hullma(close.loc[:,ticker], 120), 
        # 'hullma_240': hullma(data["收盤價"], 240), 
        'mmi_5': mmi(close.loc[:,ticker], 5), 
        'mmi_10': mmi(close.loc[:,ticker], 10), 
        'mmi_20': mmi(close.loc[:,ticker], 20), 
        'aroon_osc': aroon_osc(high.loc[:,ticker], low.loc[:,ticker], 14), 
        'osc': macd(close.loc[:,ticker]),
        'bb': bb(close.loc[:,ticker]),
        'bias': bias(close.loc[:,ticker]), 
        'alpha':alpha_beta(close.loc[:,ticker],TWA00,90 )['alpha'],  # data = stock price, TWA00 = market, window = 90, alpha = excess return
        'y_10': label(close.loc[:,ticker], 10),
        'y_20': label(close.loc[:,ticker], 20),
        'y_40': label(close.loc[:,ticker], 40),
        'y_60': label(close.loc[:,ticker], 60)
    }).fillna(0)
   
    return processed

tech_trend = tech_trend_fn(df_close, df_high, df_low, ticker).replace([np.inf, -np.inf], np.nan).fillna(0)
tech_trend.to_csv(f'D:/Feature_new/tech_trend_{ticker}.csv')

def moment(close,high, low, volume, ticker):
    k, d = stoch(close.loc[:,ticker] ,high.loc[:,ticker], low.loc[:,ticker])
    processed = pd.DataFrame({
        'rsi': rsi(close.loc[:,ticker]), 
        'k': k,
        'd': d, 
        'wr': wr(close.loc[:,ticker],high.loc[:,ticker], low.loc[:,ticker]),
        'cci': cci(close.loc[:,ticker],high.loc[:,ticker], low.loc[:,ticker]),
        'adx': adx(close.loc[:,ticker],high.loc[:,ticker], low.loc[:,ticker]), 
        'acc_5': acc(close.loc[:,ticker], 5),
        'acc_10': acc(close.loc[:,ticker], 10),
        'acc_20': acc(close.loc[:,ticker], 20),
        'acc_60': acc(close.loc[:,ticker], 60),
        'acc_120': acc(close.loc[:,ticker], 120),
        'vpt': vpt(close.loc[:,ticker], volume.loc[:,ticker]),
        'beta':alpha_beta(close.loc[:,ticker],TWA00,90 )['beta'],   # data = stock price, TWA00 = market, window = 90, beta = relative return to the market
        'y_10': label(close.loc[:,ticker], 10),
        'y_20': label(close.loc[:,ticker], 20),
        'y_40': label(close.loc[:,ticker], 40),
        'y_60': label(close.loc[:,ticker], 60)
        #'y_tb_10': d4['y_tb_10'], 
        #'y_tb_20': d4['y_tb_20'], 
        #'y_tb_40': d4['y_tb_40'], 
        #'y_tb_60': d4['y_tb_60'], 
        #'y_tbv_10': d4['y_tbv_10'], 
        #'y_tbv_20': d4['y_tbv_20'], 
        #'y_tbv_40': d4['y_tbv_40'], 
        #'y_tbv_60': d4['y_tbv_60'], 
    }).fillna(0)
    return processed

moment = moment(df_close, df_high,df_low, df_volume, ticker).replace([np.inf, -np.inf], np.nan).fillna(0)
moment.to_csv(f'D:/Feature_new/moment_{ticker}.csv')

def fundamental_rev(ticker, sheet_name):
    df = pd.read_excel('D:/CmoneyFactor/Fundamental.xlsx', skiprows = [1], sheet_name = sheet_name)
    df = df.iloc[:,4:]
    df.股票代號 = df.股票代號.apply(lambda x: x[:6]+'07')
    df.股票代號 = df.股票代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df.rename(columns={'股票代號':'日期'}, inplace = True)
    df.set_index('日期', inplace = True)
    df = df.loc[:,ticker]
    df.fillna(method='ffill', inplace = True)
    return df

def fundamental_earning(ticker, sheet_name):
    df = pd.read_excel('D:/CmoneyFactor/Fundamental.xlsx', skiprows = [1], sheet_name = sheet_name)
    df = df.iloc[:,4:]
    df.股票代號 = df.股票代號.apply(lambda x: x[:6])
    df.rename(columns={'股票代號':'日期'}, inplace = True)
    df.日期 = df.日期.apply(lambda x: pd.to_datetime(x) + pd.offsets.QuarterEnd() + timedelta(days = 45))
    df.set_index('日期', inplace = True)
    df = df.loc[:,ticker]
    df.fillna(method='ffill', inplace = True)
    return df

def fundamental_commodity_pre(path):
    df = pd.read_excel(f'D:/CmoneyFactor/{path}.xlsx', skiprows = [1])
    df = df.iloc[:,4:]
    df.代號 = df.代號.apply(lambda x: x[:8])
    df.代號 = df.代號.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    df.rename(columns={'代號':'日期'}, inplace = True)
    df.set_index('日期', inplace = True)
    df.fillna(method = 'bfill', inplace = True)
    return df

def fundamental_ticker(ticker):
    #df_commodity = fundamental_commodity('Commodity')
    margin = price_data('Margin')[ticker]
    eps_q = price_data('Eps_q')[ticker]
    eps_qoq = price_data('Eps_qoq')[ticker]
    op_acc = price_data('Op_acc')[ticker]
    op_yoy = price_data('Op_yoy')[ticker]
    op_qoq = price_data('Op_qoq')[ticker]
    earning_yoy = price_data('Earning_yoy')[ticker]
    earning_acc = price_data('Earning_acc')[ticker]
    earning_qoq = price_data('Earning_qoq')[ticker]
    rev_yoy = price_data('Rev_yoy')[ticker]
    rev_mom = price_data('Rev_mom')[ticker]
    rev_acc = price_data('Rev_acc')[ticker]
    
    margin_qoq = pd.DataFrame()
    eps_ttm = pd.DataFrame()
    eps_growth = pd.DataFrame()
    eps_forecast = pd.DataFrame()
    rev_all = pd.DataFrame(data = {'單月營收年成長(%)':rev_yoy,'單月營收月變動(%)':rev_mom,'累計營收成長(%)':rev_acc})
    df_close = pd.read_csv('D:/CmoneyFactor/Close.csv', index_col = 0, parse_dates = True)
    r_all = pd.merge_asof(df_close.loc[:,ticker], rev_all, left_index=True, right_index=True)
    r_all.rename(columns = {ticker:'收盤價'}, inplace = True)
    margin_qoq = margin.diff() / margin.shift().abs() * 100
    eps_ttm = eps_q.rolling(4).sum()
    eps_growth = (eps_q.rolling(4).sum() / eps_q.shift(4).rolling(4).sum() - 1) * 100
    eps_forecast = eps_ttm  * (1 + eps_growth / 100)
    earning_all = pd.DataFrame(data = {'稅前純益季變動率(%)':earning_qoq,'稅前純益年成長率(%)':earning_yoy,'稅前純益累計年成長率(%)':earning_acc,'營業利益季變動率(%)':op_qoq, '營業利益年成長率(%)':op_yoy,
                                      '營業利益累計年成長率(%)':op_acc,'毛利率(%)':margin, '毛利率季變動(%)':margin_qoq, '每股盈餘季變動率(%)':eps_qoq, 'EPS_growth':eps_growth,'EPS_forecast':eps_forecast})
    earning_all.fillna(0, inplace = True)
    f_all =  pd.merge_asof(r_all, earning_all, left_index=True, right_index=True)
    #f_all = pd.merge_asof()
    f_all['PE_forecast'] = f_all['收盤價'] / f_all['EPS_forecast']
    f_all['PEG'] = f_all['PE_forecast'] / f_all['EPS_growth']
    f_all.drop(columns=['收盤價', 'EPS_growth', 'EPS_forecast', 'PE_forecast'], inplace=True)
    f_all.fillna(0, inplace = True)
    
    return f_all

def fundamental(d1, d2, pe, pb, dy):

    def get_river_level(series, split_range=8):
        
        max_value, min_value = series.max(), series.min()
        interval = (max_value - min_value) / split_range
        river_borders = [round(min_value + interval * i, 2) for i in range(split_range + 1)]
        last_value = series[-1]
        level = (last_value - min_value) / (max_value - min_value)
        return level

    def norm(data, whole):
        return normal(whole.loc[:data.name].iloc[:, 0]).iloc[-1]
    
    df_close = pd.read_csv('D:/CmoneyFactor/Close.csv', index_col = 0, parse_dates = True)
    pe_temp =  pe.iloc[3*250-1:].index.to_series().apply(lambda x: get_river_level(pe.loc[:x].iloc[-3*250:], split_range=10))
    pe_temp =  pe_temp[~pe_temp.index.duplicated(keep='first')]
    pe_temp = pe_temp.reindex(df_close.index, fill_value = 0)
    pb_temp =  pb.iloc[3*250-1:].index.to_series().apply(lambda x: get_river_level(pb.loc[:x].iloc[-3*250:], split_range=10))
    pb_temp =  pb_temp[~pb_temp.index.duplicated(keep='first')]
    pb_temp = pb_temp.reindex(d1.index, fill_value = 0)
    dy =  dy[~dy.index.duplicated(keep='first')]
    dy = dy.reindex(df_close.index, fill_value = 0) 
    d2 = d2[~d2.index.duplicated(keep='first')]
    d2 = d2.reindex(df_close.index, fill_value = 0)
    
   
    
    processed = pd.DataFrame({
        
        'PE_trailing': pe_temp, 
        'PEG': d2['PEG'], 
        'PBR': pb_temp, 
        'DY': dy, 
        'R_mom': d2["單月營收月變動(%)"], 
        'R_yoy': d2["單月營收年成長(%)"], 
        'R_acc_yoy': d2["累計營收成長(%)"], 
        'E_qoq': d2["稅前純益季變動率(%)"], 
        'E_yoy': d2["稅前純益年成長率(%)"], 
        'E_acc_yoy': d2["稅前純益累計年成長率(%)"], 
        'Op_qoq': d2["營業利益季變動率(%)"], 
        'Op_yoy': d2["營業利益年成長率(%)"], 
        'Op_acc_yoy': d2["營業利益累計年成長率(%)"], 
        'Gross': d2['毛利率(%)'], 
        'Gross_qoq': d2["毛利率季變動(%)"], 
        'EPS_qoq': d2["每股盈餘季變動率(%)"], 
        'CMDTY': d2['Price_Commod'], 
        'y_10': label(d1, 10),
        'y_20': label(d1, 20),
        'y_40': label(d1, 40),
        'y_60': label(d1, 60)
        #'y_tb_10': d4['y_tb_10'], 
        #'y_tb_20': d4['y_tb_20'], 
        #'y_tb_40': d4['y_tb_40'], 
        #'y_tb_60': d4['y_tb_60'], 
        #'y_tbv_10': d4['y_tbv_10'], 
        #'y_tbv_20': d4['y_tbv_20'], 
        #'y_tbv_40': d4['y_tbv_40'], 
        #'y_tbv_60': d4['y_tbv_60'], 
    }).ffill().fillna(0)

    if d2['Price_Commod'].isnull().values.all():
        processed.drop(['CMDTY'], axis=1, inplace=True)
    return processed

f_all = fundamental_ticker(ticker)
PE_all = pd.read_csv('D:/CmoneyFactor/PE.csv', index_col = 0, parse_dates = True)[ticker]
PB_all = pd.read_csv('D:/CmoneyFactor/PB.csv', index_col = 0, parse_dates = True)[ticker]
DY_all = pd.read_csv('D:/CmoneyFactor/DY.csv', index_col = 0, parse_dates = True)[ticker]