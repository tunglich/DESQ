import requests
import datetime
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

start_date = '2023-01-02'

# 取出 ticker
files = glob.glob(f"D:/model_pred_DES_2023/*")
tickers = [x[-19:-15] for x in files]

# 讀 AGG_DES_prob 檔
def read_AGG_DES(ticker):
    file_path =  glob.glob(f"D:/model_pred_DES_2023/DES_pred_{ticker}*")[0]
    df_AGG_DES = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    df_AGG_DES.columns = ['Score']
    return df_AGG_DES

# 讀 cumSum 檔
def read_cumSum_prob(ticker):
    file_path =  glob.glob(f"./cumSum_prob_6/cumsum_prob_{ticker}*")[0]
    df_cumSum_prob = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    df_cumSum_prob.index.names=['Date']
    df_cumSum_prob.columns = ['Score']
    return df_cumSum_prob

#讀 cumSum檔
def read_cumSum(ticker):
    file_path = glob.glob(f"./cumSum/cusum_{ticker}*")[0]
    df_cumSum = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    df_cumSum.index.names=['Date']
    df_cumSum.columns = ['Score']
    df_cumSum = (df_cumSum + 1) / 2
    return df_cumSum

def get_DES_adj(ticker):
    
    DES = read_AGG_DES(ticker)
    cumSum_prob = read_cumSum_prob(ticker)[DES.index[0].date():]
    cumSum = read_cumSum(ticker)[DES.index[0].date():]
    DES_adj = 0.45*DES + 0.30*cumSum_prob + 0.25*cumSum
    DES_adj.fillna(0.5, inplace = True)

    temp = cumSum * DES_adj
    temp.fillna(method = 'bfill', inplace = True)
    #DES_adj['Score'] = temp.Score.apply(lambda x: 0.5 if x < 0 else x)
    return DES_adj

def price_data(filename):
    df = pd.read_csv(f'./CmoneyFactor/{filename}.csv', index_col = 0, parse_dates = True)
    df.fillna(method = 'ffill', inplace = True)
    df = df[~df.index.duplicated(keep='last')]
    return df

TWSE_weight = pd.read_csv('./CmoneyFactor/WeightTWSE.csv', index_col = 0, parse_dates = True)
OTC_weight = pd.read_csv('./CmoneyFactor/WeightOTC.csv', index_col = 0, parse_dates = True)

def weight_adj_DES_TWSE(ticker, start_date):
    TWSE_weight_stock = TWSE_weight.loc[start_date:, ticker]
    TWSE_weight_stock = TWSE_weight_stock.to_frame()
    TWSE_weight_stock.columns = ['Score']
    DES_adj = get_DES_adj(ticker)[start_date:]
    Weight_adj_DES_TWSE = TWSE_weight_stock * DES_adj
    Weight_adj_DES_TWSE.columns = [f'{ticker}']
    Weight_adj_DES_TWSE.to_csv(f"./TWSE_weighted_score/weightedScore_{ticker}.csv")
    return Weight_adj_DES_TWSE 

def weight_adj_DES_OTC(ticker, start_date):
    OTC_weight_stock = OTC_weight.loc[start_date:, ticker]
    OTC_weight_stock = OTC_weight_stock.to_frame()
    OTC_weight_stock.columns = ['Score']
    DES_adj = get_DES_adj(ticker)[start_date:]
    Weight_adj_DES_OTC = OTC_weight_stock * DES_adj
    Weight_adj_DES_OTC.columns = [f'{ticker}']
    Weight_adj_DES_OTC.to_csv(f"./OTC_weighted_score/weightedScore_{ticker}.csv")
    return Weight_adj_DES_OTC

TWSE_files = pd.read_csv('./CmoneyFactor/WeightTWSE.csv', index_col = 0, parse_dates = True)
TWSE_tickers = TWSE_files.columns.to_list()
intersection_TWSE = list(set(tickers) & set(TWSE_tickers))
intersection_TWSE.sort()
OTC_files = pd.read_csv('./CmoneyFactor/WeightOTC.csv', index_col = 0, parse_dates = True)
OTC_tickers = OTC_files.columns.to_list()
intersection_OTC = list(set(tickers) & set(OTC_tickers))
intersection_OTC.sort()

# Calculate TWSE stock's weight adjusted DES
for ticker in intersection_TWSE:
    try:
        weight_adj_DES_TWSE(ticker, start_date)
    except:
        print(ticker) 

# Calculate TWSE stock's weight adjusted DES
for ticker in intersection_OTC:
    try:
        weight_adj_DES_OTC(ticker, start_date)
    except:
        print(ticker)
        
# Sum all TWSE adj_DES
Weighted_score_TWSE = pd.DataFrame()
for file_path in glob.glob(f"./TWSE_weighted_score/*"):
    temp = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    Weighted_score_TWSE = pd.concat([Weighted_score_TWSE, temp], axis = 1)

Weighted_score_other = TWSE_weight.loc[Weighted_score_TWSE.index[0]:, ~TWSE_weight.columns.isin(Weighted_score_TWSE.columns.to_list())]
Weighted_score_other.fillna(0, inplace = True)

# 修正比重總和為100, 未 cover 公司的 DES 設為0.2
Weighted_score_TWSE['Other'] = (100 - Weighted_score_other.sum(axis=1))*0.25  
Weighted_score_TWSE['SUM'] = Weighted_score_TWSE.sum(axis=1,skipna=True, numeric_only=True)
Weighted_score_TWSE.to_csv(f"./Weighted_score/Weight_TWSE.csv")

# Sum all OTC adj_DES
Weighted_score_OTC = pd.DataFrame()
for file_path in glob.glob(f"./OTC_weighted_score/*"):
    temp = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    Weighted_score_OTC = pd.concat([Weighted_score_OTC, temp], axis = 1)
Weighted_score_other = OTC_weight.loc[Weighted_score_OTC.index[0]:, ~OTC_weight.columns.isin(Weighted_score_OTC.columns.to_list())]
Weighted_score_other.fillna(0, inplace = True)    

# 修正比重總和為100, 未 cover 公司的 DES 設為50
Weighted_score_OTC['Other'] = (100 - Weighted_score_other.sum(axis=1))*0.65  
Weighted_score_OTC['SUM'] = Weighted_score_OTC.sum(axis=1,skipna=True, numeric_only=True)
Weighted_score_OTC.to_csv(f"./Weighted_score/Weight_OTC.csv")

# 將分數平滑化
rolling_TWSE = Weighted_score_TWSE['SUM'].rolling(10).mean().bfill()
rolling_OTC = Weighted_score_OTC['SUM'].rolling(8).mean().bfill()
rolling_OTC = rolling_OTC[~rolling_OTC.index.duplicated(keep='last')]

diff_TWSE = (rolling_TWSE - rolling_TWSE.shift(1)).bfill()
#diff_TWSE = diff_TWSE - diff_TWSE.shift(1) # 二次差分
diff_OTC = (rolling_OTC - rolling_OTC.shift(1)).bfill()

# 圖形顯示範圍：固定從 2023-01-01 開始（避免舊殘留 csv 把起點往前拉）
chart_start = '2023-01-01'
# warmup_days：跳過前 N 個交易日，避免 rolling 起算 + 資料初始化造成的左側尖峰
warmup_days = 15
rolling_TWSE = rolling_TWSE.loc[chart_start:].iloc[warmup_days:]
diff_TWSE = diff_TWSE.loc[chart_start:].iloc[warmup_days:]
rolling_OTC = rolling_OTC.loc[chart_start:].iloc[warmup_days:]
diff_OTC = diff_OTC.loc[chart_start:].iloc[warmup_days:]

# 拉出指數為台灣報酬指數, OTC指數
TWA02 = price_data('Index')['TWA02']
TWA02 = TWA02.loc[rolling_TWSE.index[0]:rolling_TWSE.index[-1]]

OTC = price_data('Index')['TWC50']
OTC = OTC.loc[rolling_OTC.index[0]:rolling_OTC.index[-1]]

# ---- Seaborn 主題 ----
sns.set_theme(
    style='whitegrid',
    context='notebook',
    font='Microsoft JhengHei',
    rc={
        'axes.unicode_minus': False,
        'axes.edgecolor': '#666666',
        'axes.linewidth': 0.8,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'semibold',
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    },
)

# 統一配色（取自 seaborn deep / muted 配色微調）
PALETTE = {
    'index':       '#1F4E79',  # 深藍：指數
    'score':       '#E07B00',  # 暖橘：AIQuant Score
    'threshold':   '#C0392B',  # 暗紅：門檻線
    'fill_above':  '#E74C3C',  # 上方填色（>= threshold）
    'fill_below':  '#27AE60',  # 下方填色（< threshold）
    'bar_up':      '#E74C3C',  # 差分正
    'bar_down':    '#27AE60',  # 差分負
    'bar_zero':    '#BDC3C7',
    'text_muted':  '#555555',
    'text_light':  '#999999',
}


def plot_score_chart(index_series, score_series, diff_series, *,
                     index_label, threshold, title, output_path):
    """畫指數 vs AIQuant Score 雙軸圖 + Score 一階差分 bar 圖（seaborn 美化版）。"""
    fig = plt.figure(figsize=(14, 7.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.6, 1.1], hspace=0.20, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # ---- 上：指數（左軸） + AIQuant Score（右軸） ----
    ax_score = ax_top.twinx()

    sns.lineplot(
        x=index_series.index, y=index_series.values,
        ax=ax_top, color=PALETTE['index'], linewidth=1.9,
        label=index_label, legend=False,
    )
    sns.lineplot(
        x=score_series.index, y=score_series.values,
        ax=ax_score, color=PALETTE['score'], linewidth=1.9,
        label='AIQuant Score', legend=False,
    )

    # 門檻線
    ax_score.axhline(
        threshold, color=PALETTE['threshold'],
        linewidth=1.1, linestyle='--', alpha=0.75,
        label=f'門檻 {threshold}',
    )

    # 門檻上下填色
    ax_score.fill_between(
        score_series.index, threshold, score_series.values,
        where=(score_series.values >= threshold),
        color=PALETTE['fill_above'], alpha=0.13, interpolate=True,
    )
    ax_score.fill_between(
        score_series.index, threshold, score_series.values,
        where=(score_series.values < threshold),
        color=PALETTE['fill_below'], alpha=0.13, interpolate=True,
    )

    # 標出最新 Score 點 + 數值
    last_x = score_series.index[-1]
    last_y = float(score_series.values[-1])
    ax_score.scatter(
        [last_x], [last_y],
        color=PALETTE['score'], s=55, zorder=6,
        edgecolor='white', linewidth=1.4,
    )
    ax_score.annotate(
        f' {last_y:.2f}',
        xy=(last_x, last_y),
        xytext=(8, 0), textcoords='offset points',
        color=PALETTE['score'], fontsize=10.5, fontweight='bold',
        va='center',
    )

    # 軸標 / 刻度配色
    ax_top.set_ylabel(index_label, color=PALETTE['index'], fontsize=11)
    ax_score.set_ylabel('AIQuant Score', color=PALETTE['score'], fontsize=11)
    ax_top.tick_params(axis='y', labelcolor=PALETTE['index'])
    ax_score.tick_params(axis='y', labelcolor=PALETTE['score'])
    ax_top.set_xlabel('')

    # 合併左右兩軸 legend（提高 zorder，避免被線條蓋住）
    h_top, l_top = ax_top.get_legend_handles_labels()
    h_score, l_score = ax_score.get_legend_handles_labels()
    leg = ax_top.legend(
        h_top + h_score, l_top + l_score,
        loc='upper left', frameon=True, fontsize=10,
        framealpha=0.96, edgecolor='#BBBBBB', facecolor='white',
        borderpad=0.7, handlelength=2.2, borderaxespad=0.6,
    )
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(20)  # 蓋過所有線條與填色

    ax_top.set_title(title, fontsize=15, pad=12, color='#222222')
    ax_top.grid(True, axis='y', alpha=0.4)
    ax_score.grid(False)
    sns.despine(ax=ax_top, top=True)
    sns.despine(ax=ax_score, top=True, right=False)

    # ---- 下：Score 一階差分 ----
    bar_colors = np.where(
        diff_series.values > 0, PALETTE['bar_up'],
        np.where(diff_series.values < 0, PALETTE['bar_down'], PALETTE['bar_zero']),
    )
    ax_bot.bar(
        diff_series.index, diff_series.values,
        color=bar_colors, width=1.0, alpha=0.85, edgecolor='none',
        zorder=2,
    )
    ax_bot.axhline(0, color='#444444', linewidth=0.7, zorder=3)

    # 5 日均線：把日內雜訊平滑掉，看出短期趨勢方向
    diff_ma5 = (
        pd.Series(diff_series.values, index=diff_series.index)
        .rolling(5, min_periods=1).mean()
    )
    ax_bot.plot(
        diff_ma5.index, diff_ma5.values,
        color='#1B1B1B', linewidth=1.0, alpha=0.9,
        label='5MA', zorder=4,
    )
    ax_bot.legend(
        loc='upper left', fontsize=9, frameon=True,
        framealpha=0.92, edgecolor='#CCCCCC',
        borderpad=0.4, handlelength=1.8,
    ).set_zorder(20)

    # 動態 ylim：以 99% 分位數為基準（去除極端離群值），維持上下對稱
    diff_abs = np.abs(diff_series.values)
    diff_abs = diff_abs[~np.isnan(diff_abs)]
    if diff_abs.size > 0:
        q99 = float(np.quantile(diff_abs, 0.99))
        y_lim = max(q99 * 1.15, 0.5)  # 最少留 ±0.5 的視覺空間
    else:
        y_lim = 1.0
    ax_bot.set_ylim(-y_lim, y_lim)

    ax_bot.set_ylabel('Δ Score', fontsize=10, color=PALETTE['text_muted'])
    ax_bot.set_title(
        'Score Difference (Day-over-Day)',
        fontsize=12, color=PALETTE['text_muted'], pad=6,
    )
    ax_bot.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax_bot, top=True, right=True)

    # 日期軸
    ax_bot.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=12))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    fig.autofmt_xdate()

    # 右下角資料更新日期
    fig.text(
        0.995, 0.005,
        f'資料更新: {score_series.index[-1].strftime("%Y-%m-%d")}',
        ha='right', va='bottom',
        fontsize=8.5, color=PALETTE['text_light'],
    )

    plt.tight_layout(rect=[0, 0.015, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# TWSE
dateStr_TWSE = datetime.datetime.strftime(TWA02.index[-1].date(), "%Y-%m-%d")
plot_score_chart(
    TWA02, rolling_TWSE, diff_TWSE,
    index_label='TWA02',
    threshold=50,
    title='加權股價報酬指數 vs AIQuant Score',
    output_path=f"./AI_market_score/TWSE_{dateStr_TWSE}.jpg",
)

# OTC
dateStr_OTC = datetime.datetime.strftime(OTC.index[-1].date(), "%Y-%m-%d")
plot_score_chart(
    OTC, rolling_OTC, diff_OTC,
    index_label='櫃買50 (TWC50)',
    threshold=55,
    title='櫃買50指數 vs AIQuant Score',
    output_path=f"./AI_market_score/OTC_{dateStr_OTC}.jpg",
)
