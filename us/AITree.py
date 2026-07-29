import sys
# Windows 預設 console 編碼是 cp950 (Big5),無法 print emoji (🚀 ✅ ❌ 等)。
# 在 import 其他模組前先把 stdout / stderr 切到 utf-8，避免 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import finlab
from finlab import data
import plotly.express as px
import plotly.offline as offline
import plotly.graph_objects as go
import glob
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings("ignore")

# 登入
api_token = "PG323UEltzZHHyhR4wg+OIGmrIIJhedAGIOGi4udKKyCCG7Kjmpl7MHXObhql4XH#vip_m"
finlab.login(api_token=api_token)

# 已下市/已停止追蹤的標的清單,日後再有就加進這個 set。
DELISTED_TICKERS = {
    '3454',  # 晶睿 (已下市)
}

# 取出 ticker (排除已知下市)
files = glob.glob(f"D:/model_pred_DES_2023/*")
tickers = [x[-19:-15] for x in files]
tickers = [t for t in tickers if t not in DELISTED_TICKERS]

def read_cumSum(ticker):
    file_path = glob.glob(f"./cumSum/cusum_{ticker}*")[0]
    df_cumSum = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df_cumSum.index.names = ['Date']
    df_cumSum.columns = ['Score']
    df_cumSum = (df_cumSum + 1) / 2
    return df_cumSum

def read_cumSum_prob(ticker):
    file_path = glob.glob(f"./cumSum_prob_6/cumsum_prob_{ticker}*")[0]
    df_cumSum_prob = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df_cumSum_prob.index.names = ['Date']
    df_cumSum_prob.columns = ['Score']
    return df_cumSum_prob

def read_AGG_DES(ticker):
    file_path = glob.glob(f"D:/model_pred_DES_2023/DES_pred_{ticker}*")[0]
    df_AGG_DES = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df_AGG_DES.columns = ['Score']
    return df_AGG_DES

def get_DES_adj(ticker):
    DES = read_AGG_DES(ticker)
    cumSum_prob = read_cumSum_prob(ticker)[DES.index[0].date():]
    cumSum = read_cumSum(ticker)[DES.index[0].date():]
    
    DES_adj = 0.50*DES + 0.20*cumSum_prob + 0.30*cumSum
    DES_adj.fillna(0.5, inplace=True)
    
    temp = cumSum * DES_adj
    temp.fillna(method='bfill', inplace=True)
    return DES_adj

def df_date_filter(df, end=None):
    if end:
        df = df[df.index <= end]
    return df

def create_treemap_data(end, item, clip=None):
    close = data.get('price:收盤價')
    close.fillna(method='ffill', inplace=True)
    basic_info = data.get('company_basic_info')
    turnover = data.get('price:成交金額')
    turnover.fillna(method='ffill', inplace=True)
    close_data = df_date_filter(close, end)
    turnover_data = df_date_filter(turnover, end).iloc[1:].sum() / 100000000
    return_ratio = (close_data.iloc[-1] / close_data.iloc[-2]).dropna().replace(np.inf, 0)
    return_ratio = round((return_ratio - 1) * 100, 2)

    concat_list = [close_data.iloc[-1], turnover_data, return_ratio]
    col_names = ['stock_id', 'close', 'turnover', 'return_ratio']
    
    if item not in ["return_ratio", "turnover_ratio", "AI_score"]:
        try:
            custom_item = df_date_filter(data.get(item), end).iloc[-1].fillna(0)
        except Exception as e:
            print('data error, check the data is existed between start and end.')
            print(e)
            return None
        if clip:
            custom_item = custom_item.clip(*clip)
        concat_list.append(custom_item)
        col_names.append(item)

    df = pd.concat(concat_list, axis=1).dropna()
    df = df.reset_index()
    df.columns = col_names

    basic_info_df = basic_info.copy()
    basic_info_df['stock_id_name'] = basic_info_df['stock_id'].astype(str) + basic_info_df['公司簡稱']

    df = df.merge(basic_info_df[['stock_id', 'stock_id_name', '產業類別', '市場別', '實收資本額(元)']], 
                  how='left', on='stock_id')
    df = df.rename(columns={'產業類別': 'category', '市場別': 'market', '實收資本額(元)': 'base'})
    df = df.dropna(thresh=5)

    df['market_value'] = round(df['base'] / 10 * df['close'] / 100000000, 2)
    df['turnover_ratio'] = df['turnover'] / (df['turnover'].sum()) * 100
    df['country'] = 'TW-Stock'
    df.set_index('stock_id', inplace=True)
    # 用 intersection 而非嚴格 loc[tickers]: 若某些 ticker 在 finlab 沒當日資料(例如已下市/當日停牌),
    # 嚴格 .loc 會 KeyError 把整支腳本中斷;改成只取雙方都有的標的。
    available_tickers = [t for t in tickers if t in df.index]
    missing_tickers = sorted(set(tickers) - set(available_tickers))
    if missing_tickers:
        print(f"[create_treemap_data] {len(missing_tickers)} 檔在 finlab 找不到當日資料,已略過: {missing_tickers}")
    df = df.loc[available_tickers, :]
    df.reset_index(inplace=True)
    
    # Add Win_rate and P_L data
    ratio_files = sorted(glob.glob('./Ratios_2023/Ratio_*.csv'))
    path = ratio_files[-1]
    ratio = pd.read_csv(path)
    ratio_pool = ratio.loc[:, ['stock_id', 'win_rate', 'P_L']]
    ratio_pool['stock_id'] = ratio_pool['stock_id'].astype(str)

    # pandas 2.x 已移除 DataFrame.append; 改用一次 concat (也比逐筆 append 快很多)
    df_ratio = pd.concat(
        [ratio_pool[ratio_pool['stock_id'] == t] for t in tickers],
        ignore_index=True,
    )
    df = df.merge(df_ratio, on='stock_id', how='inner')
    
    AI_score = []
    for ticker in df['stock_id']:
        temp = get_DES_adj(ticker).iloc[-1, :].values[0]
        AI_score.append(temp)
    df['AI_score'] = AI_score
    df.replace({'sii': 'TWSE', 'otc': 'OTC'}, inplace=True)
    
    return df

def plot_tw_stock_treemap_enhanced(end=None, area_ind='market_value', item='AI_score', clip=None):
    """Enhanced treemap with dropdown selection and auto-navigation"""
    df = create_treemap_data(end, item, clip)
    if df is None:
        return None
    
    df['custom_item_label'] = round(df[item], 2).astype(float)

    if area_ind not in ["market_value", "turnover", "turnover_ratio", 'win_rate', 'P_L']:
        return None

    color_continuous_midpoint = 0.5

    fig = px.treemap(df,
                     path=['country', 'market', 'category', 'stock_id_name'],
                     values=area_ind,
                     color=item,
                     color_continuous_scale=['green', 'white', 'red'],
                     color_continuous_midpoint=color_continuous_midpoint,
                     custom_data=['custom_item_label', 'close', 'turnover', 'return_ratio', 'win_rate', 'P_L'],
                     title=f'Enhanced TW-Stock Market TreeMap ({end})',
                     width=1600,
                     height=900)

    fig.update_traces(textposition='middle center',
                      textfont_size=16,
                      texttemplate="%{label}<br>%{customdata[0]}",
                      hovertemplate="Price: %{customdata[1]}<br>Return: %{customdata[3]}%<br>Win: %{customdata[4]}<br>P/L: %{customdata[5]}")
                      
    return fig, df

def generate_enhanced_html(end_date):
    """Generate enhanced HTML with dropdown and auto-navigation"""
    end = end_date.strftime("%Y-%m-%d")
    area_ind = "market_value"
    item = "AI_score"
    clip = 1000
    
    print(f"正在生成 {end} 的treemap資料...")
    fig, df = plot_tw_stock_treemap_enhanced(end, area_ind, item, clip)
    
    if fig is None or df is None:
        print("無法生成treemap資料")
        return None
    
    # 創建股票選項和數據
    stock_options = []
    stock_data = {}
    
    for _, row in df.iterrows():
        stock_id = str(row['stock_id'])
        stock_name = row['stock_id_name']
        country = row['country']
        market = row['market']
        category = row['category']
        
        stock_options.append(f'<option value="{stock_id}">{stock_name}</option>')
        
        stock_data[stock_id] = {
            'country': country,
            'market': market,
            'category': category,
            'name': stock_name
        }
    
    # 獲取plotly圖表的HTML
    plot_div = offline.plot(fig, output_type='div', include_plotlyjs=True)
    
    # 創建簡化的HTML內容 - 只保留下拉選單，添加元大金控 logo
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Enhanced TW Stock Treemap - {end}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e5aa8 0%, #2d6bb8 100%);
            min-height: 100vh;
        }}
        

        
        .container {{
            max-width: 1500px;
            margin: 0 auto;
        }}
        
        .controls {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            text-align: center;
        }}
        

        
        .controls h2 {{
            margin: 0 0 20px 0;
            color: #333;
            font-size: 24px;
            font-weight: 600;
        }}
        
        .dropdown-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
        }}
        
        .dropdown-container label {{
            font-weight: 500;
            color: #555;
            font-size: 16px;
        }}
        
        #stockSelect {{
            padding: 12px 15px;
            font-size: 16px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            background-color: white;
            min-width: 400px;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        #stockSelect:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        #stockSelect:hover {{
            border-color: #667eea;
        }}
        
        .info {{
            margin-top: 15px;
            padding: 15px;
            background: linear-gradient(135deg, #1e5aa8 0%, #2d6bb8 100%);
            color: white;
            border-radius: 8px;
            display: none;
            font-weight: 500;
            text-align: left;
        }}
        
        .treemap-container {{
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .status {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border-radius: 25px;
            display: none;
            z-index: 1000;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        /* 高亮點擊的區域 */
        .highlight-click {{
            outline: 3px solid #ff6b6b !important;
            animation: pulse 0.5s ease-in-out;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
    </style>
</head>
<body>

        <div class="controls">

            <h2>🚀 元大自營台股AI智能導航系統</h2>
            <div class="dropdown-container">
                <label for="stockSelect">請輸入代號 / 公司名稱：</label>
                <select id="stockSelect" size="1">
                    <option value="">-- 請選擇股票 --</option>
                    {chr(10).join(stock_options)}
                </select>
            </div>
            <div id="stockInfo" class="info">
                <p id="stockPath"></p>
            </div>
        </div>
        
        <div class="treemap-container">
            {plot_div}
        </div>
    </div>

    <div id="status" class="status">正在導航...</div>

    <script>
        // 股票數據
        const stockData = {json.dumps(stock_data, ensure_ascii=False)};
        
        // 獲取DOM元素
        const stockSelect = document.getElementById('stockSelect');
        const stockInfo = document.getElementById('stockInfo');
        const stockPath = document.getElementById('stockPath');
        
        // 導航狀態
        let currentSelectedStock = null;
        let isNavigating = false;
        let plotlyDiv = null;
        
        // 等待Plotly載入
        function initializePlotly() {{
            return new Promise((resolve) => {{
                const checkPlotly = () => {{
                    plotlyDiv = document.querySelector('.plotly-graph-div');
                    if (plotlyDiv && window.Plotly) {{
                        console.log('✅ Plotly已載入完成');
                        
                        // 添加plotly點擊事件監聽器
                        plotlyDiv.on('plotly_click', function(data) {{
                            console.log('🖱️  Plotly點擊事件觸發:', data);
                        }});
                        
                        resolve();
                    }} else {{
                        setTimeout(checkPlotly, 100);
                    }}
                }};
                checkPlotly();
            }});
        }}
        
        // 監聽下拉選單變化 - 選擇後自動導航
        stockSelect.addEventListener('change', function() {{
            const selectedStock = this.value;
            
            if (selectedStock && stockData[selectedStock] && !isNavigating) {{
                const stock = stockData[selectedStock];
                currentSelectedStock = selectedStock;
                
                stockPath.innerHTML = `
                    <strong>🎯 選中股票：</strong> ${{stock.name}}<br>
                    <strong>📍 導航路徑：</strong> ${{stock.country}} → ${{stock.market}} → ${{stock.category}} → ${{stock.name}}
                `;
                stockInfo.style.display = 'block';
                
                // 選擇後自動開始導航
                setTimeout(() => {{
                    navigateToStock();
                }}, 500);
                
            }} else if (!selectedStock) {{
                currentSelectedStock = null;
                stockInfo.style.display = 'none';
            }}
        }});
        
        // 顯示狀態訊息
        function showStatus(message, duration = 2000) {{
            const status = document.getElementById('status');
            status.textContent = message;
            status.style.display = 'block';
            setTimeout(() => {{
                status.style.display = 'none';
            }}, duration);
        }}
        
        // 強制DOM點擊方法
        function forceClickTreemapElement(targetText) {{
            return new Promise((resolve) => {{
                try {{
                    if (!plotlyDiv) {{
                        console.log('❌ Plotly容器未找到');
                        resolve(false);
                        return;
                    }}
                    
                    console.log(`🎯 強制點擊: "${{targetText}}"`);
                    
                    // 查找所有可能的文字元素
                    const allTextElements = plotlyDiv.querySelectorAll('text, tspan');
                    let targetElement = null;
                    
                    // 優先精確匹配
                    for (let el of allTextElements) {{
                        if (el.textContent && el.textContent.trim() === targetText) {{
                            targetElement = el;
                            break;
                        }}
                    }}
                    
                    // 如果沒有精確匹配，使用包含匹配
                    if (!targetElement) {{
                        for (let el of allTextElements) {{
                            if (el.textContent && el.textContent.includes(targetText)) {{
                                targetElement = el;
                                break;
                            }}
                        }}
                    }}
                    
                    if (targetElement) {{
                        // 找到最接近的可點擊父元素
                        let clickableElement = targetElement.closest('g.slice') || 
                                             targetElement.closest('g') || 
                                             targetElement.closest('rect') ||
                                             targetElement;
                        
                        // 高亮顯示要點擊的元素
                        clickableElement.classList.add('highlight-click');
                        setTimeout(() => {{
                            clickableElement.classList.remove('highlight-click');
                        }}, 500);
                        
                        // 獲取元素的位置信息
                        const rect = clickableElement.getBoundingClientRect();
                        const centerX = rect.left + rect.width / 2;
                        const centerY = rect.top + rect.height / 2;
                        
                        console.log(`📍 點擊位置: (${{centerX}}, ${{centerY}})`);
                        
                        // 創建完整的事件序列來模擬真實點擊
                        const events = [
                            new MouseEvent('mouseenter', {{ bubbles: true, cancelable: true, clientX: centerX, clientY: centerY }}),
                            new MouseEvent('mouseover', {{ bubbles: true, cancelable: true, clientX: centerX, clientY: centerY }}),
                            new MouseEvent('mousedown', {{ bubbles: true, cancelable: true, clientX: centerX, clientY: centerY, button: 0 }}),
                            new MouseEvent('mouseup', {{ bubbles: true, cancelable: true, clientX: centerX, clientY: centerY, button: 0 }}),
                            new MouseEvent('click', {{ bubbles: true, cancelable: true, clientX: centerX, clientY: centerY, button: 0, detail: 1 }})
                        ];
                        
                        // 依序觸發所有事件
                        events.forEach((event, index) => {{
                            setTimeout(() => {{
                                clickableElement.dispatchEvent(event);
                                if (index === events.length - 1) {{
                                    console.log(`✅ 完成點擊事件序列: "${{targetText}}"`);
                                }}
                            }}, index * 10);
                        }});
                        
                        // 同時嘗試觸發Plotly的內建點擊處理
                        if (window.Plotly && plotlyDiv._fullData) {{
                            setTimeout(() => {{
                                try {{
                                    const data = plotlyDiv._fullData[0];
                                    if (data && data.labels) {{
                                        const labelIndex = data.labels.findIndex(label => 
                                            label && (label.toString() === targetText || label.toString().includes(targetText))
                                        );
                                        
                                        if (labelIndex !== -1) {{
                                            const clickData = {{
                                                points: [{{
                                                    pointNumber: labelIndex,
                                                    data: data,
                                                    fullData: data
                                                }}]
                                            }};
                                            
                                            plotlyDiv.emit('plotly_click', clickData);
                                            console.log(`✅ Plotly事件觸發: "${{targetText}}"`);
                                        }}
                                    }}
                                }} catch (e) {{
                                    console.log('⚠️  Plotly事件觸發失敗:', e);
                                }}
                            }}, 100);
                        }}
                        
                        setTimeout(() => resolve(true), 500);
                        
                    }} else {{
                        console.log(`❌ 未找到元素: "${{targetText}}"`);
                        resolve(false);
                    }}
                    
                }} catch (error) {{
                    console.log('❌ 強制點擊失敗:', error);
                    resolve(false);
                }}
            }});
        }}
        
        // 按導航路徑順序執行
        async function executeNavigationPath(navigationPath) {{
            console.log('🚀 開始執行導航路徑:', navigationPath);
            
            for (let i = 0; i < navigationPath.length; i++) {{
                const currentStep = navigationPath[i];
                const stepNumber = i + 1;
                
                showStatus(`📍 第${{stepNumber}}步：點擊 ${{currentStep}}`, 1200);
                console.log(`📍 執行第${{stepNumber}}步: "${{currentStep}}"`);
                
                const success = await forceClickTreemapElement(currentStep);
                
                if (success) {{
                    console.log(`✅ 第${{stepNumber}}步成功`);
                }} else {{
                    console.log(`❌ 第${{stepNumber}}步失敗`);
                }}
                
                // 等待treemap更新
                if (i < navigationPath.length - 1) {{
                    await new Promise(resolve => setTimeout(resolve, 500));
                }}
            }}
            
            console.log('🎯 導航完成');
        }}
        
        // 主導航函數
        function navigateToStock() {{
            const selectedStock = stockSelect.value || currentSelectedStock;
            
            if (!selectedStock || !stockData[selectedStock]) {{
                return;
            }}
            
            if (isNavigating) {{
                return;
            }}
            
            isNavigating = true;
            const stock = stockData[selectedStock];
            
            // 建立導航路徑
            const navigationPath = [
                stock.country,    // TW-Stock
                stock.market,     // TWSE/OTC
                stock.category,   // 產業類別
                stock.name        // 股票名稱
            ];
            
            console.log('🎯 準備導航至:', stock.name);
            console.log('📍 導航路徑:', navigationPath);
            
            // 確保Plotly載入後執行
            initializePlotly().then(() => {{
                showStatus('🚀 開始自動導航...', 1000);
                
                executeNavigationPath(navigationPath).then(() => {{
                    setTimeout(() => {{
                        showStatus('✅ 導航完成！', 500);
                        isNavigating = false;
                    }}, 1000);
                }});
            }});
        }}
        
        // 初始化
        initializePlotly().then(() => {{
            console.log('🚀 Enhanced AI Treemap 已載入');
            console.log('💡 選擇股票後將自動按照路徑導航');
            console.log(`📊 共載入 ${{Object.keys(stockData).length}} 支股票`);
        }});
    </script>
</body>
</html>'''
    
    return html_content

# 主程序
if __name__ == "__main__":
    try:
        end_date = datetime.today().date() - timedelta(days=1)
        
        print("🚀 正在生成增強版AI Treemap...")
        print(f"📅 目標日期: {end_date}")
        
        html_content = generate_enhanced_html(end_date)
        
        if html_content:
            filename = f"./AI_treemap/Enhanced_AI_Treemap_{end_date.strftime('%Y%m%d')}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"✅ 增強版treemap已保存至: {filename}")
            print("\n🎯 簡化界面功能：")
            print("1. 🎪 簡潔界面 - 只保留下拉選單")
            print("2. 🎯 自動導航 - 選擇股票後立即開始導航")
            print("3. 📍 路徑顯示 - 清楚顯示導航路徑")
            print("4. 🖱️  強制點擊 - 完整的鼠標事件序列")
            print("5. 💫 視覺反饋 - 高亮顯示被點擊的區域")
            
        else:
            print("❌ 生成HTML內容失敗")
            
    except Exception as e:
        print(f"❌ 程式執行錯誤: {str(e)}")
        import traceback
        traceback.print_exc()