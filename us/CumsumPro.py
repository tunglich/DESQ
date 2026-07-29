import pandas as pd
import numpy as np
import warnings
import pyodbc
import glob
import pickle
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import date

today = date.today()
start_date, end_date = '2000-01-01', today

files = glob.glob(f"D:/DES_model_2023/*")
tickers = [x[-19:-15] for x in files]

from prob_cusum.prob_cusum import CusumDetector

# Set different start_date and end_date and warmup level = 6 by default
t_warmup = 12
out_dir = 'D:/cumSum_prob_'+str(t_warmup)
today = today.strftime('%Y-%m-%d')
start_date, end_date = '2000-01-01', today

for ticker in tickers:
    try:
        df = get_data(ticker=ticker, start=start_date, end=end_date)
        y = df['Close']
                                                                           # <====================================== SET Warmup = 6 default
        detector = CusumDetector(t_warmup=t_warmup)
        outs = np.array([detector.predict_next(y[i]) for i in range(y.shape[0])])

        cps = np.where(outs[:, 1])[0]
        probs = outs[:, 0]
        probs_ = pd.Series(probs, index=df.index)
        #probs_ = (probs_ + 1) /2
        #probs_.to_csv(f'{out_dir}/cumsum_prob_{ticker}.csv')
        probs_.to_csv(f'{out_dir}/cusum_{ticker}.csv')
    except:
        print(ticker)
        continue