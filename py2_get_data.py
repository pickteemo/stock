# %%
import tushare as ts
import pandas as pd
import numpy as np
import os
import time
import tqdm
import talib as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import shutil
import akshare as ak

TOEKN_PATH = os.path.expanduser("./.tushare.token")

with open(TOEKN_PATH, "r") as f:
    token = f.read().strip()
    ts.set_token(token=token)
    pro = ts.pro_api(token=token)

get_A = True
get_HK = True
get_ETF = True


# %%
daily_folder = Path("data/daily")
plt_folder = Path("data/plt")

# %%
stock_basic_df = pd.read_csv("./data/stock_basic_df.csv")
ggt_df = pd.read_csv("./data/ggt_basic_df.csv")
etf_df = pd.read_csv("./data/etf_basic_df.csv")
fetf_df = pd.read_csv("./data/fetf_basic_df.csv")
basic_df = pd.read_csv("./data/basic_df.csv")
code = list(stock_basic_df["ts_code"])

daily_dir = os.path.join("data", "daily")
if not os.path.exists(daily_dir):
    # 目录不存在，创建目录
    os.makedirs(os.path.join(daily_dir))

daily_df = pd.DataFrame()

if get_A:
    for c in tqdm.tqdm(code, desc="Processing"):
        chunk_filename = os.path.join(daily_dir, f"{c}.csv")
        if os.path.exists(chunk_filename):
            continue
        c_df = ts.pro_bar(ts_code=c, adj="qfq", start_date="20230101")
        f_df = pro.daily_basic(
            **{
                "ts_code": c,
                "trade_date": "",
                "start_date": "20230101",
                "end_date": "",
                "limit": "",
                "offset": "",
            },
            fields=[
                "ts_code",
                "trade_date",
                # "close",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
                "limit_status",
            ],
        )
        merged_df = pd.merge(c_df, f_df, on=["ts_code", "trade_date"])
        if merged_df.empty:
            continue
        merged_df.to_csv(chunk_filename, index=False)
        time.sleep(0.1)

# %%

if get_HK:
    for c in tqdm.tqdm(ggt_df["ts_code"], desc="Processing"):
        c_num = c.split(".")[0]
        chunk_filename = os.path.join(daily_dir, f"{c}.csv")
        if os.path.exists(chunk_filename):
            continue
        stock_hk_hist_df = ak.stock_hk_hist(
            symbol=c_num,
            period="daily",
            start_date="20240101",
            end_date="22220101",
            adjust="",
        )
        stock_hk_hist_df["ts_code"] = c
        stock_hk_hist_df["circ_mv"] = 0
        stock_hk_hist_df["日期"] = stock_hk_hist_df["日期"].apply(
            lambda x: x.strftime("%Y%m%d")
        )
        stock_hk_hist_df = stock_hk_hist_df.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "vol",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover_rate",
            }
        )
        if stock_hk_hist_df.empty:
            continue
        stock_hk_hist_df.to_csv(chunk_filename, index=False)
        time.sleep(0.01)

# %%
if get_ETF:
    etf_list = list(set(etf_df["ts_code"]) | set(fetf_df["ts_code"]))
    for c in tqdm.tqdm(etf_list, desc="Processing"):
        chunk_filename = os.path.join(daily_dir, f"{c}.csv")
        if os.path.exists(chunk_filename):
            continue
        df = pro.fund_daily(
            **{
                "trade_date": "",
                "start_date": "20230101",
                "end_date": "",
                "ts_code": c,
                "limit": "",
                "offset": "",
            },
            fields=[
                "ts_code",
                "trade_date",
                "pre_close",
                "open",
                "high",
                "low",
                "close",
                "change",
                "pct_chg",
                "vol",
                "amount",
            ],
        )
        df["circ_mv"] = 0
        if df.empty:
            continue
        df.to_csv(chunk_filename, index=False)
        time.sleep(0.5)

# %%
