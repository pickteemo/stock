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

# %%
t1 = ts.pro_bar(ts_code="000001.SZ", adj="qfq", start_date="20230101", limit=1)
t2 = ak.stock_hk_hist(
    symbol="00700",
    period="daily",
    start_date="20240101",
    end_date="22220101",
    adjust="",
)

t3 = pro.fund_daily(
    **{
        "trade_date": "",
        "start_date": "20230101",
        "end_date": "",
        "ts_code": "159941.SZ",
        "limit": "1",
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
print(t1)
print(t2.tail(1))
print(t3)

# %%
stock_basic_df = pro.stock_basic(
    **{
        "ts_code": "",
        "name": "",
        "exchange": "",
        "market": "",
        "is_hs": "",
        "list_status": "",
        "limit": "",
        "offset": "",
    },
    fields=[
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "cnspell",
        "market",
        "list_date",
        "act_name",
        "act_ent_type",
    ],
)

# filter
stock_basic_df = stock_basic_df[
    stock_basic_df["market"].isin(["主板", "创业板", "科创板"])
]
stock_basic_df["list_date"] = pd.to_datetime(
    stock_basic_df["list_date"], format="%Y%m%d"
)
stock_basic_df = stock_basic_df[stock_basic_df["name"].str.contains("ST") == False]
stock_basic_df = stock_basic_df[stock_basic_df["list_date"] < "2024-06-30"]

print(stock_basic_df.shape)


# ggt
ggt_df = pd.read_csv("ggt/ggt_hk.csv")
print(ggt_df.shape)

# etf
etf_df = pd.read_csv("etf/etf.csv")
fetf_df = pd.read_csv("etf/fetf.csv")
print(etf_df.shape)
print(fetf_df.shape)

basic_df = pd.concat([stock_basic_df, ggt_df, etf_df], ignore_index=True)
print(basic_df.shape)


stock_basic_df.to_csv("data/stock_basic_df.csv", index=False)
ggt_df.to_csv("data/ggt_basic_df.csv", index=False)
etf_df.to_csv("data/etf_basic_df.csv", index=False)
fetf_df.to_csv("data/fetf_basic_df.csv", index=False)
basic_df.to_csv("data/basic_df.csv", index=False)

# %%
