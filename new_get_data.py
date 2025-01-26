import tushare as ts
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import talib as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import shutil
import akshare as ak

TOEKN_PATH = os.path.expanduser("./.tushare.token")

with open(TOEKN_PATH, "r") as f:
    token = f.read().strip()
    ts.set_token(token=token)
    pro = ts.pro_api(token=token)


def get_daily_df(trade_date):
    for _ in range(3):
        try:
            df = pro.daily(trade_date=trade_date)
        except:
            time.sleep(1)
        else:
            return df
    return None


def get_fund_daily_df(trade_date):
    for _ in range(3):
        try:
            df = pro.fund_daily(trade_date=trade_date)
        except:
            time.sleep(1)
        else:
            return df
    return None

def get_hk_df(code,trade_date):
    for _ in range(3):
        try:
            df = ak.stock_hk_hist(symbol=code, period="daily", start_date=trade_date, end_date=trade_date, adjust="")
        except:
            time.sleep(1)
        else:
            return df
    return None


def get_hk_daily_df(trade_date):
    hk_stock = ["00700", "09988", "01810", "02015", "09868", "09626"]
    df_list = []
    for code in hk_stock:
        stock_hk_hist_df = get_hk_df(code,trade_date) 
        stock_hk_hist_df["ts_code"] = code + ".HK"
        if stock_hk_hist_df is None or "日期" not in stock_hk_hist_df.columns:
            print(f"Failed to get HK data for {trade_date}")
            print(code,trade_date)
            return None
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
        df_list.append(stock_hk_hist_df)
    hk_df = pd.concat(df_list)
    hk_df = hk_df.reset_index(drop=True)
    return hk_df


def get_pro_daily(white_code_list: list):
    date_df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date="20240101",
        end_date="99999999",
        fields="cal_date",
    )

    start_date = "20240101"
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    #end_date = "20250110"
    # if parquet file exists, read from parquet file
    final_df = pd.DataFrame()
    data_frames = []
    exist_date_set = set()
    if os.path.exists(f"./pro_daily.parquet"):
        final_df = pd.read_parquet(f"./pro_daily.parquet")
        exist_date = final_df["trade_date"].tolist()
        exist_date_set = set(exist_date)
        data_frames.append(final_df)

    time_df = pro.trade_cal(
        exchange="SSE",
        is_open="1",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date",
    )
    cal_date = time_df["cal_date"].tolist()

    start_date_obj = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date_obj = datetime.datetime.strptime(end_date, "%Y%m%d")
    date_range = (end_date_obj - start_date_obj).days + 1

    current_date = start_date_obj
    
    progress_bar = tqdm(total=date_range)

    for day in range(date_range):
        progress_bar.update(1)
        progress_bar.set_description(f"Processing {current_date.strftime('%Y%m%d')}")
        trade_date = current_date.strftime("%Y%m%d")
        if trade_date not in cal_date or trade_date in exist_date_set:
            current_date = start_date_obj + datetime.timedelta(days=day)
            continue
        df = get_daily_df(trade_date)
        fund_df = get_fund_daily_df(trade_date)
        hk_df = get_hk_daily_df(trade_date)
        if df is None or fund_df is None:
            print(f"Failed to get data for {trade_date}")
        else:
            data_frames.append(df)
            data_frames.append(fund_df)
        if hk_df is None:
            print(f"Failed to get HK data for {trade_date}")
        else:
            data_frames.append(hk_df)
        # break
        current_date = start_date_obj + datetime.timedelta(days=day)

    final_df = pd.concat(data_frames, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["ts_code", "trade_date"])
    print(final_df.shape)
    final_df = final_df[final_df["ts_code"].isin(white_code_list)]
    print("filter: ", final_df.shape)
    print(final_df.tail())
    final_df.sort_values(by="trade_date", inplace=True)
    final_df.to_parquet(f"./pro_daily.parquet")


def get_pro_daily_basic():
    pro_daily_df = pd.read_parquet(f"./pro_daily.parquet")
    final_date = pro_daily_df["trade_date"].max()
    if os.path.exists(f"./pro_daily_basic.parquet"):
        daily_basic_df = pd.read_parquet(f"./pro_daily_basic.parquet")
        date = daily_basic_df["trade_date"].max()
        if final_date == date:
            print(f"Already get data for {final_date}")
            return
    daily_basic_df = pro.daily_basic(start_date=final_date, end_date="99999999")
    print(daily_basic_df.head())
    daily_basic_df.to_parquet(f"./pro_daily_basic.parquet")


def calculate_indicator(group, daily_indicator_dict):
    group = group.sort_values(by="trade_date")
    trade_date = group["trade_date"].max()
    ts_code = group["ts_code"].iloc[0]
    ma_values = {}
    ma_values = {
        f"ma{i}": ta.MA(group["close"].values, timeperiod=i) for i in range(2, 181)
    }
    k_ma = 1.0
    if ma_values["ma20"][-1] < ma_values["ma60"][-1]:
        k_ma = k_ma * 0.8
    if ma_values["ma60"][-1] < ma_values["ma120"][-1]:
        k_ma = k_ma * 0.8
    ma_df = pd.DataFrame(ma_values, index=group.index)
    close_value = group.iloc[-1]["close"]
    greater_than_ma = (close_value > ma_df[f"ma{i}"].iloc[-1] for i in range(2, 181))
    proportion = sum(greater_than_ma) / len(ma_values)
    daily_indicator_dict["ts_code"].append(ts_code)
    daily_indicator_dict["trade_date"].append(trade_date)
    daily_indicator_dict["ma_percent"].append(proportion * k_ma)
    daily_indicator_dict["ma60"].append(ma_values["ma60"][-1])


def get_pro_daily_indicator():
    daily_indicator_dict = {
        "ts_code": [],
        "trade_date": [],
        "ma_percent": [],
        "ma60": [],
    }

    pro_daily_df = pd.read_parquet(f"./pro_daily.parquet")
    final_date = pro_daily_df["trade_date"].max()
    if os.path.exists(f"./pro_daily_indicator.parquet"):
        daily_indicator_df = pd.read_parquet(f"./pro_daily_indicator.parquet")
        date = daily_indicator_df["trade_date"].max()
        if final_date == date:
            print(f"Already get data for {final_date}")
            return

    group_count = len(pro_daily_df.groupby("ts_code"))
    for _, group in tqdm(pro_daily_df.groupby("ts_code"), total=group_count):
        calculate_indicator(group, daily_indicator_dict)

    daily_indicator_df = pd.DataFrame(daily_indicator_dict)
    print(daily_indicator_df.head())
    daily_indicator_df.to_parquet(f"./pro_daily_indicator.parquet")


def get_white_list():
    stock_basic_df = pro.stock_basic()
    # filter
    stock_basic_df = stock_basic_df[
        stock_basic_df["market"].isin(["主板", "创业板", "科创板"])
    ]
    stock_basic_df["list_date"] = pd.to_datetime(
        stock_basic_df["list_date"], format="%Y%m%d"
    )
    stock_basic_df = stock_basic_df[stock_basic_df["list_date"] < "2024-06-30"]

    etf_df = pd.read_csv("etf/etf.csv")

    hk_df = [
        "00700.HK",
        "09988.HK",
        "01810.HK",
        "02015.HK",
        "09868.HK",
        "09626.HK",
    ]

    white_code_list = stock_basic_df["ts_code"].tolist() + etf_df["ts_code"].tolist() + hk_df
    print("white: ", len(white_code_list))
    print(stock_basic_df.head())
    stock_basic_df.to_parquet("./pro_stock_basic.parquet")
    return white_code_list


white_code_list = get_white_list()
get_pro_daily(white_code_list)
get_pro_daily_basic()
get_pro_daily_indicator()
