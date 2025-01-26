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

# TOEKN_PATH = os.path.expanduser("~/.tushare.token")

# with open(TOEKN_PATH, "r") as f:
#     token = f.read().strip()
#     ts.set_token(token=token)
#     pro = ts.pro_api(token=token)


# %%
stock_basic_df = pd.read_csv("./data/stock_basic_df.csv")
ggt_df = pd.read_csv("./data/ggt_basic_df.csv")
etf_df = pd.read_csv("./data/etf_basic_df.csv")
basic_df = pd.read_csv("./data/basic_df.csv")


def apply_ma(df):
    df = df.sort_values(by="trade_date", ascending=True)

    ma_values = {}
    for i in range(2, 121):
        ma_values[f"ma{i}"] = ta.MA(df["close"].values, timeperiod=i)

    k_ma = 1.0
    if ma_values["ma20"][-1] < ma_values["ma60"][-1]:
        k_ma = k_ma * 0.8

    if ma_values["ma60"][-1] < ma_values["ma120"][-1]:
        k_ma = k_ma * 0.8

    ma_df = pd.DataFrame(ma_values, index=df.index)
    df = pd.concat([df, ma_df], axis=1)

    close_value = df.iloc[-1]["close"]
    ma_columns = [f"ma{i}" for i in range(2, 120)]  # 获取所有均线列名
    greater_than_ma = [close_value > df[f"ma{i}"].iloc[-1] for i in range(2, 121)]
    proportion = sum(greater_than_ma) / len(greater_than_ma)  # 计算比例

    return df, proportion * k_ma


# %%
def apply_supertrend(df, period=10, multiplier=3):
    df = df.sort_values(by="trade_date", ascending=True)
    df = df[["trade_date", "open", "high", "low", "close", "vol"]]
    df = df.rename(columns={"vol": "volume"})
    df = df.set_index("trade_date")
    df.index = pd.to_datetime(df.index, format=r"%Y%m%d")
    df.sort_index(inplace=True)

    # REF: https://www.quantifiedstrategies.com/supertrend-indicator-trading-strategy/
    # REF: https://www.investopedia.com/terms/a/atr.asp
    # # TRANGE(high, low, close) Average True Range
    df["TR"] = ta.TRANGE(df["high"].values, df["low"].values, df["close"].values)
    # df["ATR"] = ta.SMA(df["TR"], period) # or df["TR"].rolling(window=period).mean()
    df["ATR"] = ta.ATR(
        df["high"].values, df["low"].values, df["close"].values, timeperiod=period
    )

    # 计算basic趋势线 (High + Low) / 2 + Multiplier * ATR
    df["basic-ub"] = (df["high"] + df["low"]) / 2 + multiplier * df["ATR"]
    df["basic-lb"] = (df["high"] + df["low"]) / 2 - multiplier * df["ATR"]

    # 计算final趋势线
    df["final-ub"] = 0.00
    df["final-lb"] = 0.00

    for i in range(period, len(df)):
        # 逐行遍历 根据basic计算final
        # 上趋势线只允许相等或者降低 除非上趋势线被突破了
        df["final-ub"].iat[i] = (
            df["basic-ub"].iat[i]
            if df["basic-ub"].iat[i] < df["final-ub"].iat[i - 1]
            or df["close"].iat[i - 1] > df["final-ub"].iat[i - 1]
            else df["final-ub"].iat[i - 1]
        )

        df["final-lb"].iat[i] = (
            df["basic-lb"].iat[i]
            if df["basic-lb"].iat[i] > df["final-lb"].iat[i - 1]
            or df["close"].iat[i - 1] < df["final-lb"].iat[i - 1]
            else df["final-lb"].iat[i - 1]
        )

    df["st"] = 0.0
    for i in range(period, len(df)):
        # 突破上趋势线的时候 st采用下趋势线做支撑
        # 当先前st处于上趋势线 且 close <=上趋势线 st采用上趋势线
        # 当先前st处于上趋势线 且 close > 上趋势线 st采用下趋势线
        # 当先前st处于下趋势线 且 close >=下趋势线 st采用下趋势线
        # 当先前st处于下趋势线 且 close < 下趋势线 st采用上趋势线
        df["st"].iat[i] = (
            df["final-ub"].iat[i]
            if df["st"].iat[i - 1] == df["final-ub"].iat[i - 1]
            and df["close"].iat[i] <= df["final-ub"].iat[i]
            else (
                df["final-lb"].iat[i]
                if df["st"].iat[i - 1] == df["final-ub"].iat[i - 1]
                and df["close"].iat[i] > df["final-ub"].iat[i]
                else (
                    df["final-lb"].iat[i]
                    if df["st"].iat[i - 1] == df["final-lb"].iat[i - 1]
                    and df["close"].iat[i] >= df["final-lb"].iat[i]
                    else (
                        df["final-ub"].iat[i]
                        if df["st"].iat[i - 1] == df["final-lb"].iat[i - 1]
                        and df["close"].iat[i] < df["final-lb"].iat[i]
                        else 0.00
                    )
                )
            )
        )

    df["buy-signal"] = np.nan
    df["sell-signal"] = np.nan

    for i in range(period, len(df)):

        if (
            df["close"].iat[i - 1] < df["st"].iat[i - 1]
            and df["close"].iat[i] > df["st"].iat[i]
        ):
            # 前一天突破supertrend
            df["buy-signal"].iat[i] = df["open"].iat[i]

        if (
            df["close"].iat[i - 1] > df["st"].iat[i - 1]
            and df["close"].iat[i] < df["st"].iat[i]
        ):
            # 跌破supertrend
            df["sell-signal"].iat[i] = df["open"].iat[i]

    return df


def plot_supertrend(df, save_dir):
    df = df.dropna(subset=["basic-ub", "basic-lb"])
    df = df[df.index >= datetime(2023, 8, 1)]
    plt.clf()
    addplots = [
        # mpf.make_addplot(df['basic-ub'], label="basic-ub", linestyle='dashdot', color='grey'),
        # mpf.make_addplot(df['basic-lb'], label="basic-lb", linestyle='dashdot', color='grey'),
        # mpf.make_addplot(df['final-ub'], label="final-ub"),
        # mpf.make_addplot(df['final-lb'], label="final-lb"),
        mpf.make_addplot(df["st"], label="supertrend"),
        mpf.make_addplot(df["buy-signal"], type="scatter", markersize=60, marker="^"),
        mpf.make_addplot(df["sell-signal"], type="scatter", markersize=60, marker="v"),
    ]

    # 创建一个subplot，用于绘制蜡烛图
    fig, axlist = mpf.plot(
        df,
        type="candle",
        figsize=(18, 9),
        style="yahoo",
        volume=True,
        addplot=addplots,  # 增加格外的plot 比如我这里加了bollinger线
        show_nontrading=False,
        returnfig=True,
        savefig=save_dir,
    )

    date = df.index[-1].strftime("%Y-%m-%d")

    # 显示图表
    # mpf.show()


def apply_rsi(df, period=24):
    df = df.sort_values(by="trade_date", ascending=True)
    df["rsi12"] = ta.RSI(df["close"].values, timeperiod=12)
    df["rsi60"] = ta.RSI(df["close"].values, timeperiod=60)
    df["rsi120"] = ta.RSI(df["close"].values, timeperiod=120)
    df["min_rsi"] = df[["rsi12", "rsi60", "rsi120"]].min(axis=1)

    return df


def get_industry_info(df):
    ts_code = df["ts_code"].values[0]
    df = df.sort_values(by="trade_date", ascending=True)

    ma5 = ta.SMA(df["close"].values, timeperiod=5)
    ma20 = ta.SMA(df["close"].values, timeperiod=20)
    ma60 = ta.SMA(df["close"].values, timeperiod=60)
    df["ma5"] = ma5
    df["ma20"] = ma20
    df["ma60"] = ma60

    df = df.tail(1)
    name = basic_df.loc[basic_df["ts_code"] == ts_code, "name"].values[0]
    industry = basic_df.loc[basic_df["ts_code"] == ts_code, "industry"].values[0]
    df["name"] = name
    df["industry"] = industry
    df["act_ent_type"] = basic_df.loc[
        basic_df["ts_code"] == ts_code, "act_ent_type"
    ].values[0]

    df = df[
        [
            "ts_code",
            "trade_date",
            "close",
            "ma5",
            "ma20",
            "ma60",
            "name",
            "industry",
            "act_ent_type",
            "pe",
            "pb",
            "dv_ratio",
            "dv_ttm",
            "circ_mv",
        ]
    ]
    df["dv_ratio"] = df["dv_ratio"].fillna(0)
    df["dv_ttm"] = df["dv_ttm"].fillna(0)
    df["dv"] = df[["dv_ratio", "dv_ttm"]].max(axis=1)

    return df


# %%
# 指定目录路径
daily_data_dir = "./data/daily"
directory = Path(daily_data_dir)
# 列出目录下的所有文件
files = [file.name for file in directory.iterdir() if file.is_file()]
files = [file for file in files if file.endswith(".csv")]


# 先读一个拿日期
df = pd.read_csv(os.path.join(daily_data_dir, files[0]))
trade_date0 = str(df["trade_date"].iloc[0])
plt_work_dir = os.path.join("data", "plt", trade_date0)
# 检查目录是否存在
if not os.path.exists(plt_work_dir):
    # 目录不存在，创建目录
    os.makedirs(os.path.join(plt_work_dir, "buy"))
    os.makedirs(os.path.join(plt_work_dir, "sell"))

up_df = pd.DataFrame()
up_list = []
industry_list = []
industry_df = pd.DataFrame()
pbar = tqdm.tqdm(files)
for f in pbar:
    pbar.set_description(f"Processing {f}")
    file_path = os.path.join(daily_data_dir, f)
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        print(f"Empty file: {f}")
        continue
    ts_code = df["ts_code"].iloc[0]
    name = str(ts_code)
    if ts_code in basic_df["ts_code"].values:
        name = basic_df.loc[basic_df["ts_code"] == ts_code, "name"].values[0]

    # ma
    df, proportion = apply_ma(df)

    new_row = pd.DataFrame(
        {"ts_code": ts_code, "name": name, "proportion": proportion}, index=[0]
    )

    up_df = pd.concat([up_df, new_row])

    # supertrend
    # supertrend_df = apply_supertrend(df)
    # if supertrend_df["buy-signal"].tail(5).notna().any():
    #     # plot_supertrend(supertrend_df, os.path.join(plt_work_dir, "buy", f+name+".png"))
    #     # supertrend_df.to_csv(os.path.join(plt_work_dir, "buy", f+name+".csv"), index=False)
    #     buy_list.append(ts_code)
    # if pd.notna(supertrend_df["sell-signal"].iloc[-1]):
    #     # plot_supertrend(supertrend_df, os.path.join(plt_work_dir, "sell", f+name+".png"))
    #     # supertrend_df.to_csv(os.path.join(plt_work_dir, "sell", f+name+".csv"), index=False)
    #     sell_list.append(ts_code)
    # if supertrend_df.iloc[-1]["close"] > supertrend_df.iloc[-1]["st"]:
    #     up_list.append(ts_code)

    stock_code = list(stock_basic_df["ts_code"])
    if ts_code in stock_code:
        # calculate insdustry
        ind_df = get_industry_info(df)
        ind_df["proportion"] = proportion
        industry_list.append(ind_df)

print("len up:", up_df[up_df["proportion"] > 0.5].shape[0])
industry_df = pd.concat(industry_list)
industry_df.to_csv(os.path.join("data", "industry.csv"), index=False)

up_df.to_csv(os.path.join("data", "up_df.csv"), index=False)
