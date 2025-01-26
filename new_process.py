import tushare as ts
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import talib as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import shutil
from tabulate import tabulate
import math

# pd.set_option("display.float_format", "{:.2f}".format)  # 显示两位小数
pd.set_option("display.colheader_justify", "left")  # 列头左对齐

TOEKN_PATH = os.path.expanduser("./.tushare.token")

with open(TOEKN_PATH, "r") as f:
    token = f.read().strip()
    ts.set_token(token=token)
    pro = ts.pro_api(token=token)

pro_stock_basic_df = pd.read_parquet("./pro_stock_basic.parquet")
pro_daily_df = pd.read_parquet("./pro_daily.parquet")
pro_daily_basic_df = pd.read_parquet("./pro_daily_basic.parquet")
pro_daily_indicator_df = pd.read_parquet("./pro_daily_indicator.parquet")
etf_df = pd.read_csv("etf/etf.csv")


def chnname(code):
    name = ""
    if code in pro_stock_basic_df["ts_code"].values:
        name = pro_stock_basic_df.loc[
            pro_stock_basic_df["ts_code"] == code, "name"
        ].values[0]
    elif code in etf_df["ts_code"].values:
        name = etf_df.loc[etf_df["ts_code"] == code, "name"].values[0]
    return name


def apply_score(df, period=25):
    df = df.sort_values(by="trade_date", ascending=True, inplace=True)
    df = df.tail(period)
    y = np.log(df.close)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    annualized_returns = math.pow(math.exp(slope), 250) - 1
    r_squared = 1 - (
        sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1))
    )
    score = annualized_returns * r_squared
    return score, annualized_returns, r_squared


def calc_net(df, single_pos):
    ATR_period = 14
    min_bid = 100
    min_grid = 0.003
    ATR_N = 4
    min_grid_min_profit = 10
    ts_code = df["ts_code"].iloc[-1]

    df = df.sort_values(by="trade_date", ascending=True)
    df.loc[:, "atr"] = ta.NATR(
        df["high"], df["low"], df["close"], timeperiod=ATR_period
    )
    atr = df["atr"].iloc[-1]
    close = df["close"].iloc[-1]

    high_handling_fee_list = [
        "518880.SH",
        "162411.SZ",
        "501018.SH",
        "00700.HK",
        "09988.HK",
        "01810.HK",
        "02015.HK",
        "09868.HK",
        "09626.HK",
    ]
    if ts_code in high_handling_fee_list:
        return 1, 1, 1, 1, 1, 1, close

    bid_num = math.ceil(min_bid / close / 100) * 100
    # bid
    bid_price = bid_num * close
    bid_percent = bid_num * close * (ATR_N * atr / 100) / single_pos
    bid_profit = bid_price * bid_percent

    # check grid
    bid_grid = close * bid_percent
    if bid_grid < min_grid:
        bid_grid = min_grid
        bid_percent = bid_grid / close
        bid_price = bid_percent * single_pos / (ATR_N * atr / 100)
        bid_num = math.ceil(bid_price / close / 100) * 100
        bid_price = bid_num * close
        bid_percent = bid_num * close * (ATR_N * atr / 100) / single_pos
        bid_grid = round(close * bid_percent, 3)
        bid_profit = bid_price * bid_percent

    # check profit
    if bid_profit < min_grid_min_profit:
        bid_profit = min_grid_min_profit
        bid_price = math.pow((bid_profit * single_pos / (ATR_N * atr / 100)), 0.5)
        bid_percent = bid_price * (ATR_N * atr / 100) / single_pos
        bid_num = math.ceil(bid_price / close / 100) * 100
        bid_price = bid_num * close
        bid_percent = bid_num * close * (ATR_N * atr / 100) / single_pos
        bid_grid = round(close * bid_percent, 3)
        bid_profit = bid_price * bid_percent

    bid_grid = round(bid_grid, 3) + 0.002
    bid_percent = round(bid_percent * 100, 1)
    atr = round(atr, 2)
    return bid_num, bid_grid, bid_price, bid_percent, bid_profit, atr, close

hold_df = pd.read_csv("table.csv", dtype={"证券代码": str})
hold_df["ts_code"] = hold_df.apply(
    lambda row: (
        str(row["证券代码"]) + ".SZ"
        if row["交易市场"] == "深Ａ"
        else str(row["证券代码"]) + ".SH"
    ),
    axis=1,
)
hold_df.rename(columns={"实际数量": "num"}, inplace=True)
hold_df.rename(columns={"成本价": "price"}, inplace=True)
hold_df.set_index("ts_code", inplace=True)


def apply_info(df):
    df["state"] = ""
    for ts_code in df.index:
        if ts_code in hold_df.index:
            df.loc[ts_code, "state"] = "hold"
            df.loc[ts_code, "num"] = hold_df.loc[ts_code, "num"]
            if "close" in df.columns:
                df.loc[ts_code, "per"] = round(
                    (df.loc[ts_code, "close"] - hold_df.loc[ts_code, "price"])
                    / hold_df.loc[ts_code, "price"]
                    * 100,
                    1,
                )
        else:
            df.loc[ts_code, "state"] = " "
            df.loc[ts_code, "num"] = 0


pb_df = pro_stock_basic_df.set_index("ts_code")[["name", "industry"]]
apply_info(pb_df)
ind1 = pro_daily_basic_df.set_index("ts_code")[
    ["trade_date", "close", "pe_ttm", "pb", "dv_ttm", "circ_mv"]
]
pb_df = pb_df.join(ind1)
ind2 = pro_daily_indicator_df.set_index("ts_code")[["ma_percent", "ma60"]]
pb_df = pb_df.join(ind2)

pb_df["circ_mv"] = pb_df["circ_mv"] / 10000
pb_df["rank"] = pb_df["circ_mv"].rank(method="average", ascending=False)
pb_df["rank_mv"] = round(pb_df["rank"] / len(pb_df) * 100, 2)
pb_df["rank"] = pb_df["pb"].rank(method="average", ascending=False)
pb_df["rank_pb"] = round(pb_df["rank"] / len(pb_df) * 100, 2)
pb_df["rank_mvpb"] = np.maximum(pb_df["rank_mv"], pb_df["rank_pb"])

pb_df["roe"] = pb_df["pb"] / pb_df["pe_ttm"]
pb_df["rank"] = pb_df["roe"].rank(method="average", ascending=True)
pb_df["rank_roe"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["rank"] = pb_df["dv_ttm"].rank(method="average", ascending=True)
pb_df["rank_dv"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["rank"] = pb_df["ma_percent"].rank(method="average", ascending=True)
pb_df["rank_pro"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["c2ma60"] = pb_df["close"] / pb_df["ma60"]
pb_df["rank"] = pb_df["c2ma60"].rank(method="average", ascending=False)
pb_df["rank_pri"] = round(pb_df["rank"] / len(pb_df) * 100, 2)


pb_df["rank"] = (
    pb_df["rank_mvpb"]
    + pb_df["rank_dv"]
    + pb_df["rank_roe"]
    + pb_df["rank_pro"]
    + pb_df["rank_pri"]
) / 5
pb_df["rank_rank"] = pb_df["rank"].rank(method="average", ascending=False)

pb_df = pb_df.sort_values(by="rank", ascending=False)


print(
    tabulate(
        pb_df.head(50)[
            [
                "name",
                "rank",
                "rank_rank",
                "state",
                "num",
                "industry",
                "circ_mv",
                "pe_ttm",
                "pb",
                "roe",
                "dv_ttm",
                "ma_percent",
            ]
        ],
        headers="keys",
    )
)
data = {'name': [], 'ts_code': [], 'weight': [], 'base': []}

def append_data(name, ts_code, weight, base):
    data["name"].append(name)
    data["ts_code"].append(ts_code)
    data["weight"].append(weight)
    data["base"].append(base)
    
# us40
append_data("纳指100", "159501.SZ", 20,"M")
append_data("标普500", "513650.SH", 10,"M")
append_data("道琼斯", "513400.SH", 5,"M")

append_data("美国50", "513850.SH", 1,"M")
append_data("纳指科技", "159509.SZ", 1,"M")
append_data("纳指生物科技", "513290.SH", 1,"M")
append_data("标普生物科技", "159502.SZ", 1,"M")
#append_data("标普消费", "159529.SZ", 1,"M")###

append_data("标普油气", "159518.SZ", 1,"M")
append_data("法国", "513080.SH", 1,"M")
#append_data("德国", "159561.SZ", 1,"M")###
#append_data("沙特", "520830.SH", 1,"M")###
append_data("日经", "513880.SH", 1,"M")

append_data("东证", "513800.SH", 1,"M")
append_data("中韩半导体", "513310.SH", 1,"M")
append_data("东南亚科技", "513730.SH", 1,"M")
#append_data("亚太精选", "159687.SZ", 1,"M")###
#append_data("新兴亚洲", "520580.SH", 1,"M")###

# 20
append_data("黄金", "518880.SH", 10,"C")
append_data("油气", "162411.SZ", 2,"C")
append_data("原油", "501018.SH", 2,"C")
append_data("豆粕", "159985.SZ", 2,"C")
append_data("有色", "159980.SZ", 2,"C")
append_data("能源化工", "159981.SZ", 2,"C")


# 25
append_data("恒生", "159920.SZ", 1,"A")
append_data("恒科", "513130.SH", 1,"A")
append_data("中概互联", "159605.SZ", 1,"A")
append_data("恒生红利低波", "159545.SZ", 1,"A")
append_data("港股红利", "513630.SH", 1,"A")

append_data("恒生高股息", "513690.SH", 1,"A")
append_data("香港券商", "513090.SH", 1,"A")
append_data("中证红利", "515180.SH", 1,"A")
append_data("红利低波", "563020.SH", 1,"A")
append_data("红利低波100", "159307.SZ", 1,"A")

append_data("红利质量", "159758.SZ", 1,"A")
append_data("A500", "159338.SZ", 1,"A")
append_data("中证A50", "159593.SZ", 1,"A")
append_data("沪深300", "510300.SH", 1,"A")
append_data("中证500", "510500.SH", 1,"A")

append_data("中证1000", "512100.SH", 1,"A")
append_data("中证2000", "159531.SZ", 1,"A")
append_data("创业板", "159915.SZ", 1,"A")
append_data("科创50", "588000.SH", 1,"A")
append_data("银行", "516310.SH", 1,"A")

append_data("结构调整", "512950.SH", 1,"A")
append_data("商品", "510170.SH", 1,"A")
append_data("券商", "159842.SZ", 1,"A")
# append_data("稀土", "159715.SZ", 1,"A")
# append_data("云计算", "516510.SH", 1,"A")

# append_data("科技", "159807.SZ", 1,"A")
# append_data("光伏", "516290.SH", 1,"A")
# append_data("碳中和", "516070.SH", 1,"A")


# stock 18->15
append_data("长江电力", "600900.SH", 1,"A")
append_data("中远海控", "601919.SH", 1,"A")
append_data("京沪高铁", "601816.SH", 1,"A")
append_data("招商银行", "600036.SH", 1,"A")
# append_data("腾讯", "00700.HK", 1,"A")

# append_data("阿里巴巴", "09988.HK", 1,"A")
# append_data("小米", "01810.HK", 1,"A")
# append_data("理想", "02015.HK", 1,"A")
# append_data("小鹏", "09868.HK", 1,"A")
# append_data("哔哩哔哩", "09626.HK", 1,"A")

append_data("伟隆股份", "002871.SZ", 1,"A")
append_data("民爆光电", "301362.SZ", 1,"A")
append_data("陕建股份", "600248.SH", 1,"A")
append_data("江苏银行", "600919.SH", 1,"A")
append_data("杭州银行", "600926.SH", 1,"A")

append_data("南京银行", "601009.SH", 1,"A")
append_data("成都银行", "601838.SH", 1,"A")
append_data("紫燕食品", "603057.SH", 1,"A")




m = 830000
df = pd.DataFrame(data)


df.set_index("ts_code", inplace=True)
apply_info(df)

ind2 = pro_daily_indicator_df.set_index("ts_code")[["ma_percent"]]
df = df.join(ind2)


def piecewise_linear_function(x):
    if x < 0:
        return 0.5
    elif x > 1:
        return 2
    elif x <= 0.5:
        k1 = (1 - 0.5) / (0.5 - 0)
        return 0.5 + k1 * x
    elif 0.5 < x:
        k2 = (2 - 1) / (1 - 0.5)
        return 1 + k2 * (x - 0.5)


df["k"] = df["ma_percent"].apply(piecewise_linear_function)
df["k_weight"] = df["weight"] * df["k"]
df["weight"] = df["k_weight"] / df["k_weight"].sum() * 95
df["value"] = round(df["weight"] * m * 0.01, 0)
print(df["weight"].sum())
#df = df.sort_values(by="weight", ascending=False)
df = df.sort_index()
sum = round(df["value"].sum(), 2)
res = m - sum
print(df["value"].sum(), df["value"].sum() / m)
# print(
#     tabulate(
#         df,
#         headers="keys",
#     )
# )


df["ma_percent"] = round(df["ma_percent"] * 100, 2)
df = df.join(pb_df[["rank", "rank_rank"]])


for i in df.index:
    history_df = pro_daily_df[pro_daily_df["ts_code"] == i]
    if len(history_df) == 0:
        print(i, "no history")
        continue
    bid_num, bid_grid, bid_price, bid_percent, bid_profit, atr, close = calc_net(
        history_df, df.loc[i, "value"]
    )
    df.loc[i, "name"] = chnname(i)
    df.loc[i, "diff"] = df.loc[i, "value"] - df.loc[i, "num"] * close
    if abs(df.loc[i, "diff"]) < close * 100 or abs(df.loc[i, "diff"]) < 1000:
        df.loc[i, "diff"] = 1
        df.loc[i, "diff_num"] = 0
    else:
        df.loc[i, "diff_num"] = (
            round(df.loc[i, "diff"] / close / 100, 0) * 100
        )

    df.loc[i, "bid_num"] = bid_num
    df.loc[i, "bid_grid"] = bid_grid
    df.loc[i, "bid_price"] = bid_price
    df.loc[i, "bid_percent"] = bid_percent
    df.loc[i, "bid_profit"] = bid_profit
    df.loc[i, "atr"] = atr
    df.loc[i, "close"] = close
    df.loc[i, "position"] = df.loc[i, "num"] * close
    df.loc[i, "aim_num"] = round(df.loc[i, "value"] / close / 100, 0) * 100

print(df.shape)
print(
    tabulate(
        df[
            [
                "name",
                "bid_grid",
                "bid_num",
                "diff",
                "diff_num",
                # "aim_num",
                "num",
                "value",
                "position",
                "rank",
                "rank_rank",
                "bid_percent",
                "bid_profit",
                "ma_percent",
                "atr"
            ]
        ],
        headers="keys",
        
    )
)

max_index = df["diff"].idxmax()
min_index = df["diff"].idxmin()
bs_row = df.loc[[max_index, min_index]]
print("b,s:")
print(
    tabulate(
        bs_row[
            [
                "name",
                "bid_grid",
                "bid_num",
                "diff",
                "diff_num",
                # "aim_num",
                "num",
                "value",
                "position",
                "rank",
                "rank_rank",
                "bid_percent",
                "bid_profit",
                "ma_percent",
            ]
        ],
        headers="keys",
    )
)


print("xueqiu:")
xueqiu_df = df.sort_values(by="weight", ascending=False).head(10)
xueqiu_df["weight"] = xueqiu_df["weight"] / xueqiu_df["weight"].sum() * 100
xueqiu_df["weight"] = round(xueqiu_df["weight"], 0)
print(xueqiu_df[["name","weight"]])