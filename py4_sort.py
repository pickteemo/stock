# %%
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

up_threshold = 0.7


pd.set_option("display.float_format", "{:.2f}".format)  # 显示两位小数
pd.set_option("display.colheader_justify", "left")  # 列头左对齐

# %%
stock_basic_df = pd.read_csv("./data/stock_basic_df.csv")
ggt_df = pd.read_csv("./data/ggt_basic_df.csv")
etf_df = pd.read_csv("./data/etf_basic_df.csv")
fetf_df = pd.read_csv("./data/fetf_basic_df.csv")
basic_df = pd.read_csv("./data/basic_df.csv")


def chnname(code):
    name = ""
    if code in basic_df["ts_code"].values:
        name = basic_df.loc[basic_df["ts_code"] == code, "name"].values[0]
    elif code in fetf_df["ts_code"].values:
        name = fetf_df.loc[fetf_df["ts_code"] == code, "name"].values[0]
    return name


def chnindustry(code):
    industry = ""
    if code in basic_df["ts_code"].values:
        industry = basic_df.loc[basic_df["ts_code"] == code, "industry"].values[0]
    return industry


# %%
industry_df = pd.read_csv("./data/industry.csv")
up_df = pd.read_csv("./data/up_df.csv")
# buy_list = pd.read_csv("./data/buy_list.csv")["Column1"].tolist()
# up_list = pd.read_csv("./data/up_list.csv")["Column1"].tolist()
daily_data_dir = "./data/daily"


def apply_score(df, period=25):
    df.sort_values(by="trade_date", ascending=True, inplace=True)
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
    min_bid = 500
    min_grid = 0.005
    ATR_N = 5
    min_grid_min_profit = 30

    df.sort_values(by="trade_date", ascending=True, inplace=True)
    df["atr"] = ta.NATR(df["high"], df["low"], df["close"], timeperiod=ATR_period)
    atr = df["atr"].iloc[-1]
    close = df["close"].iloc[-1]

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

    bid_grid = round(bid_grid, 3)  # 买1卖1补偿0.002
    bid_percent = round(bid_percent * 100, 1)
    atr = round(atr, 2)
    return bid_num, bid_grid, bid_price, bid_percent, bid_profit, atr, close


# %%
def apply_bbands(df, period=20):
    df.sort_values(by="trade_date", ascending=True, inplace=True)
    upper, middle, lower = ta.BBANDS(
        df["close"], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
    )
    df["rel"] = (df["close"] - lower) / (upper - lower)
    return df


def apply_rsi(df, period=24):
    df = df.sort_values(by="trade_date", ascending=True)
    df["rsi12"] = ta.RSI(df["close"].values, timeperiod=12)
    df["rsi60"] = ta.RSI(df["close"].values, timeperiod=60)
    df["rsi120"] = ta.RSI(df["close"].values, timeperiod=120)
    df["min_rsi"] = df[["rsi12", "rsi60", "rsi120"]].min(axis=1)

    return df


# %%
# 计算每个组的比例
close_greater_than_ma60 = industry_df["close"] > industry_df["ma60"]
close_greater_than_ma20 = industry_df["close"] > industry_df["ma20"]
close_greater_than_ma5 = industry_df["close"] > industry_df["ma5"]

industry_proportion_ma60 = (
    close_greater_than_ma60.groupby(industry_df["industry"]).mean() * 100
)
industry_proportion_ma20 = (
    close_greater_than_ma20.groupby(industry_df["industry"]).mean() * 100
)
industry_proportion_ma5 = (
    close_greater_than_ma5.groupby(industry_df["industry"]).mean() * 100
)

# 创建包含比例的 DataFrame
industry_proportion = pd.DataFrame(
    {
        "MA5 Proportion": round(industry_proportion_ma5, 2),
        "MA20 Proportion": round(industry_proportion_ma20, 2),
        "MA60 Proportion": round(industry_proportion_ma60, 2),
    }
)
industry_proportion.sort_values(by="MA20 Proportion", ascending=False, inplace=True)

# 计算每个组的行数
industry_counts = (
    close_greater_than_ma5.groupby(industry_df["industry"])
    .size()
    .reset_index(name="MA5 Count")
)

# 将比例和行数合并到一个 DataFrame 中
industry_proportion_with_counts = pd.merge(
    industry_proportion, industry_counts, on="industry"
)

# 设置显示选项以显示所有行
pd.set_option("display.max_rows", None)

# 输出结果
print(tabulate(industry_proportion_with_counts, headers="keys"))

# %%
industry_list = ["银行", "普钢", "煤炭开采", "小金属", "焦炭加工", "火力发电"]
print(
    tabulate(
        industry_proportion_with_counts[
            industry_proportion_with_counts["industry"].isin(industry_list)
        ],
        headers="keys",
    )
)

# %%
# hold
hold_dict = {"ts_code": [], "num": [], "price": []}


def append_hold(ts_code, num, price):
    hold_dict["ts_code"].append(ts_code)
    hold_dict["num"].append(num)
    hold_dict["price"].append(price)


append_hold("159307.SZ", 15200, 1.033)  # 100红利
append_hold("159338.SZ", 8300, 1.479)  # 中证 A500
append_hold("159501.SZ", 60800, 1.479)  # 纳指100
append_hold("159502.SZ", 7500, 0.996)  # 标普生物科技
append_hold("159509.SZ", 4700, 2.005)  # 纳指科技
append_hold("159518.SZ", 6900, 1.098)  # 标普油气
append_hold("159545.SZ", 12900, 1.156)  # 恒生红利低波
append_hold("159593.SZ", 6900, 1.086)  # A50
append_hold("159605.SZ", 17900, 0.836)  # 中概互联
append_hold("159758.SZ", 17100, 0.876)  # 红利50
append_hold("159915.SZ", 3800, 1.949)  # 创业板
append_hold("159920.SZ", 12700, 1.172)  # 恒生
append_hold("159980.SZ", 4500, 1.66)  # 有色
append_hold("159981.SZ", 5100, 1.478)  # 能源化工
append_hold("159985.SZ", 4100, 1.843)  # 豆粕
append_hold("161226.SZ", 8100, 0.916)  #  白银
append_hold("162411.SZ", 18700, 0.801)  # 华宝油气
append_hold("501018.SH", 5700, 1.015)  # 南方原油
append_hold("501300.SH", 7800, 0.953)  # 美元债
append_hold("510170.SH", 8700, 0.858)  # 商品
append_hold("510300.SH", 2000, 3.802)  # 沪深 300
append_hold("510500.SH", 1400, 5.459)  # 中证 500
append_hold("511020.SH", 200, 118.117)  # 活跃国债
append_hold("511090.SH", 200, 125.986)  # 30年国债
append_hold("511520.SH", 200, 112.58)  # 政金债
append_hold("512950.SH", 11600, 1.286)  # 央企改革
append_hold("513080.SH", 4900, 1.530)  # 法国 ETF
append_hold("513130.SH", 26500, 0.565)  # 恒生科技
append_hold("513290.SH", 6500, 1.16)  # 纳指生物科技
append_hold("513310.SH", 5200, 1.488)  # 中韩半导体
append_hold("513400.SH", 13100, 1.144)  # 道琼斯
append_hold("513630.SH", 12400, 1.299)  # 香港红利
append_hold("513650.SH", 37800, 1.586)  # 标普500
append_hold("513690.SH", 17500, 0.853)  # 恒生股息
append_hold("513730.SH", 5600, 1.345)  # 东南亚科技
append_hold("513800.SH", 5500, 1.368)  # 东证
append_hold("513850.SH", 5300, 1.843)  # 美国50
append_hold("513880.SH", 5700, 1.318)  #  日经 225
append_hold("515180.SH", 11600, 1.293)  # 100 红利
append_hold("516310.SH", 13100, 1.160)  # 银行
append_hold("518880.SH", 12200, 6.137)  # 黄金
append_hold("563020.SH", 13700, 1.147)  # 低波红利
append_hold("588000.SH", 7400, 1.002)  # 科创 50
append_hold("600036.SH", 300, 38.84)  #  招商银行
append_hold("600919.SH", 1100, 9.44)  #  江苏银行
append_hold("600926.SH", 700, 14.232)  # 杭州银行
append_hold("601009.SH", 1000, 10.41)  #  南京银行
append_hold("601838.SH", 600, 16.80)  #   成都银行


hold_df = pd.DataFrame(hold_dict)
hold_df.set_index("ts_code", inplace=True)


def apply_info(df):
    df["state"] = ""
    df["per"] = ""
    for ts_code in df.index:
        if ts_code in hold_df.index:
            df.loc[ts_code, "state"] = "hold"
            daily_df = pd.read_csv(os.path.join(daily_data_dir, f"{ts_code}.csv"))
            daily_df = daily_df.sort_values(by="trade_date", ascending=True)
            close = daily_df.iloc[-1]["close"]
            df.loc[ts_code, "per"] = round(
                (close - hold_df.loc[ts_code, "price"])
                / hold_df.loc[ts_code, "price"]
                * 100,
                1,
            )


# # # %%
# mini_df = industry_df[industry_df["proportion"] > up_threshold]


# mini_df = mini_df.sort_values(by="circ_mv", ascending=True)
# mini_df.loc[:, "circ_mv"] = round(mini_df["circ_mv"] / 10000, 2)
# mini_df.loc[:, "dv_ttm"] = round(mini_df["dv_ttm"], 2)
# mini_df.loc[:, "proportion"] = round(mini_df["proportion"] * 100, 2)

# apply_info(mini_df)
# print(len(mini_df))

# mini_df.to_csv("data/mini_df.csv")

# print(
#     tabulate(
#         mini_df[
#             [
#                 "ts_code",
#                 "name",
#                 "industry",
#                 "act_ent_type",
#                 "state",
#                 "per",
#                 "circ_mv",
#                 "dv_ttm",
#                 "proportion",
#             ]
#         ].head(100),
#         headers="keys",
#     )
# )


# # # %%
# pb_df = industry_df[industry_df["proportion"] > up_threshold]
pb_df = industry_df.copy()
pb_df.set_index("ts_code", inplace=True)
pb_df.loc[:, "circ_mv"] = round(pb_df["circ_mv"] / 10000, 2)


pb_df["rank"] = pb_df["circ_mv"].rank(method="average", ascending=False)
pb_df["rank_mv"] = round(pb_df["rank"] / len(pb_df) * 100, 2)
pb_df["rank"] = pb_df["pb"].rank(method="average", ascending=False)
pb_df["rank_pb"] = round(pb_df["rank"] / len(pb_df) * 100, 2)
pb_df["rank_mvpb"] = np.maximum(pb_df["rank_mv"], pb_df["rank_pb"])

pb_df["roe"] = pb_df["pb"] / pb_df["pe"]
pb_df["rank"] = pb_df["roe"].rank(method="average", ascending=True)
pb_df["rank_roe"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["rank"] = pb_df["dv_ttm"].rank(method="average", ascending=True)
pb_df["rank_dv"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["rank"] = pb_df["proportion"].rank(method="average", ascending=True)
pb_df["rank_pro"] = round(pb_df["rank"] / len(pb_df) * 100, 2)

pb_df["rank"] = (
    pb_df["rank_mvpb"] + pb_df["rank_dv"] + pb_df["rank_roe"] + pb_df["rank_pro"]
) / 4

pb_df = pb_df.sort_values(by="rank", ascending=False)
pb_df.loc[:, "pb"] = round(pb_df["pb"], 2)
pb_df.loc[:, "dv_ttm"] = round(pb_df["dv_ttm"], 2)
pb_df.loc[:, "roe"] = round(pb_df["roe"] * 100, 2)
pb_df.loc[:, "proportion"] = round(pb_df["proportion"] * 100, 2)


apply_info(pb_df)
# pb_df = pb_df.drop_duplicates(subset='industry', keep='first')
pb_df.to_csv("data/pb_df.csv")

print(
    tabulate(
        pb_df[
            [
                #"ts_code",
                "name",
                "industry",
                "state",
                "per",
                "circ_mv",
                "pb",
                "rank_mvpb",
                "dv_ttm",
                "rank_dv",
                "roe",
                "rank_roe",
                "proportion",
                "rank_pro",
                "rank",
            ]
        ].head(50),
        headers="keys",
        showindex=False,
    )
)


# # # %%
# final_df = pd.DataFrame()

# for b in tqdm.tqdm(up_df["ts_code"].values, desc="Processing"):
#     # if b in pb_df["ts_code"].values:
#     #     continue
#     if b not in up_df["ts_code"].values:
#         print(b, "is not in up_df")
#         continue
#     if up_df[up_df["ts_code"] == b].iloc[0]["proportion"] < up_threshold:
#         continue
#     # 从 daily 里读 csv,只读第一行然后按 circ_mv 排序
#     df = pd.read_csv(os.path.join(daily_data_dir, f"{b}.csv"))
#     score, annualized_returns, r_squared = apply_score(df, 60)
#     df = df.sort_values(by="trade_date", ascending=False)
#     # df["abs_vol"] = np.where(df["close"] > df["open"], df["vol"], -df["vol"])
#     vol = df["vol"].head(10).sum()
#     # vol = (
#     #     df.iloc[0]["vol"]
#     #     if df.iloc[0]["close"] > df.iloc[0]["open"]
#     #     else -df.iloc[0]["vol"]
#     # )
#     vol_ratio = vol / df["vol"].head(120).sum()
#     # hh = df["high"].head(120).max()
#     # ll = df["low"].head(120).min()
#     ma60 = df["close"].head(60).mean()
#     price_ratio = (df.iloc[0]["close"] - ma60) / ma60
#     df = df.head(1)
#     ts_code = df["ts_code"].iloc[0]
#     name = chnname(ts_code)
#     industry = chnindustry(ts_code)
#     df["name"] = name
#     df["industry"] = industry
#     df["vol_r"] = round(vol_ratio * 100, 2)
#     df["score"] = round(score * 100, 2)
#     df["a_re"] = round(annualized_returns, 2)
#     df["r_sq"] = round(r_squared, 2)
#     df["p_r"] = round(price_ratio * 100, 2)
#     df["proportion"] = up_df[up_df["ts_code"] == ts_code].iloc[0]["proportion"]
#     final_df = pd.concat([final_df, df])

# final_df["rank"] = final_df["score"].rank(method="average", ascending=False)
# final_df["rank_s"] = round(final_df["rank"] / len(final_df) * 100, 2)
# final_df["rank"] = final_df["vol_r"].rank(method="average", ascending=False)
# final_df["rank_v"] = round(final_df["rank"] / len(final_df) * 100, 2)

# # final_df["rank"] = final_df["p_r"].rank(method="average", ascending=False)
# # final_df["rank_p"] = round(final_df["rank"] / len(final_df) * 100, 2)

# final_df["rank"] = (final_df["rank_v"] + final_df["rank_s"]) / 2
# final_df["proportion"] = round(final_df["proportion"] * 100, 2)
# final_df["circ_mv"] = round(final_df["circ_mv"] / 1e4, 2)

# # final_df = pd.concat([pb_df, final_df], axis=0)

# print(len(final_df))
# final_df = final_df.sort_values(by="rank", ascending=False)
# final_df["rank"] = round(final_df["rank"], 2)
# final_df.reset_index(inplace=True)
# apply_info(final_df)
# final_df.to_csv("data/final.csv")
# print(
#     tabulate(
#         final_df[
#             [
#                 "ts_code",
#                 "name",
#                 "industry",
#                 "state",
#                 "per",
#                 "circ_mv",
#                 # "pb",
#                 # "rank_mvpb",
#                 # "roe",
#                 # "rank_roe",
#                 # "dv_ttm",
#                 # "rank_dv",
#                 "vol_r",
#                 "rank_v",
#                 "p_r",
#                 "score",
#                 "rank_s",
#                 "rank",
#                 "proportion",
#             ]
#         ].head(50),
#         headers="keys",
#     )
# )

# # %%
data = {"name": [], "ts_code": [], "weight": []}


def append_data(name, ts_code, weight):
    data["name"].append(name)
    data["ts_code"].append(ts_code)
    data["weight"].append(weight)


# 40% US
append_data("纳指100", "159501.SZ", 0.12)
append_data("标普500", "513650.SH", 0.08)
append_data("道琼斯", "513400.SH", 0.02)
append_data("美国50", "513850.SH", 0.01)
append_data("纳指科技", "159509.SZ", 0.01)
append_data("纳指生物科技", "513290.SH", 0.01)
append_data("标普生物科技", "159502.SZ", 0.01)
# append_data("标普消费", "159529.SZ", 0.01)
append_data("标普油气", "159518.SZ", 0.01)
append_data("法国", "513080.SH", 0.01)
# append_data("德国", "159561.SZ", 0.01)
# append_data("沙特", "520830.SH", 0.01)
append_data("日经", "513880.SH", 0.01)
append_data("东证", "513800.SH", 0.01)
append_data("中韩半导体", "513310.SH", 0.01)
append_data("东南亚科技", "513730.SH", 0.01)
# append_data("亚太精选", "159687.SZ", 0.01)
# append_data("新兴亚洲", "520580.SH", 0.01)

append_data("美债", "501300.SH", 0.01)
append_data("油气", "162411.SZ", 0.02)
append_data("豆粕", "159985.SZ", 0.01)
append_data("商品", "510170.SH", 0.01)
append_data("原油", "501018.SH", 0.01)
append_data("有色", "159980.SZ", 0.01)
append_data("能源化工", "159981.SZ", 0.01)
append_data("白银", "161226.SZ", 0.01)

append_data("黄金", "518880.SH", 0.1)

append_data("政金债", "511520.SH", 0.03)
append_data("30年国债", "511090.SH", 0.03)
append_data("活跃国债", "511020.SH", 0.03)

append_data("恒生", "159920.SZ", 0.02)
append_data("恒科", "513130.SH", 0.02)
append_data("中概互联", "159605.SZ", 0.02)
append_data("A500", "159338.SZ", 0.01)
append_data("中证A50", "159593.SZ", 0.01)
append_data("沪深300", "510300.SH", 0.01)
append_data("中证500", "510500.SH", 0.01)
append_data("创业板", "159915.SZ", 0.01)
append_data("科创板", "588000.SH", 0.01)

append_data("银行", "516310.SH", 0.02)
append_data("中证红利", "515180.SH", 0.02)
append_data("红利低波", "563020.SH", 0.02)
append_data("红利低波100", "159307.SZ", 0.02)
append_data("红利质量", "159758.SZ", 0.02)
append_data("结构调整", "512950.SH", 0.02)
append_data("恒生红利低波", "159545.SZ", 0.02)
append_data("港股红利", "513630.SH", 0.02)
append_data("恒生高股息", "513690.SH", 0.02)

m = 750000
df = pd.DataFrame(data)
df["value"] = df["weight"] * 750000
sum = round(df["value"].sum(), 2)
res = m - sum

df.set_index("ts_code", inplace=True)
for i in df.index:
    history_df = pd.read_csv(os.path.join(daily_data_dir, f"{i}.csv"))
    bid_num, bid_grid, bid_price, bid_percent, bid_profit, atr, close = calc_net(
        history_df, df.loc[i, "value"]
    )
    df.loc[i, "bid_num"] = bid_num
    df.loc[i, "bid_grid"] = bid_grid
    df.loc[i, "bid_price"] = bid_price
    df.loc[i, "bid_percent"] = bid_percent
    df.loc[i, "bid_profit"] = bid_profit
    df.loc[i, "atr"] = atr
    df.loc[i, "close"] = close
    if i in hold_df.index:
        df.loc[i, "num"] = float(hold_df.loc[i, "num"])
        df.loc[i, "hold_value"] = float(hold_df.loc[i, "num"]) * close
        df.loc[i, "hold_diff"] = float(df.loc[i, "value"]) - float(
            df.loc[i, "hold_value"]
        )
        if abs(df.loc[i, "hold_diff"]) < close * 100:
            df.loc[i, "hold_diff"] = 1
    else:
        df.loc[i, "num"] = 0
        df.loc[i, "hold_value"] = 0
        df.loc[i, "hold_diff"] = 0

print(round(df["weight"].sum(), 2), res)
# df = df.sort_values(by='weight', ascending=False)

apply_info(df)
print(
    tabulate(
        df[
            [
                "name",
                #"ts_code",
                "bid_num",
                "bid_grid",
                "hold_diff",
                "weight",
                "value",
                "bid_price",
                "bid_percent",
                "bid_profit",
                "atr",
                # "close",
                "state",
                "per",
                "num",
                "hold_value",
            ]
        ],
        headers="keys",
    )
)


print("sell: ")
filtered_df = up_df[up_df["ts_code"].isin(hold_df.index)]
filtered_df = filtered_df[~filtered_df["ts_code"].isin(df.index)]
filtered_df = filtered_df.sort_values(by="proportion", ascending=False)
filtered_df["name"] = filtered_df["ts_code"].apply(chnname)
filtered_df["proportion"] = round(filtered_df["proportion"] * 100, 2)
filtered_df.set_index("ts_code", inplace=True)
for i in filtered_df.index:
    history_df = pd.read_csv(os.path.join(daily_data_dir, f"{i}.csv"))
    bid_num, bid_grid, bid_price, bid_percent, bid_profit, atr, close = calc_net(
        history_df, 10000
    )
    filtered_df.loc[i, "bid_num"] = bid_num
    filtered_df.loc[i, "bid_grid"] = bid_grid
    filtered_df.loc[i, "bid_price"] = bid_price
    filtered_df.loc[i, "bid_percent"] = bid_percent
    filtered_df.loc[i, "bid_profit"] = bid_profit
    filtered_df.loc[i, "atr"] = atr
    filtered_df.loc[i, "close"] = close
    filtered_df.loc[i, "rank"] = pb_df.loc[i, "rank"]

print(
    tabulate(
        filtered_df[
            [
                "name",
                "proportion",
                "bid_num",
                "bid_grid",
                "bid_price",
                "bid_percent",
                "bid_profit",
                "atr",
                "close",
                "rank",
            ]
        ],
        headers="keys",
    )
)

# %%
