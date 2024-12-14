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
from tabulate import tabulate


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
buy_list = pd.read_csv("./data/buy_list.csv")["Column1"].tolist()
up_list = pd.read_csv("./data/up_list.csv")["Column1"].tolist()
daily_data_dir = "./data/daily"


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
# hold and recent sell
hold_dict = {
    "000028.SZ": 29.78,  # 国药一致
    "000157.SZ": 7.00,  # 中联重科
    "000537.SZ": 9.45,  # 中绿电
    "001234.SZ": 16.25,  # 泰慕士
    "001260.SZ": 11.88,  # 坤泰
    "001299.SZ": 13.73,  # 美能能源
    "001366.SZ": 11.64,  # 播恩
    "001376.SZ": 15.8,  # 百通
    "001979.SZ": 11.65,  # 招商蛇口
    "002546.SZ": 4.23,  # 新联电子
    "159329.SZ": 1.035,  # 沙特
    "159687.SZ": 1.285,  # 亚太
    "159869.SZ": 1.064,  # 游戏 etf
    "159941.SZ": 1.203,  # 纳指
    "300784.SZ": 68.90,  # 利安
    "300998.SZ": 17.20,  # 宁波方正
    "301107.SZ": 28.84,  # 瑜欣电子
    "301212.SZ": 26.35,  # 联盛化学
    "511090.SH": 118.687,  # 30年国债
    "511380.SH": 11.703,  # 可转债
    "511520.SH": 112.668,  # 政金债
    "513030.SH": 1.487,  # 德国
    "513500.SH": 2.23,  # 标普500
    "513730.SH": 1.412,  # 东南亚
    "515790.SH": 0.894,  # 光伏
    "513800.SH": 1.388,  # 东证
    "516160.SH": 2.062,  # 新能源
    "562800.SH": 0.562,  # 稀有金属
    "588290.SH": 1.501,  # 芯片科创
    "600016.SH": 4.09,  # 民生银行
    "600096.SH": 22.09,  # 云天化
    "600325.SH": 6.61,  # 华发股份
    "600373.SH": 13.51,  # 中文传媒
    "600585.SH": 26.25,  # 海螺水泥
    "600780.SH": 6.00,  # 通宝能源
    "600900.SH": 28.39,  # 长江电力
    "601068.SH": 4.75,  # 中铝
    "601101.SH": 8.9,  # 昊华科技
    "601117.SH": 8.38,  # 中国化学
    "601666.SH": 10.13,  # 平煤股份
    "603056.SH": 14.4,  # 德邦股份
    "603167.SH": 8.71,  # 渤海轮渡
    "603172.SH": 14.81,  # 万丰
    "603588.SH": 5.4,  # 高能环境
    "688036.SH": 96.98,  # 传音控股
    "688443.SH": 30.56,  # 智翔金泰
    "688530.SH": 21.67,  # 欧莱
    "688638.SH": 30.07,  # 誉辰智能
    "688695.SH": 34.91,  # 中创
}
recent_sell_dict = {
    # 12.12
    "601107.SH": 5.27,  # 四川成渝
    # 12.10
    "300995.SZ": 19.47,  # 奇德新材
    "600036.SH": 38.92,  # 招商银行
    "600064.SH": 8.33,  # 南京高科
    # 12.9
    "000543.SZ": 8.36,  # 皖能电力
    "002508.SZ": 23.01,  # 老板电器
    "600328.SH": 8.49,  # 中盐化工
    # 12.6
    "603280.SH": 23.71,  # 南方路机
    "560010.SH": 2.58,  # 1000
    # 12.5
    "601669.SH": 5.5,  # 中国电建
    # 12.4
    "600089.SH": 13.05,  # 特变电工
    "601077.SH": 5.69,  # 渝农商行
    "002394.SH": 8.21,  # 联发股份
    # 12.3
    "600894.SH": 13.36,  # 广日股份
    "002233.SZ": 7.71,  # 塔牌
    "002191.SZ": 4.67,  # 劲嘉
    "000517.SZ": 2.55,  # 荣安
    "000504.SZ": 10.39,  # 南华
    # 12.2
    "600153.SH": 10.8,  # 建发
    # 11.28
    "600162.SH": 1.9,  # 香江
    # 11.27
    "515220.SH": 1.166,  # 煤炭
    # 11.26
    "601006.SH": 6.66,  # 大秦铁路
    # 11.25
    "513090.SH": 1.479,  # 香港证券
    "603816.SH": 28.06,  # 顾家家居
    "600008.SH": 3.25,  # 首创
    "301359.SH": 23.35,  # 东南
    "159922.SZ": 6.065,  # 500
    "512200.SH": 1.569,  # 地产
    "512480.SH": 0.961,  # 半导体
    # 11.21
    "001256.SZ": 18.28,  # 炜冈
    "001259.SZ": 24.02,  # 利仁
    "601069.SH": 12.64,  # 西部黄金
    # 11.20
    "603272.SH": 13.93,  # 联翔
    "301234.SZ": 27.7,  # 五洲医疗
    "001209.SZ": 14.22,  # 洪兴
    "301309.SZ": 27.85,  # 万德凯
    "001226.SZ": 25.56,  # 拓山
    "301061.SZ": 55.61,  # 匠心家居
    "000581.SZ": 18.29,  # 威孚高科
    "603048.SH": 14.96,  # 浙江黎明
    "688750.SH": 29.85,  # 金天
    "002895.SZ": 21.4,  # 川恒股份
    # 11.19
    "301043.SZ": 27.8,  # 绿岛风
    "301388.SZ": 22.58,  # 欣灵电气
    "001368.SZ": 19.55,  # 通达创智
    "603216.SH": 10.49,  # 梦天家居
    "603307.SH": 31.25,  # 金泉
    "601919.SH": 14.6,  # 中远海控
}


def apply_info(df):
    df["state"] = ""
    df["per"] = np.NAN
    for i in df.index:
        ts_code = df.loc[i, "ts_code"]
        if ts_code in hold_dict:
            df.loc[i, "state"] = "hold"
            daily_df = pd.read_csv(os.path.join(daily_data_dir, f"{ts_code}.csv"))
            daily_df = daily_df.sort_values(by="trade_date", ascending=True)
            close = daily_df.iloc[-1]["close"]
            df.loc[i, "per"] = (close - hold_dict[ts_code]) / hold_dict[ts_code] * 100
        elif ts_code in recent_sell_dict:
            df.loc[i, "state"] = "sell"
            daily_df = pd.read_csv(os.path.join(daily_data_dir, f"{ts_code}.csv"))
            daily_df = daily_df.sort_values(by="trade_date", ascending=True)
            close = daily_df.iloc[-1]["close"]
            df.loc[i, "per"] = (
                (close - recent_sell_dict[ts_code]) / recent_sell_dict[ts_code] * 100
            )


# %%
mini_df = industry_df[industry_df["pe"] > 0]
mini_df = mini_df[mini_df["ts_code"].isin(up_list)]
print(len(mini_df))
base_mv_per = industry_df["circ_mv"].quantile(0.05)
mini_df = mini_df[mini_df["circ_mv"] < base_mv_per]
mini_df["circ_mv"] = mini_df["circ_mv"] / 10000
print(base_mv_per)
print("mini: ", len(mini_df))


for i in mini_df.index:
    ts_code = mini_df.loc[i, "ts_code"]
    df = pd.read_csv(os.path.join(daily_data_dir, f"{ts_code}.csv"))
    df = apply_bbands(df, period=60)
    df = df.sort_values(by="trade_date", ascending=False)
    df = df.head(1)
    mini_df.loc[i, "rel"] = df["rel"].iloc[0]

mini_df = mini_df.sort_values(by="rel", ascending=True)
apply_info(mini_df)

mini_df["per"] = round(mini_df["per"], 3)
mini_df["circ_mv"] = round(mini_df["circ_mv"], 3)
mini_df["pe"] = round(mini_df["pe"], 3)
mini_df["pb"] = round(mini_df["pb"], 3)
mini_df["dv_ttm"] = round(mini_df["dv_ttm"], 3)
mini_df["rel"] = round(mini_df["rel"], 3)
# mini_df = mini_df.sort_values(by='circ_mv')
# mini_df = mini_df.drop_duplicates(subset='industry', keep='first')
mini_df.to_csv("data/mini_df.csv")

print(
    tabulate(
        mini_df[
            [
                "ts_code",
                "name",
                "industry",
                "act_ent_type",
                "state",
                "per",
                "circ_mv",
                "pe",
                "pb",
                "dv_ttm",
                "rel",
            ]
        ].head(50),
        headers="keys",
    )
)


# %%
ex_dict = {
    "000517.SZ": "荣安地产,民",
    "000672.SZ": "上峰水泥,民",
    "002191.SZ": "劲嘉股份,民营",
    "002233.SZ": "塔牌集团,民营",
    "002478.SZ": "常宝股份,民营,钢加工",
    "002623.SZ": "亚玛顿,民营",
    "002727.SZ": "一心堂,民营",
    "600884.SH": "杉杉股份,民营",
    "601339.SH": "百隆东方,民",
    "601886.SH": "江河集团,民营",
    "603588.SH": "高能环境,民营",
    "600998.SH": "九州通,民营",
    "002539.SZ": "云图控股,民营",
    "601877.SH": "正泰电器,民营",
    "601222.SH": "林洋能源,民营",
    "301276.SZ": "嘉曼服饰,民营",
    "603368.SH": "柳药股份,民营",
    "603035.SH": "常熟汽饰,民",
    "603599.SH": "广信股份,民营",
}

base_df_per = industry_df["pb"].quantile(0.1)
base_pb = 1 if base_df_per < 1 else base_df_per
print("base_pb: ", base_pb, "base_df_per: ", base_df_per)

pb_df = industry_df[industry_df["pb"] <= base_pb]
print(len(pb_df))
pb_df = pb_df[pb_df["pe"] > 0]
print(len(pb_df))
pb_df = pb_df[pb_df["ts_code"].isin(up_list)]
print(len(pb_df))
pb_df.loc[:, "circ_mv"] = pb_df["circ_mv"] / 10000
pb_df = pb_df[~pb_df["ts_code"].isin(list(ex_dict.keys()))]
pb_df = pb_df[pb_df["dv_ttm"] > 2]
print("pb: ", len(pb_df))

for i in pb_df.index:
    ts_code = pb_df.loc[i, "ts_code"]
    df = pd.read_csv(os.path.join(daily_data_dir, f"{ts_code}.csv"))
    df = apply_bbands(df, period=60)
    df = df.sort_values(by="trade_date", ascending=False)
    df = df.head(1)
    pb_df.loc[i, "rel"] = df["rel"].iloc[0]

pb_df = pb_df.sort_values(by="rel", ascending=True)
apply_info(pb_df)
pb_df["per"] = round(pb_df["per"], 3)
pb_df["circ_mv"] = round(pb_df["circ_mv"], 3)
pb_df["pe"] = round(pb_df["pe"], 3)
pb_df["pb"] = round(pb_df["pb"], 3)
pb_df["dv_ttm"] = round(pb_df["dv_ttm"], 3)
pb_df["rel"] = round(pb_df["rel"], 3)
# pb_df = pb_df.drop_duplicates(subset='industry', keep='first')
pb_df.to_csv("data/pb_df.csv")

print(
    tabulate(
        pb_df[
            [
                "ts_code",
                "name",
                "industry",
                "act_ent_type",
                "state",
                "per",
                "circ_mv",
                "pe",
                "pb",
                "dv_ttm",
                "rel",
            ]
        ].head(50),
        headers="keys",
    )
)

# %%
buy_df = pd.DataFrame()

for b in buy_list:
    # 从 daily 里读 csv,只读第一行然后按 circ_mv 排序
    df = pd.read_csv(os.path.join(daily_data_dir, f"{b}.csv"))
    df = apply_bbands(df, period=60)

    df = df.sort_values(by="trade_date", ascending=False)
    df = df.head(1)
    ts_code = df["ts_code"].iloc[0]
    name = chnname(ts_code)
    industry = chnindustry(ts_code)
    df["name"] = name
    df["industry"] = industry
    buy_df = pd.concat([buy_df, df])

buy_df = buy_df.sort_values(by="rel", ascending=True)
buy_df["circ_mv"] = round(buy_df["circ_mv"] / 1e4, 3)
buy_df["rel"] = round(buy_df["rel"], 3)
buy_df.reset_index(inplace=True)
apply_info(buy_df)
buy_df.to_csv("data/buy_df.csv")
print("buy: ", len(buy_df))
print(
    tabulate(
        buy_df[["ts_code", "name", "industry", "state", "per", "circ_mv", "rel"]].head(
            50
        ),
        headers="keys",
    )
)
# # %%
# up_df = pd.DataFrame()

# for b in tqdm.tqdm(up_list, desc="Processing"):
#     # 从 daily 里读 csv,只读第一行然后按 circ_mv 排序
#     df = pd.read_csv(os.path.join(daily_data_dir, f"{b}.csv"))
#     df = apply_bbands(df, period=60)
#     df = df.sort_values(by="trade_date", ascending=False)
#     df = df.head(1)
#     ts_code = df["ts_code"].iloc[0]
#     name = chnname(ts_code)
#     industry = chnindustry(ts_code)
#     df["name"] = name
#     df["industry"] = industry
#     up_df = pd.concat([up_df, df])

# print(len(up_df))
# up_df = up_df.sort_values(by="rel", ascending=True)
# up_df["circ_mv"] = up_df["circ_mv"] / 1e4
# up_df.reset_index(inplace=True)
# apply_info(up_df)
# up_df.to_csv("data/up_df.csv")
# up_df[["ts_code", "name", "industry", "state", "per", "circ_mv", "rel"]].head(50)

# %%
f_df = pd.DataFrame()

for c in fetf_df["ts_code"]:
    df = pd.read_csv(os.path.join(daily_data_dir, f"{c}.csv"))
    df = apply_bbands(df, period=60)
    df = df.sort_values(by="trade_date", ascending=False)
    df = df.head(1)
    df["name"] = chnname(c)
    f_df = pd.concat([f_df, df], ignore_index=True)

apply_info(f_df)
f_df = f_df.sort_values(by="rel", ascending=False)
f_df.reset_index(inplace=True)
f_df["rel"] = round(f_df["rel"], 3)
f_df["per"] = round(f_df["per"], 3)

money = 900000
f = money * 0.5
print(f, f / 10)
f3 = money * 0.5 / 3
print(f3)
print(tabulate(f_df[["ts_code", "name", "rel", "state", "per"]], headers="keys"))
