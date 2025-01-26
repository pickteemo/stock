from tqdm import tqdm
import time
import os
import pandas as pd
import tushare as ts

data = {'name': [], 'ts_code': [], 'weight': []}

def append_data(name, ts_code, weight):
    data["name"].append(name)
    data["ts_code"].append(ts_code)
    data["weight"].append(weight)

append_data("纳指100", "159501.SZ", 0.12)
append_data("标普500", "513650.SH", 0.08)
append_data("道琼斯", "513400.SH", 0.02)
append_data("美国50", "513850.SH", 0.01)
append_data("纳指科技", "159509.SZ", 0.01)
append_data("纳指生物科技", "513290.SH", 0.01)
append_data("标普生物科技", "159502.SZ", 0.01)
append_data("标普消费", "159529.SZ", 0.01)
append_data("标普油气", "159518.SZ", 0.01)
append_data("法国", "513080.SH", 0.01)
append_data("德国", "159561.SZ", 0.01)
append_data("沙特", "520830.SH", 0.01)
append_data("日经", "513880.SH", 0.01)
append_data("东证", "513800.SH", 0.01)
append_data("中韩半导体", "513310.SH", 0.01)
append_data("东南亚科技", "513730.SH", 0.01)
append_data("亚太精选", "159687.SZ", 0.01)
append_data("新兴亚洲", "520580.SH", 0.01)

append_data("美债", "501300.SH", 0.01)
append_data("油气", "162411.SZ", 0.02)
append_data("豆粕", "159985.SZ", 0.02)
append_data("原油", "501018.SH", 0.01)
append_data("有色", "159980.SZ", 0.01)
append_data("能源化工", "159981.SZ", 0.01)
append_data("白银", "161226.SZ", 0.01)

append_data("黄金", "512800.SH", 0.1)

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


daily_dir = os.path.join("data", "daily")
TOEKN_PATH = os.path.expanduser("./.tushare.token")
with open(TOEKN_PATH, "r") as f:
    token = f.read().strip()
    ts.set_token(token=token)
    pro = ts.pro_api(token=token)


for c in tqdm(data["ts_code"], desc="Processing"):
    chunk_filename = os.path.join(daily_dir, f"{c}.csv")
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
        print(f"No data for {c}")
        continue
    df.to_csv(chunk_filename, index=False)