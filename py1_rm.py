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
daily_folder = Path("data/daily")
plt_folder = Path("data/plt")
for folder in [daily_folder, plt_folder]:
    if not folder.exists():
        folder.mkdir(parents=True)
    else:
        shutil.rmtree(folder)
        folder.mkdir(parents=True)
