# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
import pandas as pd
from helper import plotAllBoxes

ROOT_DIR = os.path.abspath(os.curdir)

df1 = pd.read_csv(f"{ROOT_DIR}/results/optimals500k--tick.csv", index_col=0)
df2 = pd.read_csv(f"{ROOT_DIR}/results/optimals60--min.csv", index_col=0)

df = pd.concat([df1, df2], axis=1)

plotAllBoxes(
    data=df,
    initInvest=28000,
    showPlot=True
)

df.describe().to_latex()
