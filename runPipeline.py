# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
from pipeline import pipeline

ROOT_DIR = os.path.abspath(os.curdir)

ticker = 'WINJ21'

# running pipeline for hyper-parameter search
pipeline(
    filePath=ROOT_DIR+f"/data/{ticker}/{ticker}_500000ticks.csv",
    initInvest=5600*5,
    params=None,
    outFolder=None,
    verbose=True
)
