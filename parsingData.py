# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import os
from pipeline import pipeline
from helper import getAnAsset, parseIntoTimeBars, parseIntoTickBars

ticker = 'WINJ21'

# Extracting trades only for ticker WINJ21
getAnAsset(
    ticker=ticker,
    in_folder=None,
    out_folder=None,
    verbose=True
)

# parsing market data for 60min time framed intervals
parsedMinData = parseIntoTimeBars(
    ticker=ticker,
    candles_periodicity='60min',
    in_folder=None,
    out_folder=None,
    verbose=True
)

parsedTickDat = parseIntoTickBars(
    ticker=ticker,
    numTicks=500000,
    in_folder=None,
    out_folder=None,
    verbose=True
)

# running pipeline for hyper-parameter search
ROOT_DIR = os.path.abspath(os.curdir)
pipeline(
    filePath=ROOT_DIR+'/data/WINJ21/WINJ21_500000ticks.csv',
    initInvest=5600*5,
    params=None,
    outFolder=None,
    verbose=True
)
