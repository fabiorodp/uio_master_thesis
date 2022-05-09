# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from helper import getAnAsset, parseIntoTimeBars, parseIntoTickBars

# ########## parsing data
tickers = ['WINJ21', 'WINM21', 'WINQ21', 'WINV21', 'WINZ21', 'WING22']

for ticker in tickers:

    # Extracting trades only for ticker
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

    # parsing market data for 500k ticks framed intervals
    parsedTickDat = parseIntoTickBars(
        ticker=ticker,
        numTicks=500000,
        in_folder=None,
        out_folder=None,
        verbose=True
    )
