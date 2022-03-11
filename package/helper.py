# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import json
import pickle
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.graph_objects as go


# setting parent directory to be accessed
# os.chdir('..')

def savePythonObject(pathAndFileName: str, pythonObject,
                     savingType: str = "pickle") -> None:
    """Function to save a python's object to a file..."""
    if savingType == "pickle":
        f = open(pathAndFileName, "wb")
        pickle.dump(pythonObject, f)

    elif savingType == "json":
        savedJson = json.dumps(pythonObject)
        f = open(f"{pathAndFileName}.json", "w")
        f.write(savedJson)

    else:
        raise ValueError(f"Error: savingType = {savingType} not recognized!")


def readPythonObjectFromFile(path: str, openingType: str = "pickle"):
    """Function to read a file and convert to python's objects..."""
    if openingType == "pickle":
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        return saved

    elif openingType == "json":
        with open(path, 'r') as f:
            saved = f.read()
        return json.loads(saved)

    else:
        raise ValueError(f"Error: openingType = {openingType} not recognized!")


def plottingCandlestickChart(OHLCV: pd.DataFrame) -> None:
    """Plotting candlestick chart for OHLCV DataFrame..."""
    fig = go.Figure(
        data=[go.Candlestick(
            x=OHLCV.index,
            open=OHLCV['open'],
            high=OHLCV['high'],
            low=OHLCV['low'],
            close=OHLCV['close'])
        ]
    )
    fig.show()
