# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import re
import numpy as np
from tqdm import tqdm
from helper import readPythonObjectFromFile


def mergeResults(files: list) -> dict:
    """Merging json files containing the pipeline results..."""

    saved = {
        "params": [],
        "numTrades": [],
        "histRprime": [],
        "meanPLs": []
    }

    for file in files:

        if len(re.findall("WINJ21", file)) != 0:
            _saved1 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        elif len(re.findall("WINM21", file)) != 0:
            _saved2 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        elif len(re.findall("WINQ21", file)) != 0:
            _saved3 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        elif len(re.findall("WINV21", file)) != 0:
            _saved4 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        elif len(re.findall("WINZ21", file)) != 0:
            _saved5 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        elif len(re.findall("WING22", file)) != 0:
            _saved6 = readPythonObjectFromFile(
                path=file,
                openingType="json"
            )

        else:
            raise ValueError(f"ERROR: Link {file} not found!")

    ord = (
        _saved1,
        _saved2,
        _saved3,
        _saved4,
        _saved5,
        _saved6
    )

    for idx, o in enumerate(tqdm(ord, desc="Merging results...")):

        if idx == 0:
            saved["params"] = ord[idx]["params"]
            saved["numTrades"] = np.array(
                [np.mean([len(ee) for ee in e])
                 for e in ord[idx]["histTradePLs"]
                 ])
            saved["histRprime"] = ord[idx]["histRprime"]
            saved["meanPLs"] = np.array(ord[idx]["meanSumTradePLs"])

        else:
            # ########## appending numTrades
            saved["numTrades"] = saved["numTrades"] + np.array(
                [np.mean([len(ee) for ee in e])
                 for e in ord[idx]["histTradePLs"]
                 ])

            # ########## fixing histRprime to be able to store
            # ########## in a continuing and ordered manner
            l = []
            for e, e1 in zip(saved["histRprime"], ord[idx]["histRprime"]):
                arr = np.array(e)
                arr = arr[:, -1][:, None]
                arr1 = np.array(e1) - 28000
                arr1 = arr1 + arr
                l.append(arr1.tolist())

            # ########## merging previous histRprime with current histRprime
            merged = [
                [ee+ee1 for ee, ee1 in zip(e, e1)]
                for e, e1 in zip(saved["histRprime"], l)
            ]
            saved["histRprime"] = merged

            # ########## summing meanPLs
            saved["meanPLs"] = saved["meanPLs"] + np.array(
                ord[idx]["meanSumTradePLs"])

    # ########## returning to be list
    saved["numTrades"] = saved["numTrades"].tolist()
    saved["meanPLs"] = saved["meanPLs"].tolist()
    return saved
