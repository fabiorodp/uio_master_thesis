# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from package.helper import readPythonObjectFromFile


files = [
    "results/WING22/saved_sigmoid_WING22_60min.json",
    "results/WINZ21/saved_sigmoid_WINZ21_60min.json",
    "results/WINV21/saved_sigmoid_WINV21_60min.json",
    "results/WINQ21/saved_sigmoid_WINQ21_60min.json",
    "results/WINM21/saved_sigmoid_WINM21_60min.json",
    "results/WINJ21/saved_sigmoid_WINJ21_60min.json"
]

saved_WING22 = readPythonObjectFromFile(
    path=files[0],
    openingType="json"
)

saved_WINZ21 = readPythonObjectFromFile(
    path=files[1],
    openingType="json"
)

saved_WINV21 = readPythonObjectFromFile(
    path=files[2],
    openingType="json"
)

saved_WINQ21 = readPythonObjectFromFile(
    path=files[3],
    openingType="json"
)

saved_WINM21 = readPythonObjectFromFile(
    path=files[4],
    openingType="json"
)

saved_WINJ21 = readPythonObjectFromFile(
    path=files[5],
    openingType="json"
)

objects = [
    saved_WING22,
    saved_WINZ21,
    saved_WINV21,
    saved_WINQ21,
    saved_WINM21,
    saved_WINJ21
]

gains = np.array(saved_WING22["meanSumTradePLs"]) + \
        np.array(saved_WINZ21["meanSumTradePLs"]) +  \
        np.array(saved_WINV21["meanSumTradePLs"]) +  \
        np.array(saved_WINQ21["meanSumTradePLs"]) +  \
        np.array(saved_WINM21["meanSumTradePLs"]) +  \
        np.array(saved_WINJ21["meanSumTradePLs"])

b = np.argsort(gains)
c = np.sort(gains)

for obj in objects:
    opt = None
    for idx, i in enumerate(obj["histRprime"][b[-1]]):
        if idx == 0:
            opt = np.array(i)
        else:
            opt += np.array(i)

    plt.plot(opt/100)
    plt.grid()
    plt.show()

# b = np.argsort(saved["meanSumTradePLs"])
# c = np.sort(saved["meanSumTradePLs"])
