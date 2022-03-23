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

gains = np.array(saved_WING22["meanSumTradePLs"]) + \
        np.array(saved_WINZ21["meanSumTradePLs"]) +  \
        np.array(saved_WINV21["meanSumTradePLs"]) +  \
        np.array(saved_WINQ21["meanSumTradePLs"]) +  \
        np.array(saved_WINM21["meanSumTradePLs"]) +  \
        np.array(saved_WINJ21["meanSumTradePLs"])

b = np.argsort(gains)
c = np.sort(gains)

best = b[-1]  # -2 is very good for visualization, -8 has a huge outlier

objects = [
    saved_WING22,
    saved_WINZ21,
    saved_WINV21,
    saved_WINQ21,
    saved_WINM21,
    saved_WINJ21
]

g1 = saved_WINJ21["histRprime"][best]
g11 = []
for i in range(len(g1)):
    g111 = [0]
    for ii in range(1, len(g1[i])):
        g111.append(g1[i][ii] - g1[i][ii-1])
    g11.append(g111)

g2 = saved_WINM21["histRprime"][best]
for i in range(len(g2)):
    g111 = [0]
    for ii in range(1, len(g2[i])):
        g111.append(g2[i][ii] - g2[i][ii-1])
    g11[i] += g111

g3 = saved_WINQ21["histRprime"][best]
for i in range(len(g3)):
    g111 = [0]
    for ii in range(1, len(g3[i])):
        g111.append(g3[i][ii] - g3[i][ii-1])
    g11[i] += g111

g4 = saved_WINV21["histRprime"][best]
for i in range(len(g4)):
    g111 = [0]
    for ii in range(1, len(g4[i])):
        g111.append(g4[i][ii] - g4[i][ii-1])
    g11[i] += g111

g5 = saved_WINZ21["histRprime"][best]
for i in range(len(g5)):
    g111 = [0]
    for ii in range(1, len(g5[i])):
        g111.append(g5[i][ii] - g5[i][ii-1])
    g11[i] += g111

g6 = saved_WING22["histRprime"][best]
for i in range(len(g6)):
    g111 = [0]
    for ii in range(1, len(g6[i])):
        g111.append(g6[i][ii] - g6[i][ii-1])
    g11[i] += g111

# ##########
histGains = []
for i in range(len(g11)):
    g111 = [0]
    for ii in range(1, len(g11[i])):
        g111.append(sum(g11[i][:ii]))
    histGains.append(g111)

# ##########
for gain in histGains:
    plt.plot(gain)

plt.grid()
plt.show()

# ##########
meanGains = np.zeros(shape=(len(histGains[0]),))
for e in histGains:
    meanGains += np.array(e)

meanGains /= 100
plt.plot(meanGains)
plt.grid()
plt.show()
