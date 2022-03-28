# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from helper import readPythonObjectFromFile


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

import seaborn as sns
hist = []
for i in histGains:
    hist.append(i[-1])

sns.displot(hist)
plt.show()

sns.boxplot(hist)
plt.show()

import pandas as pd
df = pd.Series(hist)
df.describe()

# ##########
sorted_saved = {
    "params": [],
    "TDErrors": [],
    "histRprime": [],
    "sumTradePLs": [],
    "meanSumTradePLs": [],
}

_SARSA = {
    "n": [],
    "basisFctType": [],
    "rewardType": [],
    "eta": [],
    "gamma": [],
    "epsilon": [],
    "initType": [],
    "lrScheduler": [],
    "group": []
}

for i in range(len(saved["params"])):
    if ("SARSA" in saved["params"][i]):
        _SARSA["n"].append(saved["params"][i][1])
        _SARSA["basisFctType"].append(saved["params"][i][2])
        _SARSA["rewardType"].append(saved["params"][i][3])
        _SARSA["eta"].append(saved["params"][i][4])
        _SARSA["gamma"].append(saved["params"][i][5])
        _SARSA["epsilon"].append(saved["params"][i][6])
        _SARSA["initType"].append(saved["params"][i][7])
        _SARSA["lrScheduler"].append(saved["params"][i][8])

        if saved["meanSumTradePLs"][i] < -1000:
            _SARSA["group"].append(0)

        elif (saved["meanSumTradePLs"][i] >= -1000) \
                and (saved["meanSumTradePLs"][i] <= 0):
            _SARSA["group"].append(1)

        elif (saved["meanSumTradePLs"][i] > 0) \
                and (saved["meanSumTradePLs"][i] <= 1000):
            _SARSA["group"].append(2)

        elif (saved["meanSumTradePLs"][i] > 1000):
            _SARSA["group"].append(3)
    """
    sorted_saved["params"].append(saved["params"][i])
    sorted_saved["TDErrors"].append(saved["TDErrors"][i])
    sorted_saved["histRprime"].append(saved["histRprime"][i])
    sorted_saved["sumTradePLs"].append(saved["sumTradePLs"][i])
    sorted_saved["meanSumTradePLs"].append(saved["meanSumTradePLs"][i])
    """
sns.boxplot(sorted_saved["meanSumTradePLs"])
plt.show()

plt.hist(sorted_saved["meanSumTradePLs"], bins=10)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.get_dummies(pd.DataFrame(_SARSA))

count = df[df["group"]==3]
plt.hist(count["basisFctType_sigmoid123"])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["group"], axis=1),
    _SARSA["group"],
    test_size=0.2,
    random_state=1
)

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

sns.histplot(y_train)
plt.show()

import plotly.express as px
from sklearn.decomposition import PCA

features = df.columns.to_list()
del features[5]

pca = PCA()
components = pca.fit_transform(df[features])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=df["group"]
)
fig.update_traces(diagonal_visible=False)
fig.show()
