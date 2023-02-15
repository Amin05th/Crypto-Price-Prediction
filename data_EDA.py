import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

df = pd.read_csv("crypto_data.csv", skiprows=1)
fig, ax = plt.subplots()

# plot a line diagram
x = df['Date']
y = df['High']
ax.plot(x, y, linewidth=2)
ticks = list(df['Date'])
plt.xticks([ticks[i] for i in range(len(ticks)) if i % 100 == 0], rotation='vertical')
fig.tight_layout()
plt.show()

