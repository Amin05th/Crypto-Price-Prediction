import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv("crypto_data.csv", parse_dates=True, index_col="Date")
fig, ax = plt.subplots()

# plot a line diagram
x = df.index
y = df['High']
ax.plot(x, y, linewidth=2)
ticks = list(df.index)
plt.xticks([ticks[i] for i in range(len(ticks)) if i % 100 == 0], rotation='vertical')
fig.tight_layout()
plt.show()

# plot animated plot
df_resampled = df.resample('5T').mean()

# Create the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(df_resampled.index, df_resampled['Close'])


def update(frame):
    # Generate new y data from DataFrame and interpolate to match x data
    x = df_resampled.index
    y = np.interp(x, x + pd.Timedelta(minutes=frame), df_resampled['Close'])
    line.set_ydata(y)  # update the y-data
    return line,


ani = FuncAnimation(fig, update, frames=60 * 24 * 10, interval=100, blit=True)

plt.show()
