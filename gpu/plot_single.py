import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(5, 2, figsize=(10, 20))  # 5 rows, 2 columns for 10 plots
GRID_SIZE = 2048
step_size = 100
folder = f'GRID_{GRID_SIZE}/'

fig, ax = plt.subplots()
data = np.loadtxt(folder + f'state_1000.txt')
data = np.reshape(data, (GRID_SIZE, GRID_SIZE))

## True PDF plot
#sns.heatmap(data, cbar = False)

## Quick-n-dirty matplotlib plot
plt.imshow(data)

fig.savefig(f'{GRID_SIZE}.pdf')
plt.show()



