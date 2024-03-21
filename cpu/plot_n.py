import numpy as np
import matplotlib.pyplot as plt
import glob

GRID_SIZE = 1024 #grid size
num_cols = 2 #number of columns 
tempNum = 10 #number of steps included
offset = 0 #controls at what point you want to start
fig, axs = plt.subplots(int(tempNum/num_cols), num_cols)
folder = f'GRID_{GRID_SIZE}/*'

ourFiles = glob.glob(folder)
ourFiles = sorted(ourFiles)[offset:tempNum+offset]
i = 0
for i, file in enumerate(ourFiles): 
    data = np.loadtxt(file)
    data = np.reshape(data, (GRID_SIZE, GRID_SIZE))
    
    row = i // num_cols
    col = i % num_cols
    axs[row, col].imshow(data)
    axs[row, col].set_title(f" T = {file[-12:-4]}")

plt.tight_layout()
plt.show()
