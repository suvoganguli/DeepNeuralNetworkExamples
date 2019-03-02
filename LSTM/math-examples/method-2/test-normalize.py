import pandas as pd
import numpy as np

window = np.array([[3,2,3,4,5,10]]).T

print(window.shape)

for col_i in range(window.shape[1]):
	normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]

print(normalised_col)
