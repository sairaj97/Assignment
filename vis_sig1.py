# visualization of signal 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('/Users/kustarddev/Downloads/intern_dataset.csv')

# compute_Hc will help in computing the hurst component of the data for a label A, B, C and separately
# code below to separate the data
df_A = df[df['Label'] == 'A']

df_B = df[df['Label'] == 'B']

df_C = df[df['Label'] == 'C']

# the dataframe is separated


#visualization of signal 1, class A

plt.figure(dpi=1200)
plt.grid(b=False)
plt.axis(False)
plt.plot(np.array(df_A['Signal1']), linewidth = 0.25)
plt.show()

#visualization of signal 1, class B

plt.figure(dpi=1200)
plt.grid(b=False)
plt.axis(False)
plt.plot(np.array(df_B['Signal1']), linewidth = 0.25)
plt.show()

#visualization of signal 2, class C

plt.figure(dpi=1200)
plt.grid(b=False)
plt.axis(False)
plt.plot(np.array(df_C['Signal1']), linewidth = 0.25)
plt.show()


