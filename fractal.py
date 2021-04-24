import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import imageio

df = pd.read_csv('/Users/kustarddev/Downloads/intern_dataset.csv')
print("Total no of rows in given dataframe are",len(df))

# small ending part of data frame.
print(df.tail())

# different labels in given data.
print(df['Label'].unique())

# the amount of data is a lot so, some intervals of data is reviewed.
print(sns.lineplot(data=df[2:501], x='Time', y='Signal1', hue='Label'))
print(plt.show())
print(sns.lineplot(data=df[901000:901500], x='Time', y='Signal2', hue='Label'))
print(plt.show())


# Some source I used for making concept more clear¶
# 3blue 1brown's video and https://www.youtube.com/watch?v=-RmxLZF8adI for understanding
# https://support.numxl.com/hc/en-us/articles/216033663-Hurst-Hurst-Exponent#:~:
# text=The%20Hurst%20Exponent%20is%20estimated,(i.e.%20Hurst%20Exponent%20Estimate).
# Found this package where the DFA calculation had been done before and also plotted.
# (link): https://github.com/dokato/dfa/blob/master/dfa.py


def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0] // scale, scale)
    X = np.lib.stride_tricks.as_strided(x, shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut - xfit) ** 2))
    return rms


def dfa(x, scale_lim=[7, 12], scale_dens=0.125, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.
    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free
    view on neuronal oscillations, (2012).
    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ]
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2 ** np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc) ** 2))
    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    if show:
        fluctfit = 2 ** np.polyval(coeff, np.log2(scales))
        plt.loglog(scales, fluct, 'bo')
        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f' % coeff[0])
        plt.title('DFA')
        plt.xlabel(r'$\log_{10}$(time window)')
        plt.ylabel(r'$\log_{10}$<F(t)>')
        plt.legend()
        plt.show()
        print(coeff)
    return scales, fluct, coeff[0]


# compute_Hc will help in computing the hurst component of the data for a label A, B, C and separately
# code below to separate the data

# rows with label A in dataframe df
df_A = df[df['Label'] == 'A']
print("Total signals with Label A are",len(df_A))

# rows with label B in dataframe df
df_B = df[df['Label'] == 'B']
print("Total signals with Label B are",len(df_B))

# rows with label B in dataframe df
df_C = df[df['Label'] == 'C']
print("Total signals with Label B are",len(df_C))
# the dataframe is separated


# DFA for signal 1, label A

series_A = np.array(df_A['Signal1'])
scales, fluct, alpha = dfa(series_A, show=1)
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))


# DFA for signal 2, label A

series_A = np.array(df['Signal2'])
scales, fluct, alpha = dfa(series_A, show=True)
# 12000 values suggest 20 minutes of data.
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))


# DFA for signal 1, label B

series_B = np.array(df_B['Signal1'])
scales, fluct, alpha = dfa(series_B, show=1)
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))


# DFA for signal 2, label B

series_B = np.array(df_B['Signal2'])
scales, fluct, alpha = dfa(series_B, show=1)
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))


# DFA for signal 1, label C

series_C = np.array(df_C['Signal1'])
scales, fluct, alpha = dfa(series_C, scale_dens=0.2 ,show=1)
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))


# DFA for signal 2, label C

series_C = np.array(df_C['Signal2'])
scales, fluct, alpha = dfa(series_C, show=1)
print(scales)
print(fluct)
print("DFA exponent: {}".format(alpha))





