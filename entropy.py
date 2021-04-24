import pandas as pd
import numpy as np



df = pd.read_csv('/Users/kustarddev/Downloads/intern_dataset.csv')

# compute_Hc will help in computing the hurst component of the data for a label A, B, C and separately
# code below to separate the data
df_A = df[df['Label'] == 'A']

df_B = df[df['Label'] == 'B']

df_C = df[df['Label'] == 'C']

# the dataframe is separated

# Approximate entropy - calculating entropy for a time series¶
# Found this code on : https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41
# which is in reference to the article on : https://en.wikipedia.org/wiki/Approximate_entropy


def ApEn_new(U, m, r):
    U = np.array(U)
    N = U.shape[0]

    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i + m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= 3, axis=0) / z
        return np.log(C).sum() / z

    return abs(_phi(m + 1) - _phi(m))



# using the above found code to calculate entropy,
entropy = 0
"""
for i in range(30):
    entropy = entropy + (ApEn_new(np.array(df_A['Signal1'])[i*10000:(i+1)*10000], 2, 300000)) 
print(abs(entropy))
"""
# it takes a long time to calculate so the code will be commented with the results pre-calculated.

print(ApEn_new(np.array(df_A['Signal1'])[0:10000], 2, 3)) # entropy : 0.0036
# The entropy came out to be 0.0 for the whole series Signal 1 label A.
print(ApEn_new(np.array(df_B['Signal1'])[0:10000], 2, 3)) # entropy : 0.00620
print(ApEn_new(np.array(df_C['Signal1'])[0:10000], 2, 3)) # entropy : 0.00660
print(ApEn_new(np.array(df_A['Signal2'])[0:10000], 2, 3)) # entropy : 0 for all three
print(ApEn_new(np.array(df_B['Signal2'])[0:10000], 2, 3))
print(ApEn_new(np.array(df_C['Signal2'])[0:10000], 2, 3))

#The above example is not the best way to assess, but it will roughly give a good idea
# of what is happening in the data, 0 entropy suggest that the time series not very random,
# and it may be due to the fact that Signal 2 is not changing very slowly with time and gets
# enropy zero, the previous values are very close to their next values. I divided the data
# into batch and calculated entropy for each batch and hit maximum of about 0.009(closer to 0.01).