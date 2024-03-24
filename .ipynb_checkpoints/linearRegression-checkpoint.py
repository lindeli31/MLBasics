import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
#la funzione permette di suddivedere il dataset in uno di test e uno di train
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d 


dataset = load_diabetes() #ritorna un dizionario 

print(dataset.data.shape)
print(dataset.keys())
print(dataset.feature_names)
print(dataset.DESCR)

x = np.array(dataset.data)
print(x)
y = np.array(dataset.target)
print(y)
plt.figure(figsize=(10,5))
kwargs = dict(histtype = 'stepfilled', alpha = 0.3, density = False, bins = 30, ec="k")
plt.hist(y, **kwargs)
plt.show()
