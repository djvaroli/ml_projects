import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

X,y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
fig1, ax = plt.subplots()

ax.scatter(X[:,0], X[:,1])
fig1.show()

n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]

fig2, ax2 = plt.subplots()

ax2.plot(n_components, [m.bic(X) for m in models], label='BIC')
ax2.plot(n_components, [m.aic(X) for m in models], label='AIC')
ax2.legend(loc='best')
ax2.set_xlabel('n_components')

fig2.show()

# lowest AIC and BIC scores when n_components = 4
gmm = GaussianMixture(n_components=4)
# fit the data using the best model
gmm.fit(X)

fig3, ax3 = plt.subplots()
labels = gmm.predict(X)
print(labels)
t = gmm.predict_proba(X)
print(t)
ax3.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
fig3.show()

ax = sns.heatmap(t)
plt.show()
