import numpy as np
from matplotlib import pyplot as plt

c1, c2 = [], []
qas, qab = [], []
for i in range(100):
    n = 1000
    alpha = i/100
    scale = np.random.rand(1, n)
    means = np.random.randint(1, 100, [1, n])
    R = np.random.normal(means, scale, [4, n])
    R = np.random.randint(1, 100, [4, n])

    E = np.max(R, 0)

    qa = np.quantile(E, 1-alpha, method='higher')
    qa1 = np.quantile(R[0], 1-alpha/4, method='higher')
    qa2 = np.quantile(R[1], 1-alpha/4, method='higher')
    qa3 = np.quantile(R[2], 1-alpha/4, method='higher')
    qa4 = np.quantile(R[3], 1-alpha/4, method='higher')

    c1.append(((((R - qa) <= 0).sum(0)) == 4).sum()/n)
    R[0] -= qa1
    R[1] -= qa2
    R[2] -= qa3
    R[3] -= qa4

    c2.append((((R <= 0).sum(0)) == 4).sum()/n)
    qas.append(qa)
    qab.append((qa1+qa2+qa3+qa4)/4)

plt.figure()
plt.plot(1-np.linspace(0,1, 100), np.array(c1), label='Max Method')
plt.plot(1-np.linspace(0,1, 100), np.array(c2), label='Bonferonni Method')
plt.plot(1-np.linspace(0,1, 100), 1-np.linspace(0,1, 100), label='Coverage goal')
plt.legend()

plt.figure()
plt.plot(qas)
plt.plot(qab)
plt.show()
