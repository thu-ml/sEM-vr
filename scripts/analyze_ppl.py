import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

train = np.array([float(line.split()[0]) for line in open('../ppl_study/ppl_to.log').readlines()])
test  = np.array([float(line.split()[0]) for line in open('../ppl_study/ppl_cv_to.log').readlines()])
train = np.cumsum(np.array(train))
test = np.cumsum(np.array(test))
print(train)
print(test)


# def smooth(f, ws = 100):
#     N = len(f)
#     sum = 0
#     series = []
#     for i in range(N):
#         sum += f[i]
#         if i-ws >= 0:
#             sum -= f[i-ws]
#             series.append(sum / ws)
#     return series


fig, ax = plt.subplots()
# ax.semilogx(smooth(train))
# ax.semilogx(smooth(test))
ax.plot(train - test)
# ax.semilogy(test, 'b')
fig.savefig('ppl_comparison.pdf')