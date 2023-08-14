import numpy as np
import matplotlib.pyplot as plt

# x_pred = np.loadtxt('force_val_pred.txt')
# x_true = np.loadtxt('force_val_true.txt')

# print(len(x_pred))
# print(len(x_true))

dat = np.loadtxt('tmp.dat')
x_pred = dat[:,0]
x_true = dat[:,1]
x_pred_max = np.max(x_pred)
x_pred_min = np.min(x_pred)
x_true_max = np.max(x_true)
x_true_min = np.min(x_true)
x_pred_bins = int((x_pred_max - x_pred_min) / 0.01)
x_true_bins = int((x_true_max - x_true_min) / 0.01)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.scatter(x_pred, x_true, s=2)
ax1.set_xlabel('x_pred')
ax1.set_ylabel('x_pred')
ax2.hist(x_true, bins=x_true_bins, label='x_true')
ax2.hist(x_pred, bins=x_pred_bins, label='x_pred')
ax2.set_xlabel('Force')
ax2.set_ylabel('Count')
plt.xlim([-1.5, 1.5])
# plt.ylim([0.0, 13000])
plt.legend()
plt.show()


# debug
plt.scatter(x_pred, x_true, s=0.01, alpha=0.5)
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.hist(x_pred, bins=1000)
plt.xlim([-5, 5])
plt.ylim([0, 50000])
plt.hist(x_true, bins=2000)
plt.xlim([-0.5, 0.5])
plt.ylim([0.0, 200000])
