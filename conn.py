from scipy import io
import numpy as np
import matplotlib.pyplot as plt

ge_mlp = np.load(r'ret\mlp-GE-best-it_500.npy')
ge_mlp += 1

mat_name = r'E:\code\code_mask_hw\mats\ge_avg-mELE-u10000000-d10-k0.02-t1000-i500-c9-s200-f.mat'
# mat_name = r'E:\code\code_mask_hw\mats\ge_avg-mELE-u10000000-d10-k0.02-t1000-i500-c9-s200-f.mat'
dict = io.loadmat(mat_name)
ge_ele = dict['ge_avg'][0, :]
ge_ele = ge_ele[0:100:2]

mat_name2 = r'E:\code\code_mask_hw\mats\ge_avg-mSNPPE-u10000000-d10-k0.02-t1000-i500-c9-s200-f.mat'

x = range(0, 500, 10)
plt.plot(x, ge_mlp, label='MLP')
plt.plot(x, ge_ele, label='ELE')
# plt.title('Compare')
plt.ylabel('GE')
plt.xlabel('Number of attack traces')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.show()

pass