# coding: utf-8
import numpy as np

all_temp_C = np.loadtxt('all_temp.csv', delimiter=',') # 1122*88
all_heat_W = np.loadtxt('all_heat.csv', delimiter=',') # 1122*88

all_temp_K = all_temp_C + np.ones(all_temp_C.shape) * 273

temp_diff = np.zeros((all_temp_K.shape[0], int(all_temp_K.shape[1] * (all_temp_K.shape[1] - 1)/2)))
temp_diff_4 = np.zeros(temp_diff.shape)
h = max(all_temp_K.shape[0], all_heat_W.shape[0])
N = all_temp_K.shape[1]
O = temp_diff.shape[1]
P = all_heat_W.shape[1]
Total_data = np.zeros((h, int(O + O + P)))

print(N)

print(Total_data.shape)

for i in range(N):
    for j in range(i + 1, N):
        #print(N)
        #print(i)
        temp_diff[:, int(N * i - i * (1 + i) / 2 + j - i - 1)] = all_temp_K[:, i] - all_temp_K[:, j]
        temp_diff_4[:, int(N * i - i * (1 + i) / 2 + j - i - 1)] = all_temp_K[:, i]**4 - all_temp_K[:, j]**4

for i in range(N):
    Total_data[:, i] = temp_diff[:, i]

for i in range(N):
    Total_data[:, N + i] = temp_diff_4[:, i]

for i in range(P):
    Total_data[:, N + N + i] = all_heat_W[:, i]

np.savetxt('Total_data.csv', Total_data, delimiter=", ")
