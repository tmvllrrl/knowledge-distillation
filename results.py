import numpy as np

kd_rn18 = [0.8016, 0.7951, 0.8071]
kd_rn18_in1k = [0.8609, 0.8613, 0.8641]

rn18 = [0.7818, 0.7881, 0.7827]
rn18_in1k = [0.8479, 0.8484, 0.8474]

print(np.mean(rn18), np.std(rn18))
print(np.mean(kd_rn18), np.std(kd_rn18))
print(np.mean(rn18_in1k), np.std(rn18_in1k))
print(np.mean(kd_rn18_in1k), np.std(kd_rn18_in1k))