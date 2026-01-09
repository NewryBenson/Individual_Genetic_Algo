import matplotlib.pyplot as plt
import numpy as np

results = np.genfromtxt("r0123456.csv", delimiter=",")

plt.plot(results[:,1], results[:,2], label="average")
plt.plot(results[:,1], results[:,3], label="best")
plt.legend()
plt.ylim([67000,72000])
plt.xlabel("time (s)")
plt.ylabel("cycle length")
plt.show()