import numpy as np
import matplotlib.pyplot as plt

MMDs = np.transpose(np.loadtxt("current/MMDs.csv",delimiter=","))
print(MMDs)


plt.rc('font', size=20)
plt.plot(MMDs[0],MMDs[1],label = "2D MMD Values")
plt.plot(MMDs[0],MMDs[2],label = "4D MMD Values")
plt.title("Convergences of MMDs in 2 and 4 Dimensions")
plt.ylabel("MMD Values (Recorded per 5 iterations and normalized)")
plt.xlabel("Iteration Steps")
plt.legend()
plt.show()