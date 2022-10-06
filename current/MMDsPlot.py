import numpy as np
import matplotlib.pyplot as plt

MMDs = np.transpose(np.loadtxt("current\MMDsMonteCarlo.csv",delimiter=","))
print(MMDs)


plt.rc('font', size=20)
plt.plot(MMDs[0],np.log(MMDs[1]),label = "2D MMD Values with nD program")
plt.plot(MMDs[0],np.log(MMDs[2]),label = "4D MMD Values")
plt.plot(MMDs[0],np.log(MMDs[3]),label = "7D MMD Values")
plt.title("Convergences of MMDs in 2, 4 and 7 Dimensions")
plt.ylabel("MMD Values (Recorded per 5 iterations and normalized)")
plt.xlabel("Iteration Steps")
plt.legend()
plt.show()