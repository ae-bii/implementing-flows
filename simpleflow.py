import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import math
import numdifftools as nd

#----------------------------------------------------- GENERATE TARGET DISTRIBUTION -----------------------------------------------------
mean, std = 0, 1 # mean and standard deviation
mu_X = np.sort(np.random.normal(mean, std, 100))
mu_Y = (1 / np.sqrt(2*np.pi*std**2)) * np.exp(-(mu_X-mean)**2 / (2*std**2))
#----------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------- GENERATE INITIAL DISTRIBUTION ----------------------------------------------------
m1, m2, s1, s2 = -1, 1, 0.5, 0.5
proportion = 0.7
rho_X = np.sort(np.random.normal(mean, std, 100))
component1 = (1 / np.sqrt(2*np.pi*s1**2)) * np.exp(-(rho_X-m1)**2 / (2*s1**2))
component2 = (1 / np.sqrt(2*np.pi*s2**2)) * np.exp(-(rho_X-m2)**2 / (2*s2**2))
rho_0_Y = (proportion * component1 + (1-proportion) * component2)
#----------------------------------------------------------------------------------------------------------------------------------------

def F_k(z):
    r = abs(z - np.median(steps[len(steps)-1][1]))
    alpha = 0.5 # Not sure how to choose this value
    return r * math.erf(r/alpha) + (alpha/math.sqrt(math.pi)) * math.exp(-(r/alpha) ** 2)  

def u_k(z, Beta):
    return (z ** 2 / 2) + Beta * F_k(z)

def uConjugate_k(y, Beta, rho_Y):
    ConvexCandidate = []
    for i in range(0, len(rho_Y)):
        ConvexCandidate.append((rho_Y[i] * y) - u_k(rho_Y[i], Beta))
    return max(ConvexCandidate)

def LLCalculation(Beta, rho, mu):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(rho)):
        xSummation += u_k(rho[i], Beta)
    for j in range(0, len(mu)):
        ySummation += uConjugate_k(mu[j], Beta, rho)

    LL = 1/len(rho) * xSummation + 1 / \
        len(mu) * ySummation

    return LL

def BetaCalculation(rho, mu):
    Proportion = 1  # Not sure how to choose this value
    xSummationDerivative = 0
    ySummationDerivative = 0
    for i in range(0, len(rho)):
        xSummationDerivative += F_k(rho[i])
    for j in range(0, len(mu)):
        ySummationDerivative += F_k(mu[j])
    Beta = (-1/len(rho)) * xSummationDerivative + (1/len(mu)) * ySummationDerivative
    return Beta * Proportion

def SamplesUpdate(rho_k, Beta):
    rho_kplusone = []
    for i in range(0,len(rho_k)):
        rho_kplusone.append( rho_k[i] + Beta * nd.Gradient(F_k)([rho_k[i]]) )
    return np.array(rho_kplusone)


steps = [(rho_X, rho_0_Y)]

for i in range(0, 20):
    rho_k_Y = steps[len(steps)-1][1]
    Beta = BetaCalculation(rho_k_Y, mu_Y)
    steps.append((rho_X, SamplesUpdate(rho_k_Y, Beta)))
    LL = LLCalculation(Beta, rho_k_Y, mu_Y)
    print(LL)
    

#----------------------------------------------------- GENERATE ANIMATION ---------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

numSteps = len(steps)

def animate(i):
    ax.clear()
    ax.plot(steps[i%numSteps][0], steps[i%numSteps][1], color = 'b')
    ax.plot(mu_X, mu_Y, color = 'r')
    plt.xlim([-4,4])
    plt.ylim([0,8])

ani = animation.FuncAnimation(fig, animate, interval = 50, repeat = True)
plt.show()
plt.close()
#----------------------------------------------------------------------------------------------------------------------------------------