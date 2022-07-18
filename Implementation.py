import numpy as np
import random
import math
import numdifftools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def F_1(z, Flag = 1): 
    if Flag == 1:
        r = abs(z - RandomMixtureSample) # Random point in the cluster
    else:
        r = abs(z - RandomNormalSample)
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_2(z, Flag = 1):
    if Flag == 1:
        r = abs(z - RandomMixtureSample)
    else:
        r = abs(z - RandomNormalSample)
    alpha = 1.5
    return alpha + r - alpha * math.log(abs(alpha + r))

def Beta_1Calculation():  # Through gradient descent
    Proportion = 0.5 # Not sure how to choose this value
    xSummationDerivative = 0
    ySummationDerivative = 0
    for j in range(0, len(StandardNormalSamples)):
        ySummationDerivative += F_1(StandardNormalSamples[j], 2)
    for i in range(0, len(MixtureSamples)):
        xSummationDerivative += F_1(MixtureSamples[i], 1)
    Beta_1 = (-1/len(MixtureSamples)) * xSummationDerivative + \
        (1/len(StandardNormalSamples)) * ySummationDerivative
    return Beta_1 * Proportion

def Beta_2Calculation():
    Proportion = 0.5
    xSummationDerivative = 0
    ySummationDerivative = 0
    for j in range(0, len(StandardNormalSamples)):
        ySummationDerivative += F_2(StandardNormalSamples[j], 2)
    for i in range(0, len(MixtureSamples)):
        xSummationDerivative += F_2(MixtureSamples[i], 1)
    Beta_2 = (-1/len(MixtureSamples)) * xSummationDerivative + \
        (1/len(StandardNormalSamples)) * ySummationDerivative
    return Beta_2 * Proportion



def u(x, Beta_1, Beta_2):
    return (x ** 2 / 2) + Beta_1 * F_1(x, 1) + Beta_2 * F_2(x, 1)


def uConjugate(y, Beta_1, Beta_2):
    ConvexCandidate = []
    for i in range(0, len(MixtureSamples)):
        ConvexCandidate.append(
            (MixtureSamples[i] * y) - u(MixtureSamples[i], Beta_1, Beta_2))

    return max(ConvexCandidate)


def MixtureSampleGenerator():
    SubSamples1 = np.random.normal(0, 2, 100)
    SubSamples2 = np.random.normal(2, 2, 100)
    MixtureSamples = []
    for i in range(0, 100):
        RandomSelector = random.random()
        if RandomSelector < 0.7:
            MixtureSamples.append(SubSamples1[i])
        else:
            MixtureSamples.append(SubSamples2[i])
    # So with probability 0.7, choose from subsample 1, otherwise choose from subsample 2
    MixtureSamples = np.array(MixtureSamples)
    return MixtureSamples


def LLCalculation(Beta_1, Beta_2):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSamples)):
        xSummation += u(MixtureSamples[i], Beta_1, Beta_2)
    for j in range(0, len(StandardNormalSamples)):
        ySummation += uConjugate(StandardNormalSamples[j], Beta_1, Beta_2)

    LL = 1/len(MixtureSamples) * xSummation + 1 / \
        len(StandardNormalSamples) * ySummation

    return LL


def SamplesUpdate(MixtureSamples):
    NewMixtureSamples = []
    for i in range(0, len(MixtureSamples)):
        NewMixtureSamples.append(MixtureSamples[i] + Beta_1 * numdifftools.Gradient(F_1)([MixtureSamples[i]]) + Beta_2 * numdifftools.Gradient(F_1)([MixtureSamples[i]]))
    NewMixtureSamples = np.array(NewMixtureSamples)

    return NewMixtureSamples


StandardNormalSamples = np.random.standard_normal(100)
MixtureSamples = MixtureSampleGenerator()


steps = [MixtureSamples]

for i in range(0, 50): # Maybe there is a problem of overfitting
    RandomMixtureSample = MixtureSamples[random.randint(0,len(MixtureSamples) - 1)]
    RandomNormalSample = StandardNormalSamples[random.randint(0,len(StandardNormalSamples) - 1)]
    Beta_1 = Beta_1Calculation()
    Beta_2 = Beta_2Calculation()
    MixtureSamples = SamplesUpdate(MixtureSamples)
    LL = LLCalculation(Beta_1, Beta_2)
    print(LL)

    steps.append(MixtureSamples)

#----------------------------------------------------- GENERATE ANIMATION ---------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def animate(i):
    ax.clear()
    ax.hist(StandardNormalSamples, bins=15, color='r', alpha=0.5)
    ax.hist(steps[i], bins=20, color='b', alpha=0.5)
    plt.xlim([-5,5])
    plt.ylim([0,25])

ani = animation.FuncAnimation(fig, animate, interval = 150, repeat = True, frames = len(steps), repeat_delay = 500000)
plt.show()
plt.close()
#----------------------------------------------------------------------------------------------------------------------------------------