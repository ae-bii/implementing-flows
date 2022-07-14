import numpy as np
import random
import math
import numdifftools

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi


def F(z):
    r = abs(z - np.median(MixtureSamples))
    alpha = 0.5  # Not sure how to choose this value
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)


def BetaCalculation():  # Through gradient descent
    Proportion = 1  # Not sure how to choose this value
    xSummationDerivative = 0
    ySummationDerivative = 0
    for i in range(0, len(MixtureSamples)):
        xSummationDerivative += F(MixtureSamples[i])
    for j in range(0, len(StandardNormalSamples)):
        ySummationDerivative += F(StandardNormalSamples[j])
    Beta = (-1/len(MixtureSamples)) * xSummationDerivative + \
        (1/len(StandardNormalSamples)) * ySummationDerivative
    return Beta * Proportion


def u(x, Beta):
    return (x ** 2 / 2) + Beta * F(x)


def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(0, len(MixtureSamples)):
        ConvexCandidate.append(
            (MixtureSamples[i] * y) - u(MixtureSamples[i], Beta))

    return max(ConvexCandidate)


def MixtureSampleGenerator():
    SubSamples1 = np.random.normal(0, 2, 100)
    SubSamples2 = np.random.normal(6, 2, 100)
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


def LLCalculation(Beta):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSamples)):
        xSummation += u(MixtureSamples[i], Beta)
    for j in range(0, len(StandardNormalSamples)):
        ySummation += uConjugate(StandardNormalSamples[j], Beta)

    LL = 1/len(MixtureSamples) * xSummation + 1 / \
        len(StandardNormalSamples) * ySummation

    return LL


def SamplesUpdate(MixtureSamples):
    NewMixtureSamples = []
    for i in range(0, len(MixtureSamples)):
        NewMixtureSamples.append(MixtureSamples[i] + Beta * numdifftools.Gradient(F)([MixtureSamples[i]]))
    NewMixtureSamples = np.array(NewMixtureSamples)

    return NewMixtureSamples


StandardNormalSamples = np.random.standard_normal(100)
MixtureSamples = MixtureSampleGenerator()

for i in range(0, 10):
    Beta = BetaCalculation()
    MixtureSamples = SamplesUpdate(MixtureSamples)
    LL = LLCalculation(Beta)
    print(LL)
