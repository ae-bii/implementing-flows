import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


e = math.e
pi = math.pi

def F_1(z):
    rx = abs(z[0] - center[0])
    ry = abs(z[1] - center[1])# Random point in the cluster
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    xval = rx * math.erf(rx/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(rx/alpha) ** 2)
    yval = ry * math.erf(ry/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(ry/alpha) ** 2)
    return [xval, yval]

def F_2(z):
    rx = abs(z[0] - center[0])
    ry = abs(z[1] - center[1])
    alpha = 1.5
    xval = alpha + rx - alpha * math.log(abs(alpha + rx))
    yval = alpha + ry - alpha * math.log(abs(alpha + ry))
    return [xval, yval]

def F_Individual_1(z, element):
    curCenter = center[element]
    r = abs(z - curCenter)# Random point in the cluster
    alpha = 1.5 # Consistent with the density (As this gets larger, less samples are moved close to 0)
    return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)

def F_Individual_2(z, element):
    curCenter = center[element]
    r = abs(z - curCenter)
    alpha = 1.5
    return alpha + r - alpha * math.log(abs(alpha + r))

def Beta_1Calculation():
    Proportion = 0.5
    xSummationDerivative = [0, 0]
    ySummationDerivative = [0, 0]
    Beta = []
    for j in range(0, len(StandardNormal)):
        #CurTuple = F(StandardNormal[j])
        ySummationDerivative[0] += F_1(StandardNormal[j])[0]
        ySummationDerivative[1] += F_1(StandardNormal[j])[1]
    for i in range(0, len(MixtureSample)):
        xSummationDerivative[0] += F_1(MixtureSample[i])[0]
        xSummationDerivative[1] += F_1(MixtureSample[i])[1]
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[0] + \
        (1/len(StandardNormal)) * ySummationDerivative[0])
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[1] + \
        (1/len(StandardNormal)) * ySummationDerivative[1])
    return np.multiply(Beta, Proportion)

def Beta_2Calculation():
    Proportion = 0.5
    xSummationDerivative = [0, 0]
    ySummationDerivative = [0, 0]
    Beta = []
    for j in range(0, len(StandardNormal)):
        #CurTuple = F(StandardNormal[j])
        ySummationDerivative[0] += F_2(StandardNormal[j])[0]
        ySummationDerivative[1] += F_2(StandardNormal[j])[1]
    for i in range(0, len(MixtureSample)):
        xSummationDerivative[0] += F_2(MixtureSample[i])[0]
        xSummationDerivative[1] += F_2(MixtureSample[i])[1]
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[0] + \
        (1/len(StandardNormal)) * ySummationDerivative[0])
    Beta.append((-1/len(MixtureSample)) * xSummationDerivative[1] + \
        (1/len(StandardNormal)) * ySummationDerivative[1])
    return np.multiply(Beta, Proportion)

def u(x, Beta_1, Beta_2):
    xval = (x[0] ** 2 / 2) +  Beta_1[0] * F_1(x)[0] + Beta_2[0] * F_2(x)[0]
    yval = (x[1] ** 2 / 2) +  Beta_1[1] * F_1(x)[1] + Beta_2[1] * F_2(x)[1]
    return [xval, yval]

def uConjugate(y, Beta_1, Beta_2):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConjugateVector = np.subtract(np.dot(MixtureSample[i], y), u(MixtureSample[i], Beta_1, Beta_2))
        ConvexCandidate.append(ConjugateVector)
    SumList = []
    for i in range(len(ConvexCandidate)):
        SumList.append(np.linalg.norm(ConvexCandidate[i]))
    index = SumList.index(max(SumList))
    return ConvexCandidate[index]

def D(Beta_1, Beta_2):
    xvalSummationOfx = 0
    yvalSummationOfx = 0
    xvalSummationOfy = 0
    yvalSummationOfy = 0
    for i in range(0, len(MixtureSample) - 1):
        xvalSummationOfx += u(MixtureSample[i], Beta_1, Beta_2)[0]
        yvalSummationOfx += u(MixtureSample[i], Beta_1, Beta_2)[1]
    for j in range(0, len(StandardNormal) - 1):
        xvalSummationOfy += uConjugate(StandardNormal[j], Beta_1, Beta_2)[0]
        yvalSummationOfy += uConjugate(StandardNormal[j], Beta_1, Beta_2)[1]
    D = np.linalg.norm([(1/len(MixtureSample)) * xvalSummationOfx + (1/len(StandardNormal)) * xvalSummationOfy, (1/len(MixtureSample)) * yvalSummationOfx + (1/len(StandardNormal)) * yvalSummationOfy])
    return D



def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        xval = OldMixtureSample[i][0] + Beta_1[0] * nd.Gradient(F_Individual_1)([OldMixtureSample[i][0]], 0) + Beta_2[0] * nd.Gradient(F_Individual_2)([OldMixtureSample[i][0]], 0)
        yval = OldMixtureSample[i][1] + Beta_1[1] * nd.Gradient(F_Individual_1)([OldMixtureSample[i][1]], 1) + Beta_2[1] * nd.Gradient(F_Individual_2)([OldMixtureSample[i][1]], 1)
        NewMixtureSample.append([xval, yval])
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = [1, -1]
    cov1 = [[0.5, 0], [0, 0.5]]
    mean2 = [-1, 1]
    cov2 = [[0.5, 0], [0, 0.5]]
    x = np.random.multivariate_normal(mean1, cov1, 500)
    y = np.random.multivariate_normal(mean2, cov2, 500)
    MixtureSample = []
    for i in range(500):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator():
    Sample = []
    x = np.random.standard_normal(500)
    y = np.random.standard_normal(500)
    for i in range(500):
        Sample.append([x[i], y[i]])
    return Sample


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------
MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()
CenterGeneratorList = MixtureSample + StandardNormal

plt.subplot(1,2,1)
plt.scatter(*zip(*StandardNormal), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.subplot(1,2,2)
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)

DValue = 0
while True: # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    center = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
    Beta_1 = Beta_1Calculation()
    Beta_2 = Beta_2Calculation()
    OldD = DValue
    DValue = D(Beta_1, Beta_2)
    print(abs(DValue - OldD))
    MixtureSample = SamplesUpdate(MixtureSample)
    if abs(DValue - OldD) < 0.001:
        break

plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
