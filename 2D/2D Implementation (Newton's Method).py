import numpy as np
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import family_of_f

random.seed(0)
np.random.seed(0)

e = math.e
pi = math.pi

def distance(z1, z2):
    sum = 0
    for i in range(0,len(z1)-1):
        sum += (z1[i] - z2[i]) ** 2
    return math.sqrt(sum)

def BetaNewton(): # Newton's method (Experimental)
    xSummationGradient_1 = 0
    ySummationGradient_1 = 0
    xSummationGradient_2 = 0
    ySummationGradient_2 = 0
    xSummationGradient_3 = 0
    xSummationGradient_4 = 0
    xSummationGradient_5 = 0
    ySummationGradient_3 = 0
    ySummationGradient_4 = 0
    ySummationGradient_5 = 0
    for i in range(0, len(MixtureSample)):
        xSummationGradient_1 += PotentialF_1.giulio_F(MixtureSample[i])
        xSummationGradient_2 += PotentialF_2.gaussian_F(MixtureSample[i])
        xSummationGradient_3 += PotentialF_3.multiquadric_F(MixtureSample[i])
        xSummationGradient_4 += PotentialF_4.inverseQuadratic_F(MixtureSample[i])
        xSummationGradient_5 += PotentialF_5.inverseMultiquadric_F(MixtureSample[i])
    for j in range(0, len(StandardNormal)):
        ySummationGradient_1 += PotentialF_1.giulio_F(StandardNormal[j])
        ySummationGradient_2 += PotentialF_2.gaussian_F(StandardNormal[j])
        ySummationGradient_3 += PotentialF_3.multiquadric_F(StandardNormal[j])
        ySummationGradient_4 += PotentialF_4.inverseQuadratic_F(StandardNormal[j])
        ySummationGradient_5 += PotentialF_5.inverseMultiquadric_F(StandardNormal[j])
    G_1 = (1/len(MixtureSample)) * xSummationGradient_1 - (1/len(StandardNormal)) * ySummationGradient_1
    G_2 = (1/len(MixtureSample)) * xSummationGradient_2 - (1/len(StandardNormal)) * ySummationGradient_2
    G_3 = (1/len(MixtureSample)) * xSummationGradient_3 - (1/len(StandardNormal)) * ySummationGradient_3
    G_4 = (1/len(MixtureSample)) * xSummationGradient_4 - (1/len(StandardNormal)) * ySummationGradient_4
    G_5 = (1/len(MixtureSample)) * xSummationGradient_5 - (1/len(StandardNormal)) * ySummationGradient_5
    G = np.array([G_1, G_2, G_3, G_4, G_5])
    yHessian_11 = 0
    yHessian_12 = 0
    yHessian_13 = 0
    yHessian_14 = 0
    yHessian_15 = 0
    yHessian_21 = 0
    yHessian_22 = 0
    yHessian_23 = 0
    yHessian_24 = 0
    yHessian_25 = 0
    yHessian_31 = 0
    yHessian_32 = 0
    yHessian_33 = 0
    yHessian_34 = 0
    yHessian_35 = 0
    yHessian_41 = 0
    yHessian_42 = 0
    yHessian_43 = 0
    yHessian_44 = 0
    yHessian_45 = 0
    yHessian_51 = 0
    yHessian_52 = 0
    yHessian_53 = 0
    yHessian_54 = 0
    yHessian_55 = 0
    for l in range(0, len(StandardNormal)):
        yHessian_11 += np.dot(nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]), nd.Gradient(PotentialF_1.giulio_F)(StandardNormal[l]))
        yHessian_12 += np.dot(nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]), nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]))
        yHessian_13 += np.dot(nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]), nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]))
        yHessian_14 += np.dot(nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]), nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]))
        yHessian_15 += np.dot(nd.Gradient(PotentialF_1.giulio_F)([StandardNormal[l]]), nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l]))
        yHessian_21 += np.dot(nd.Gradient(PotentialF_2.gaussian_F)([StandardNormal[l]]), nd.Gradient(PotentialF_1.giulio_F)(StandardNormal[l]))
        yHessian_22 += np.dot(nd.Gradient(PotentialF_2.gaussian_F)([StandardNormal[l]]), nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]))
        yHessian_23 += np.dot(nd.Gradient(PotentialF_2.gaussian_F)([StandardNormal[l]]), nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]))
        yHessian_24 += np.dot(nd.Gradient(PotentialF_2.gaussian_F)([StandardNormal[l]]), nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]))
        yHessian_25 += np.dot(nd.Gradient(PotentialF_2.gaussian_F)([StandardNormal[l]]), nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l]))
        yHessian_31 += np.dot(nd.Gradient(PotentialF_3.multiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_1.giulio_F)(StandardNormal[l]))
        yHessian_32 += np.dot(nd.Gradient(PotentialF_3.multiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]))
        yHessian_33 += np.dot(nd.Gradient(PotentialF_3.multiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]))
        yHessian_34 += np.dot(nd.Gradient(PotentialF_3.multiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]))
        yHessian_35 += np.dot(nd.Gradient(PotentialF_3.multiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l]))
        yHessian_41 += np.dot(nd.Gradient(PotentialF_4.inverseQuadratic_F)([StandardNormal[l]]), nd.Gradient(PotentialF_1.giulio_F)(StandardNormal[l]))
        yHessian_42 += np.dot(nd.Gradient(PotentialF_4.inverseQuadratic_F)([StandardNormal[l]]), nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]))
        yHessian_43 += np.dot(nd.Gradient(PotentialF_4.inverseQuadratic_F)([StandardNormal[l]]), nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]))
        yHessian_44 += np.dot(nd.Gradient(PotentialF_4.inverseQuadratic_F)([StandardNormal[l]]), nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]))
        yHessian_45 += np.dot(nd.Gradient(PotentialF_4.inverseQuadratic_F)([StandardNormal[l]]), nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l]))
        yHessian_51 += np.dot(nd.Gradient(PotentialF_5.inverseMultiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_1.giulio_F)(StandardNormal[l]))
        yHessian_52 += np.dot(nd.Gradient(PotentialF_5.inverseMultiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_2.gaussian_F)(StandardNormal[l]))
        yHessian_53 += np.dot(nd.Gradient(PotentialF_5.inverseMultiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_3.multiquadric_F)(StandardNormal[l]))
        yHessian_54 += np.dot(nd.Gradient(PotentialF_5.inverseMultiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_4.inverseQuadratic_F)(StandardNormal[l]))
        yHessian_55 += np.dot(nd.Gradient(PotentialF_5.inverseMultiquadric_F)([StandardNormal[l]]), nd.Gradient(PotentialF_5.inverseMultiquadric_F)(StandardNormal[l]))
    H_11 = (1/len(StandardNormal)) * yHessian_11
    H_12 = (1/len(StandardNormal)) * yHessian_12
    H_13 = (1/len(StandardNormal)) * yHessian_13
    H_14 = (1/len(StandardNormal)) * yHessian_14
    H_15 = (1/len(StandardNormal)) * yHessian_15
    H_21 = (1/len(StandardNormal)) * yHessian_21
    H_22 = (1/len(StandardNormal)) * yHessian_22
    H_23 = (1/len(StandardNormal)) * yHessian_23
    H_24 = (1/len(StandardNormal)) * yHessian_24
    H_25 = (1/len(StandardNormal)) * yHessian_25
    H_31 = (1/len(StandardNormal)) * yHessian_31
    H_32 = (1/len(StandardNormal)) * yHessian_32
    H_33 = (1/len(StandardNormal)) * yHessian_33
    H_34 = (1/len(StandardNormal)) * yHessian_34
    H_35 = (1/len(StandardNormal)) * yHessian_35
    H_41 = (1/len(StandardNormal)) * yHessian_41
    H_42 = (1/len(StandardNormal)) * yHessian_42
    H_43 = (1/len(StandardNormal)) * yHessian_43
    H_44 = (1/len(StandardNormal)) * yHessian_44
    H_45 = (1/len(StandardNormal)) * yHessian_45
    H_51 = (1/len(StandardNormal)) * yHessian_51
    H_52 = (1/len(StandardNormal)) * yHessian_52
    H_53 = (1/len(StandardNormal)) * yHessian_53
    H_54 = (1/len(StandardNormal)) * yHessian_54
    H_55 = (1/len(StandardNormal)) * yHessian_55
    H = np.array([[H_11,H_12,H_13,H_14,H_15],
                 [H_21,H_22,H_23,H_24,H_25],
                 [H_31,H_32,H_33,H_34,H_35],
                 [H_41,H_42,H_43,H_44,H_45],
                 [H_51,H_52,H_53,H_54,H_55]])
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    LearningRate = 0.01 # Not sure how to choose this value
    ParameterList = [1, LearningRate/np.linalg.norm(Beta)]
    return Beta * min(ParameterList) # min(ParameterList) can be understood as similar to the "Proportion" in gradient descent

def u(x, Beta):
    return (((x[0] ** 2) + (x[1] ** 2)) / 2) + Beta[0] * PotentialF_1.giulio_F(x) + Beta[1] * PotentialF_2.gaussian_F(x) + Beta[2] * PotentialF_3.multiquadric_F(x) + Beta[3] * PotentialF_4.inverseQuadratic_F + Beta[4] * PotentialF_5.inverseMultiquadric_F(x)

def uConjugate(y, Beta):
    ConvexCandidate = []
    for i in range(0, len(MixtureSample)):
        ConjugateValue = np.dot(MixtureSample[i], y) - u(MixtureSample[i], Beta)
        ConvexCandidate.append(ConjugateValue)
    return max(ConvexCandidate)

def D(Beta):
    xSummation = 0
    ySummation = 0
    for i in range(0, len(MixtureSample)):
        xSummation += u(MixtureSample[i], Beta)
    for j in range(0, len(StandardNormal)):
        ySummation += uConjugate(StandardNormal[j], Beta)

    LL = 1/len(MixtureSample) * xSummation + 1 / \
        len(StandardNormal) * ySummation

    return LL



def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    for i in range(0, len(OldMixtureSample)):
        xval = OldMixtureSample[i][0] + Beta[0] * nd.Gradient(PotentialF_1.giulio_F)(OldMixtureSample[i])[0] + Beta[1] * nd.Gradient(PotentialF_2.gaussian_F)(OldMixtureSample[i])[0] + Beta[2] * nd.Gradient(PotentialF_3.multiquadric_F)(OldMixtureSample[i])[0] + Beta[3] * nd.Gradient(PotentialF_4.inverseQuadratic_F)(OldMixtureSample[i])[0] + Beta[4] * nd.Gradient(PotentialF_5.inverseMultiquadric_F)(OldMixtureSample[i])[0]
        yval = OldMixtureSample[i][1] + Beta[0] * nd.Gradient(PotentialF_1.giulio_F)(OldMixtureSample[i])[1] + Beta[1] * nd.Gradient(PotentialF_2.gaussian_F)(OldMixtureSample[i])[1] + Beta[2] * nd.Gradient(PotentialF_3.multiquadric_F)(OldMixtureSample[i])[1] + Beta[3] * nd.Gradient(PotentialF_4.inverseQuadratic_F)(OldMixtureSample[i])[1] + Beta[4] * nd.Gradient(PotentialF_5.inverseMultiquadric_F)(OldMixtureSample[i])[1]
        NewMixtureSample.append([xval,yval])
    NewMixtureSample = np.array(NewMixtureSample)
    return NewMixtureSample

def MixtureSampleGenerator():
    mean1 = [1, -1]
    cov1 = [[0.5, 0], [0, 0.5]]
    mean2 = [-1, 1]
    cov2 = [[0.5, 0], [0, 0.5]]
    x = np.random.multivariate_normal(mean1, cov1, 100)
    y = np.random.multivariate_normal(mean2, cov2, 100)
    MixtureSample = []
    for i in range(100):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator():
    Sample = []
    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    for i in range(100):
        Sample.append([x[i], y[i]])
    return Sample


#------------------------------------------------------------------ TESTING (change to heatmap, add animtation)------------------------------------------------------------
MixtureSample = MixtureSampleGenerator()
StandardNormal = StandardNormalGenerator()
CenterGeneratorList = MixtureSample + StandardNormal

PotentialF_1 = family_of_f.PotentialF()
PotentialF_2 = family_of_f.PotentialF()
PotentialF_3 = family_of_f.PotentialF()
PotentialF_4 = family_of_f.PotentialF()
PotentialF_5 = family_of_f.PotentialF()


plt.subplot(1,3,3)
plt.title("Target")
plt.scatter(*zip(*StandardNormal), color = 'r', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.subplot(1,3,1)
plt.title("Initial")
plt.scatter(*zip(*MixtureSample), color = 'b', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

DValue = 0
Iteration = 0
while True: # Maybe there is a problem of overfitting
    #print("Iteration " + str(i))
    Iteration += 1
    CenterList = []
    for i in range(0,5):
        CenterList.append(CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)])
    PotentialF_1.set_center(CenterList[0])
    PotentialF_2.set_center(CenterList[1])
    PotentialF_3.set_center(CenterList[2])
    PotentialF_4.set_center(CenterList[3])
    PotentialF_5.set_center(CenterList[4])
    PotentialF_1.set_alpha(0.75)
    PotentialF_2.set_alpha(0.75)
    PotentialF_3.set_alpha(0.75)
    PotentialF_4.set_alpha(0.75)
    PotentialF_5.set_alpha(0.75)
    PotentialF_2.set_constant(1)
    PotentialF_3.set_constant(1)
    PotentialF_4.set_constant(1)
    PotentialF_5.set_constant(1)
    Beta = BetaNewton()
    print(Beta)
    OldD = DValue
    DValue = D(Beta)
    print(DValue)
    MixtureSample = SamplesUpdate(MixtureSample)
    if abs(DValue - OldD) < 0.0001 or Iteration > 25:
        break

plt.subplot(1,3,2)
plt.title("Optimal Transport")
plt.scatter(*zip(*MixtureSample), color = 'g', alpha = 0.2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
