from ast import Del, Num
from msilib.schema import Error
import sys
from turtle import color

sys.path.append("./")
from statistics import median
import numpy as np
import time
import random
import math
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import functions
import SampleGeneratorND
import torch
import MMDFunctions
import sympy as sp
from scipy.interpolate import griddata

random.seed(0)
np.random.seed(0)

Start = time.time()

e = math.e
pi = math.pi

def distance(z1, z2):
    return np.sqrt(sum(np.square(np.subtract(z1,z2))))

def norm(i):
    return np.sqrt(sum(np.square(i)))

def GradientApprox(VariableList): # Calculate the gradient of RBFs in BetaNewton using finite difference method (central) at each point in the VariableList
    Gradient = []
    delta = 1e-5
    for f in range(NumFs):
        grad = [] # The gradient is calculated for each function
        for i in range(dim): # And the partial derivative is calculated w.r.t. to each variable
            temp = np.zeros(dim)
            temp[i] = delta/2 # Add a small number at the ith variable (This is the h/2 in the formula (f(x+h/2) - f(x-h/2))/h)
            grad.append(np.longdouble((PotentialFsVectorized[f](VariableList + temp) - (PotentialFsVectorized[f](VariableList - temp)))/delta)) # Apply the formula above to get the gradient 
        Gradient.append(np.squeeze(np.transpose(grad))) 
        # Gradient has "NumFs" sublists, each has a length equal to len(VariableList), which corresponding to the gradient of each RBF evaluated at every point in VariableList. 
        # It needs to be transposed to be used in BetaNewton

    return Gradient

def JacobianAnalytical(VariableList): # Calculate the Jacobians at each point given by VariableList

    Partials = [] # Space reserved for partial derivatives
    
    for m in range(NumFs): # Finding the Jacobian for each function w.r.t. to each variable
        for n in range(dim):
                if m == n: # Since the function takes the form x_n + sum(F_m(x)) for each n, we know dx_n/dx_n = 1 and dx_m/dx_n = 0 for all m not equal to n 
                    Partials.append(np.longdouble(1 + sum([Beta[f] * PotentialPartialVectorized[f](VariableList, fnum=m,wrt=n) for f in range(NumFs)])))
                else: 
                    Partials.append(np.longdouble(sum([Beta[f] * PotentialPartialVectorized[f](VariableList, fnum=m,wrt=n) for f in range(NumFs)])))
    Jacobian = np.transpose(np.array(Partials)) # Partials has m * n sublists, each sublist contains the gradients at all points in VariableList calculated from the mth function and w.r.t. to the nth variable.
    Jacobian = Jacobian.reshape(len(Initial),NumFs,dim) # Reshape to a list of 2 * 2 matrices, each corresponds to one point in VariableList
    for i in range(len(Initial)):
        Jacobian[i][:,0] = np.array(([1] + [0 for l in range(dim - 1)]),dtype=np.longdouble) # Maunally set the first column of the Jacobians to [1,0] because we are using a block-triangular method, thus the  x-value of each point is unchanged.

    return Jacobian
    



def JacobianApprox(VariableList): # Not used

    delta = 1e-3
    def Vals(VariableList, DimNeeded):
        NewMixtureSample = []
        vals = [VariableList[:,0]]
        F_eval = [[None] * NumFs for i in range(dim)]
        for f in range(0,NumFs):
            gradient = GradientApprox(VariableList)[f]
            for i in range(dim):
                F_eval[i][f] = (gradient[:,i])
        F_eval = np.array(F_eval)
        for i in range(1,dim):
            vals.append(VariableList[:,i] + (np.multiply(np.transpose(F_eval[i]), Beta)).sum(axis = 1))
        NewMixtureSample = np.array(vals, dtype=np.longdouble)
        return NewMixtureSample[DimNeeded]
        
    def PartialGradient(func, DimNeeded, funcPara):
        DeltaVector = [0 for i in range(dim)]
        DeltaVector[DimNeeded] += delta/2
        return ((func((VariableList + DeltaVector), funcPara) - (func((VariableList - DeltaVector), funcPara)))/delta)

    Partials = []

    for m in range(dim):
        for n in range(dim):
            Partials.append(PartialGradient(Vals, m, n))

    Jacobian = np.transpose(np.array(Partials))

    return Jacobian.reshape(len(Initial),dim,dim)


def BetaNewton(): # Newton's method (Experimental)

    # Initialize variables
    xSummationGradient = np.zeros(NumFs) 
    ySummationGradient = np.zeros(NumFs) 
    G = np.zeros(NumFs)
    # Initialize variables end

    # Formula (54), Tabak-Trigila
    xSummationGradient = [sum(PotentialFsVectorized[f](Initial)) for f in range(NumFs)] 
    ySummationGradient = [sum(PotentialFsVectorized[f](Target)) for f in range(NumFs)] 
    G = [(1/len(Initial)) * xSummationGradient[k] - (1/len(Target)) * ySummationGradient[k] for k in range(NumFs)] 
    # Formula (54), Tabak-Trigila end

    G = np.array(G)

    yHessian = np.zeros([NumFs,NumFs]) # Initialize Hessian
    F_gradient = GradientApprox(Target) # Calculate the gradient at every point in the target samples, for each RBF

    # Formula (55), Tabak-Trigila 
    for m in range(0, NumFs):
        for n in range(0, NumFs):
            yHessian[m][n] = sum((F_gradient[m]*F_gradient[n]).sum(axis = 1)) # Summing up the product of gradients of each pair of RBF at every point 
    H = np.multiply(yHessian, 1/len(Target)) # Divide by 1/n
    # Formula (55), Tabak-Trigila end

    # Formula (51), Tabak-Trigila
    HInverseNeg = (-1) * np.linalg.inv(H)
    Beta = np.matmul(HInverseNeg, G)
    # Formula (51), Tabak-Trigila end

    LearningRate = 1 # Not sure how to choose this value
    return Beta/norm(Beta) * LearningRate

def u(x, Beta): # Not used
    F_eval = [(PotentialFs[f](x)) for f in range(NumFs)]
    return (np.dot(x,x) / 2) + np.dot(Beta, F_eval)

def uVectorized(x):
    F_eval = [PotentialFsVectorized[f](x) for f in range(NumFs)]
    return np.array([np.dot(x[z],x[z]) for z in range(len(x))])/2 + (np.transpose(F_eval) * Beta).sum(axis = 1)

def uConjugate(y, Beta): # Not used
    temp = Initial * y
    sum = np.zeros(len(temp))
    for i in range(dim):
        sum += temp[:,i]
    ConvexVector = sum - uVectorized(Initial)
    return max(ConvexVector)

def uConjugateVec(y):
    ConvexMatrix = np.array(np.matmul(y, np.transpose(Initial))) - uVectorized(Initial)
    return ConvexMatrix.max(axis = 1)

def D():
    xSummation = sum(uVectorized(Initial))
    ySummation = sum(uConjugateVec(Target))

    LL = 1/len(Initial) * xSummation + 1 / \
        len(Target) * ySummation
    return LL

def SamplesUpdate(OldMixtureSample):
    NewMixtureSample = []
    vals = [OldMixtureSample[:,0]]
    F_eval = [[None] * NumFs for i in range(dim)]
    for f in range(0,NumFs):
        gradient = GradientApprox(OldMixtureSample)[f]
        for i in range(dim):
            F_eval[i][f] = (gradient[:,i])
    F_eval = np.array(F_eval)
    for i in range(1,dim):
        vals.append(OldMixtureSample[:,i] + (np.multiply(np.transpose(F_eval[i]), Beta)).sum(axis = 1))
    NewMixtureSample = np.array(vals)
    NewMixtureSample = np.transpose(NewMixtureSample)

    return NewMixtureSample

def MixtureSampleGenerator(): # Not used
    mean1 = []
    mean2 = []
    mean3 = []
    for i in range(dim):
        mean1.append(random.random()*0.25)
        mean2.append(random.random()*0.25)
        mean3.append(random.random()*0.25)
    mean1, mean2, mean3 = np.array(mean1), np.array(mean2), np.array(mean3)
    cov = np.array([0.5]*dim)
    cov = np.diag(cov**dim)
    x = np.random.multivariate_normal(mean1, cov, len(Target))
    y = np.random.multivariate_normal(mean2, cov, len(Target))
    z = np.random.multivariate_normal(mean3, cov, len(Target))
    MixtureSample = []
    for i in range(len(Target)):
        RandomSelector = random.random()
        if RandomSelector > 0.7:
            MixtureSample.append(x[i])
        elif RandomSelector < 0.3:
            MixtureSample.append(z[i])
        else:
            MixtureSample.append(y[i])
    MixtureSample = np.array(MixtureSample)
    return MixtureSample

def StandardNormalGenerator(): # Not used
    Sample = []
    Normals = []
    for i in range(dim):
        Normals.append(np.random.standard_normal(500))
    for j in range(500):
        cur = []
        for k in range(dim):
            cur.append(Normals[k][j])
        Sample.append(np.array(cur))
    return np.array(Sample)

def SigCalculation(X,Y): # Not used
    Z = X + Y
    DistList = []
    for i in range(len(Z)):
        for j in range(len(Z)):
            DistList.append(functions.distance(Z[i],Z[j]))

    return median(DistList)

def JacobianApproxTest(VariableList,delta): # Temporary function used to test Jacobian, not used
    def Vals(VariableList, DimNeeded):
        NewMixtureSample = []
        vals = [VariableList[:,0]]
        F_eval = [[None] * NumFs for i in range(dim)]
        for f in range(0,NumFs):
            gradient = GradientApprox(VariableList)[f]
            for i in range(dim):
                F_eval[i][f] = (gradient[:,i])
        F_eval = np.array(F_eval)
        for i in range(1,dim):
            vals.append(VariableList[:,i] + (np.multiply(np.transpose(F_eval[i]), Beta)).sum(axis = 1))
        NewMixtureSample = np.array(vals, dtype=np.longdouble)
        return NewMixtureSample[DimNeeded]
        
    def PartialGradient(func, DimNeeded, funcPara):
        DeltaVector = [0 for i in range(dim)]
        DeltaVector[DimNeeded] += delta
        return (np.longdouble(func((VariableList + DeltaVector), funcPara) - (func((VariableList - DeltaVector), funcPara)))/(2 * delta))

    Partials = []

    for m in range(dim):
        for n in range(dim):
            Partials.append(PartialGradient(Vals, m, n))

    Jacobian = np.transpose(np.array(Partials))

    return Jacobian.reshape(len(Initial),dim,dim)

#------------------------------------------------------------------ TESTING ------------------------------------------------------------
dim = 2 # Dimension of samples

# Testing and Plot:
Target = np.loadtxt("old/2D/EllipseMoon.csv",dtype=np.float64,delimiter=",") # Currently using the moon sample
Initial = np.transpose([np.random.standard_normal(len(Target)),np.random.normal(0,2,len(Target))]) # x follows a standard normal distribution, y follows a normal distribution with standard deviation = 1

CenterGeneratorList = Target # Initialize the list of centers to the list of target samples
InitialSaved = Initial # Save a copy of initial samples

PotentialFs = [functions.Giulio_F(), # Not used
                functions.Gaussian_F(),
                functions.Multiquadric_F(),
                functions.InverseQuadratic_F(),
                functions.InverseMultiquadric_F(),
                functions.PolyharmonicSpline_F(),
                functions.ThinPlateSpline_F()]

PotentialFsVectorized = [functions.Gaussian_F_Vectorized(), # Vectorized RBFs
                         functions.InverseQuadratic_F_Vectorized()]
PotentialPartialVectorized = [functions.Gaussian_fgrad_Vectorized(), # The partial derivative expressions of vectorized RBFs
                              functions.InverseQuadratic_fgrad_alpha_is_1()]
NumFs = len(PotentialFsVectorized) # Number of RBFs being used

Jacobian = [[[1,0],[0,1]] for k in range(len(Initial))] # Initialize Jacobian

#Initialize values
DValue = 0
Iteration = 0
Beta = 0
SamplesSaved = []
MMDList = []
DistCenter = []
JacDetList = []
MinDetPoints = []
#Initialize values end

# Converts time in seconds to hours, minutes, seconds
# def time_convert(sec):
#   mins = sec // 60
#   sec = sec % 60
#   hours = mins // 60
#   mins = mins % 60
#   print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

# start_time = time.time()

for i in range(1000): # Maybe there is a problem of overfitting
    Iteration += 1 # Record the number of iterations. For later iterations, change the CenterGeneratorList to the list of all initial and target sample points
    if Iteration >= 10:
        CenterGeneratorList = np.concatenate((Initial,Target), axis=0)
    CenterList = []
    
    for f in range(0,NumFs):
        DistFlag = False # Test whether two centers are too close
        c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]
        while f > 0 and DistFlag == False:
            if all([functions.distance(c,CenterList[k]) >= 2 for k in range(0,f)])== True: # If too close, generate another center until distance between centers > 2
                DistFlag = True
            else:
                c = CenterGeneratorList[random.randint(0, len(CenterGeneratorList) - 1)]

        CenterList.append(c)
        PotentialFsVectorized[f].setCenter(c) # Once a center is chosen, set it to the center of a RBF and its partial derivatives
        PotentialPartialVectorized[f].setCenter(c) 

    if i > 0:
        DistCenter.append([functions.DistanceVec(np.array(CenterList), np.array(PreviousCenterList))])

    PreviousCenterList = CenterList

    OldBeta = Beta 
    Beta = BetaNewton() # Calculate the value of Beta (Formula (51), Tabak-Trigila)
    OldD = DValue
    DValue = D() # Calculate the value of D (Formula (43), Tabak-Trigila)
    Jacobian = np.matmul(JacobianAnalytical(Initial), Jacobian) # Multiplying Jacobians where JacobianAnalytical(Initial) is J_(n+1) and Jacobian is J_n 
    JacDetList.append(min(np.absolute(np.linalg.det(Jacobian)))) # Add the determinant at this stage to a list for plotting
    MinDetPoints.append(Initial[np.argmin(np.absolute(np.linalg.det(Jacobian)))]) # Add the minimum determinant to a list for plotting
    Initial = SamplesUpdate(Initial) # Update the initial sample points

JacDeterminant = np.linalg.det(Jacobian) # Calculate the overall determinant of the product of all jacobians

DistCenter = np.squeeze(np.array(DistCenter))


# Determinant Plot (Minumum absolutue value)
TimeStep = np.linspace(0,999,1000)
plt.plot(TimeStep,JacDetList)
plt.title("Determinant (Minimum absolute value) at each time step")
plt.show()

# Point where det are minimized
MinDetPoints = np.array(MinDetPoints)
plt.scatter(MinDetPoints[:,0],MinDetPoints[:,1],color="Red",label="Problematic points (All iterations)")
plt.scatter(Initial[:,0],Initial[:,1],color="Blue",label="Samples after transport")
plt.scatter(InitialSaved[:,0],InitialSaved[:,1],color="Green",label="Samples before transport")
plt.title("Point where minimum determinants occured")
plt.legend()
plt.show()


xVal = InitialSaved[:,0]
yVal = InitialSaved[:,1]
z = []

# x,y = sp.symbols("x y")
# MarginalX = sp.integrate((sp.exp(-(y ** 2)/2)/(sp.sqrt(2 * pi))) * ((sp.exp(-(1/2) * ((x - (y ** 2)) ** 2)))/(sp.sqrt(2 * pi))),(x,-sp.oo,sp.oo))
# MarginalY = sp.integrate((sp.exp(-(y ** 2)/2)/(sp.sqrt(2 * pi))) * ((sp.exp(-(1/2) * ((x - (y ** 2)) ** 2)))/(sp.sqrt(2 * pi))),(y,-sp.oo,sp.oo))

# Marginal distributions
MarginalX = lambda x: (1/np.sqrt(2 * pi)) * np.exp(-np.square(x)/2) 
MarginalY = lambda y: (1/np.sqrt(8 * pi)) * np.exp(-np.square(y)/8)

# Calculate the product of marginals at points (xVal,yVal), which, ideally, should be the density of the independent coupling at this point, i.e. rho(x) 
for i in range(len(xVal)):
    z = np.multiply(MarginalX(xVal),MarginalY(yVal))
    print(i)

z = np.divide(z,JacDeterminant) # Divide rho(x) by the determinant of jacobians
z = np.array(z,dtype="float64")

# Attempt to do some manual interpolation in order to make the plot looks better
xLin = np.linspace(-2.5,2.5,len(xVal))
yLin = np.linspace(-2.5,2.5,len(yVal))
zGrid = griddata((xVal,yVal),z,(xLin[None,:],yLin[:,None]),method="linear")




# z2 = SampleGeneratorND.JointBananaDensity(Initial[:,0],Initial[:,1])

#fig,axs = plt.subplots(1,2)
#ax0 = axs[0].contour(xLin,yLin,zGrid)
# ax1 = axs[1].tricontour(Initial[:,0],Initial[:,1],z2)
#plt.colorbar(ax0)
# plt.colorbar(ax1)
#plt.show()

plt.scatter(xVal,yVal,c=z)
c = plt.colorbar()
plt.show()

plt.subplot(1,3,1)
plt.scatter(InitialSaved[:,0],InitialSaved[:,1])
plt.subplot(1,3,2)
plt.scatter(Initial[:,0],Initial[:,1])
plt.subplot(1,3,3)
plt.scatter(Target[:,0],Target[:,1])
plt.show()