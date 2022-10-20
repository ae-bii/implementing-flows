import math
import numpy as np
from scipy.special import expi
from scipy.special import erf
from torch import alpha_dropout

e = math.e
pi = math.pi



def distance(z, center):
    return np.sqrt(sum(np.square(np.subtract(z,center))))

def DistanceVec(z, center):
    DistanceList = z - center
    DistanceSquared = np.square(DistanceList[:,0]) + np.square(DistanceList[:,1])
    return np.sqrt(DistanceSquared)



import math
import numpy as np
from scipy.special import expi
from scipy.special import erf
from torch import alpha_dropout

e = math.e
pi = math.pi



def distance(z, center):
    return np.sqrt(sum(np.square(np.subtract(z,center))))

def DistanceVec(z, center):
    DistanceList = z - center
    dim = np.shape(DistanceList)[1]
    DistanceSquared = np.zeros(len(DistanceList[:,1]))
    for i in range(dim):
        DistanceSquared += np.square(DistanceList[:,i])
    return np.sqrt(DistanceSquared)



class Bump_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        if r < 1/self._alpha:
            r2 = np.square(r)
            return (np.exp(1 / (self._alpha**2 * r2 - 1)) * (self._alpha**2 * r2 - 1) - expi(1 / (self._alpha**2 * r2 - 1))) / (2 * self._alpha**2) + self._constant
        return r + self._constant

class Giulio_F:
    def __init__(self, alpha=1):
        self._alpha = alpha
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return np.multiply(r, erf(r/self._alpha)) + (self._alpha/math.sqrt(pi)) * np.exp(-1 * np.square(r/self._alpha))

class Giulio_F_Vectorized:
    def __init__(self, alpha = 1):
        self._alpha = alpha
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return np.multiply(r, erf(r/self._alpha)) + (self._alpha/math.sqrt(pi)) * np.exp(-1 * np.square(r/self._alpha))

class Gaussian_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return -np.exp(-self._alpha**2 * np.square(r)) / (2 * self._alpha**2) + self._constant

class Gaussian_F_Vectorized:
    def __init__(self, alpha = 1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return -np.exp(-self._alpha**2 * np.square(r)) / (2 * self._alpha**2) + self._constant 

class Multiquadric_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return np.power(self._alpha**2 * np.square(r) + 1, 3/2) / (3 * self._alpha**2) + self._constant

class Multiquadric_F_Vectorized:
    def __init__(self, alpha = 1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return np.power(self._alpha**2 * np.square(r) + 1, 3/2) / (3 * self._alpha**2) + self._constant

class InverseQuadratic_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return np.log(self._alpha**2 * np.square(r) + 1) / (2 * self._alpha**2) + self._constant

class InverseQuadratic_F_Vectorized:
    def __init__(self, alpha = 1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return np.log(self._alpha**2 * np.square(r) + 1) / (2 * self._alpha**2) + self._constant

class InverseMultiquadric_F:
    def __init__(self, alpha=1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return np.sqrt(self._alpha**2 * np.square(r) + 1) / (self._alpha**2) + self._constant

class InverseMultiquadric_F_Vectorized:
    def __init__(self, alpha = 1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return np.sqrt(self._alpha**2 * np.square(r) + 1) / (self._alpha**2) + self._constant

class PolyharmonicSpline_F:
    def __init__(self, constant=0, k=0):
        self._k = k
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        if self._k % 2 == 0:
            # Since k and r are positive
            return (np.multiply(np.power(r,self._k+1), (self._k+2) * np.log(r) - 1)) / ((self._k+2)**2) + self._constant
        else:
            return np.power(r,self._k+2) / (self._k+2) + self._constant

class PolyharmonicSpline_F_Vectorized:
    def __init__(self, constant=0, k=0):
        self._k = k
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        if self._k % 2 == 0:
            # Since k and r are positive
            return (np.multiply(np.power(r,self._k+1), (self._k+2) * np.log(r) - 1)) / ((self._k+2)**2) + self._constant
        else:
            return np.power(r,self._k+2) / (self._k+2) + self._constant


class ThinPlateSpline_F:
    def __init__(self, constant=0):
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = distance(z, self._center)
        return np.power(r,4) * (np.log(r)/4 - 1/16) + self._constant

class ThinPlateSpline_F_Vectorized:
    def __init__(self, constant=0):
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z):
        r = DistanceVec(z, self._center)
        return np.power(r,4) * (np.log(r)/4 - 1/16) + self._constant


class Gaussian_fgrad_Vectorized:
    def __init__(self, constant=0, alpha=1):
        self._constant = constant
        self._alpha = alpha
    def setCenter(self, center):
        self._center = center
    def __call__(self, z, fnum, wrt):
        r = DistanceVec(z, self._center)
        if fnum == wrt:
            return np.longdouble(-2 * (self._alpha)**2 * (z[:,wrt] - self._center[wrt]) ** 2 * np.exp(-self._alpha**2 * np.square(r)) + np.exp(-self._alpha**2 * np.square(r)))
        else:
            return  np.longdouble(-2 * (self._alpha)**2 * (z[:,wrt] - self._center[wrt]) * (z[:,fnum] - self._center[fnum]) * np.exp(-self._alpha**2 * np.square(r)))

class Giulio_fgrad_alpha_is_1: # The expresion of this function may not be correct. 
    def __init__(self, alpha=1):
        self._alpha = alpha
    def setCenter(self, center):
        self._center = center
    def __call__(self, z, fnum, wrt):
        r = DistanceVec(z, self._center)
        if fnum == wrt:
            return np.longdouble((2 * z[:,wrt] * np.exp(-(r ** 2)))/(np.sqrt(pi) * (r ** 2))) - np.multiply((-(r ** 2) + (z[:,wrt] ** 2)), (erf(r)/(r ** 3)))
        else:
            return np.longdouble(np.multiply((np.multiply(z[:,fnum], z[:,wrt])), ((2 * np.exp(-(r ** 2)))/((np.sqrt(pi) * (r ** 2)) - erf(r)/(r ** 3))))) 
    
class InverseQuadratic_fgrad_alpha_is_1:
    def __init__(self, alpha = 1, constant=0):
        self._alpha = alpha
        self._constant = constant
    def setCenter(self, center):
        self._center = center
    def __call__(self, z, fnum, wrt):
        r = DistanceVec(z, self._center)
        if fnum == wrt:
            return np.longdouble(-((-(r ** 2) + (2 * z[:,wrt]) - 1)/(((r ** 2) + 1) ** 2)))
        else:
            return np.longdouble(-((2 * np.multiply(z[:,fnum], z[:,wrt]))/(((r ** 2) + 1) ** 2)))    