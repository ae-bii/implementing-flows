import math
import numpy as np
from scipy.special import expi

e = math.e
pi = math.pi

# TODO: Implement in a way so that functions can be used in higher dimensions too
class Potentialf:
    def bump_f(z, center):
        value = 0
        r = abs(z - center)
        alpha = 0.5
        if r < 1/alpha:
            value = math.pow(e, -(1/(1-(alpha*r)**2)))
        return value
    
    def giulio_f(z, center):
        r = abs(z - center)
        alpha = 1.5
        return math.erf(r/alpha) / r
    
    def gaussian_f(z, z_center):
        r = abs(z - z_center)
        alpha = 1
        return math.pow(e, -((alpha*r)**2))
    
    def multiquadric_f(z, z_center):
        r = abs(z - z_center)
        alpha = 1
        return math.sqrt(1+(alpha*r)**2)
    
    def inverseQuadratic_f(z, z_center):
        r = abs(z - z_center)
        alpha = 1
        return 1/(1+(alpha*r)**2)
    
    def inverseMultiquadric_f(z, z_center):
        r = abs(z - z_center)
        alpha = 1
        return 1/math.sqrt(1+(alpha*r)**2)
    
    def polyharmonicSpline_f(z, z_center, k):
        r = abs(z - z_center)
        if k%2 == 0:
            return r**(k-1)*np.log(r**r) 
        else:
            return r**k
    
    def thinPlateSpline_f(z, z_center):
        r = abs(z - z_center)
        return r**2*np.log(r)
    
class PotentialF:
    def bump_F(z, center, constant):
        r = abs(z - center)
        alpha = 0.5
        if r < 1/alpha:
            return (math.exp(1 / (alpha**2 * r**2 - 1)) * (alpha**2 * r**2 - 1) - expi(1 / (alpha**2 * r**2 - 1))) / (2 * alpha**2) + constant
        return r + constant
    
    def giulio_F(z, center):
        r = abs(z - center)
        alpha = 1.5  # Consistent with the density 
        return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)
    
    
    def gaussian_F(z, z_center, alpha, constant):
        r = abs(z - z_center)
        return -math.pow(e, -alpha**2 * r**2) / (2 * alpha**2) + constant
    
    def multiquadric_F(z, z_center, alpha, constant):
        r = abs(z - z_center)
        return math.pow(alpha**2 * r**2 + 1, 3/2) / (3 * alpha**2) + constant
    
    def inverseQuadratic_F(z, z_center, alpha, constant):
        r = abs(z - z_center)
        return math.log(alpha**2 * r**2 + 1) / (2 * alpha**2) + constant
    
    def inverseMultiquadric_F(z, z_center, alpha, constant):
        r = abs(z - z_center)
        return math.sqrt(alpha**2 * r**2 + 1) / (alpha**2) + constant
    
    def polyharmonicSpline_F(z, z_center, constant, k):
        r = abs(z - z_center)
        if k%2 == 0:
            return (r**(k+1) * ((k+2) * math.log(r) - 1)) / ((k+2)**2)  + constant #Since k and r are positive
        else:
            return r**(k+2) / (k+2) + constant
    
    def thinPlateSpline_F(z, z_center, constant):
        r = abs(z - z_center)
        return r**4 * (math.log(r)/4 - 1/16) + constant