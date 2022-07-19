import math
import numpy as np

e = math.e
pi = math.pi

# TODO: Implement in a way so that functions can be used in higher dimensions too
class Potentialf:
    # Find differentiated version of bump_F
    def bump_f(z, center):
        return
    
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
    def bump_F(z, center):
        value = 0
        r = abs(z - center)
        alpha = 0.5
        if r < 1/alpha:
            value = math.pow(e, -(1/(1-(alpha*r)**2)))
        return value
    
    def giulio_F(z, center):
        r = abs(z - center)
        alpha = 1.5  # Consistent with the density 
        return r * math.erf(r/alpha) + (alpha/math.sqrt(pi)) * math.pow(e, -(r/alpha) ** 2)
    
    def gaussian_F(z, z_center, alpha, constant):
        r = abs(z - z_center)
        return -math.pow(e, -alpha**2 * r**2) / (2 * alpha**2) + constant
    
    def multiquadric_F(z, z_center, alpha, constant):
        return
    
    def inverseQuadratic_F(z, z_center, alpha, constant):
        return
    
    def inverseMultiquadric_F(z, z_center, alpha, constant):
        return
    
    def polyharmonicSpline_F(z, z_center, constant, k):
        return
    
    def thinPlateSpline_F(z, z_center, constant):
        return