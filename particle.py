import numpy as np
from typing import List, Callable

class Particle:
    def __init__(self, p: List[float], v: list[float], m: float, r: float):
        """
        Base particle class
        
        Parameters:
        p (list): A vector repersenting the position in 3d space
        v (list): A vector repersenting the velocety components in 3d space
        m (float): The mass of the particle
        r (float): The radius of the particle
        """

        self.p=np.array(p, dtype=float)
        self.v=np.array(v, dtype=float)
        self.m = m 
        self.r = r 
        self.f = np.zeros(3)
        self.f_old = np.zeros(3)

    def __repr__(self):
        #return "<Particle " + ", ".join([str(i) for i in self.p]) + ">"
        return "<Particle>"

    def dist(self, particle: 'Particle'):
        """
        Calculate the Euclidean distance between corresponding points in two 3D numpy arrays.

        Parameters:
        array1 (np.ndarray): A numpy array of shape (n, 3).
        array2 (np.ndarray): A numpy array of shape (n, 3).

        Returns:
        np.ndarray: A 1D array containing the Euclidean distances.
        """
        if self.p.shape != particle.p.shape:
            raise ValueError("Both arrays must have the same shape.")
        
        squared_diff = np.square(self.p - particle.p)
        sum_squared_diff = np.sum(squared_diff)
        distances = np.sqrt(sum_squared_diff)
        
        return distances

    def angle(self, particle: 'Particle'):
        """
        Decomposes a 3D vector into its magnitude, azimuthal angle, and elevation angle in radients.
        
        Parameters:
            v (array-like): A 3D vector.
            
        Returns:
            tuple: (magnitude, azimuthal angle (alpha), elevation angle (beta))
        """
        # Ensure the vector is a numpy array
        r = self.p - particle.p
        r_unit = r / np.linalg.norm(r)
        alpha = np.arctan2(r_unit[1], r_unit[0])
        beta = np.arccos(r_unit[2])
        return r, alpha, beta
        

    def move(self, dt: float):
        """
        Use newtons law of motion to move the particle a timestep
        d = v*t + 1/2*a*t^2 
        Use fores instead of accselleration, use f=ma => a=f/m
        d = v*t + 1/2*(f/m)*t^2 

        Parameters:
        dt (float): The timestep
        """
        self.p += self.v*dt + 0.5*(self.f/self.m)*dt**2

    def forces(self, p):
        ...

    def apply_force(self, p: 'Particle', force: Callable):
        """
        Applys a force between two particles, given a force function that returns the size of the force.
        This function will automatically split the force into its x, y and z components

        Parameters:
        p (Particle): The other particle that the force should be calculated ageinst
        force (Callable): A function given two particles returns a scalar value of the size of the force
        """
        r = self.dist(p)
        _, alpha, beta = self.angle(p)

        f = force(self, p)

        # Calculate components of the force
        f_y = np.sin(beta) * np.sin(alpha) * f
        f_x = np.sin(beta) * np.cos(alpha) * f
        f_z = np.cos(beta) * f

        # Update forces on both particles
        self.f += np.array([
            f_x, 
            f_y, 
            f_z 
        ])
        p.f -= np.array([
            f_x, 
            f_y, 
            f_z 
        ])


if __name__ == "__main__":
    p1 = Particle(
        p=[2,1,0],
        v=[1,1,0],
        m=1,
        r=1
    )

    p2 = Particle(
        p=[1,1,2],
        v=[1,1,0],
        m=1,
        r=1
    )

    print(p1.dist(p2))
    print(p1.angle(p2))

    print(p1)
    p1.move(1)
    print(p1)


