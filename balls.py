import numpy as np
from particle import Particle
from typing import List
import time

class MonoAtomicGasParticle(Particle):
    def __init__(self, n, *args, **kwargs):
        """
        A class describing a normal monoatomic gas particle such as argon or helium
        """
        self.n = n

        super().__init__(*args, **kwargs)

    def forces(self, p):
        """
        Calculate the Lennard-Jones potential for two particles 

        Parameters:
        p (MonoAtomicGasParticle): The other particle to calculate the forces ageinst
        """
        # Set cutoff distance for Lennard-Jones
        if abs(self.dist(p)) < 3:
            # Apply the Lennard-Jones potential
            self.apply_force(p, lambda p1, p2 : 48 * ((1/p1.dist(p2)**13) - 0.5 * (1/p1.dist(p2)**7)))


class IntraMolecularParticle(Particle):
    def __init__(self, n: int, connected: List[int], q: int, *args, **kwargs):
        """
        A class describing an intra molecular particle such as a particle in a bigger molecule

        Parameters:
        n (int): The id of the particle
        connections (List[int]): A list of its connected particles
        q (int): The charge of the particle
        """
        self.n = n
        self.connected = connected
        self.q = q

        super().__init__(*args, **kwargs)

    def forces(self, p):
        """
        Calculate the Lennard-Jones potential, covelent bond force and columb potential for two particles 

        Parameters:
        p (IntraMolecularParticle): The other particle to calculate the forces ageinst
        """
        # Set cutoff distance for Lennard-Jones
        if abs(self.dist(p)) < 4:
            # Apply the Lennard-Jones potential
            self.apply_force(p, lambda p1, p2: 48 * ((1/p1.dist(p2)**13) - 0.5 * (1/p1.dist(p2)**7)))

        # Covelent bond forces
        if p.n in self.connected or self.n in p.connected:
            # f = k * (r_opt - r_ij) - dist
            self.apply_force(p, lambda p1, p2: 900*((p1.r+p2.r)-p1.dist(p2)))

        # Columb forces
        # f = k * (q1*q2)/r_ij^2
        self.apply_force(p, lambda p1, p2: (332 * (p1.q * p2.q)/p1.dist(p2)**2))
