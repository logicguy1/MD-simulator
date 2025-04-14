from multiprocessing import Pool, cpu_count
from typing import List
import numpy as np
import random
import math
import time

from particle import Particle
from balls import MonoAtomicGasParticle, IntraMolecularParticle
from write_data import write_data

def calc_forces_threaded(*args):
    p1, p2 = args[0]
    for i in p1:
        for j in p2:
            if i.n < j.n:
                i.forces(j)

    return p1

class System:
    def __init__(
            self, 
            size: float, 
            dt: float, 
            simcount: int, 
            T_start: float=90, 
            T_stop: float=90, 
            particles: List[Particle] = [], 
            gravaty: float = 0, 
            filename: str="output.dump", 
            freq: int=100
        ):
        """
        Particle system manager, this class with handle the particles and manage temepreture and collisions

        Parameters:
        size (float): The size of the simulation, this will be cubed, so s*s*s
        dt (float): The timestep of the simulation
        simcount (int): The amount of iterations the simulation is expected to run
        T_start (float): The start temepreture
        T_stop (float): The final temepreture
        particles (list): A list of particles to simulate
        gravaty (float): The gravaty of the system
        filename (str): The filename the simulation will be saved under
        freq (int): The save frequency, based on cycle counts
        """
        self.size = size
        self.dt = dt
        self.T_start = T_start
        self.T_stop = T_stop
        self.particles = particles
        self.gravaty = gravaty
        self.simcount = simcount
        self.simstep = 0

        # Rendering based parameters
        self.filename = filename
        self.freq = freq
        self.ts_start = None
        self.progress_length = 30
        
    def step(self):
        """
        Run the velo verlet solver as well as any other step related code
        """
        self.simstep += 1
        if self.ts_start is None:
            self.ts_start = time.time()
        
        # Update position using velo verlet
        for p in self.particles:
            p.move(self.dt)

        # Reset forces and apply gravaty
        for p in self.particles:
            p.f_old = p.f
            p.f = np.zeros(3)
            p.f[2] = -self.gravaty

        # Calculate forces
        num_particles = len(self.particles)
        if num_particles < 200:
            for i in range(num_particles):
                for j in range(i + 1, num_particles):
                    p1 = self.particles[i]
                    p2 = self.particles[j]
                    p1.forces(p2)
        else:
            cpus = cpu_count()
            particle_split = num_particles // cpus

            p = Pool(processes=cpus)
            self.particles = [item for sublist in p.map(
                calc_forces_threaded, 
                [(self.particles[particle_split*i:particle_split*i+particle_split], self.particles) for i in range(cpus)]
            ) for item in sublist]

        # Collisions
        self.collision()

        # Calculate velocities
        for p in self.particles:
            p.v += 0.5 * self.dt * (p.f_old + p.f)

        # Temperature
        if self.T_start != -1 or self.T_stop != -1:
            t = self.simstep / self.simcount  # Liniar temperature lerp
            temp = (1-t) * self.T_start + t * self.T_stop
            self.set_temperature(temp)
        else:
            temp = -1

        # Save sim
        if not self.simstep % self.freq:
            self.save_sim(temperature=temp)

    def collision(self):
        """
        Check for collisions between the wall of the system as well as between other particles
        """
        # Get position and velocity vectors
        positions = np.array([p.p for p in self.particles])
        velocities = np.array([p.v for p in self.particles])
        forces = np.array([p.f for p in self.particles])
        # Create an array marking every out of bounds position
        out_of_bounds = (positions < 0) | (positions > self.size)
        # Ajust the velocities for out of bounds particles
        velocities[out_of_bounds] *= -1
        forces[out_of_bounds] *= -1
        # Apply the new values
        for i, p in enumerate(self.particles):
            if 1 in out_of_bounds[i]:
                p.v = velocities[i]
                p.f = forces[i]
                p.move(self.dt)

        # This code is technically not neccesarry as the lennard jones potential handles collisions with a small dt
        """
        # Check collisoin between particles
        num_particles = len(self.particles)
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                p1 = self.particles[i]
                p2 = self.particles[j]

                if p1.dist(p2) < p1.r + p2.r:
                    v1_old = p1.v 
                    v2_old = p2.v 

                    # Calculate new velocities from collision using mass
                    p1.v = ((p1.m-p2.m)/(p1.m+p2.m))*v1_old + ((2*p2.m)/(p1.m+p2.m))*v2_old
                    p2.v = ((2*p1.m)/(p1.m+p2.m))*v1_old    + ((p2.m-p1.m)/(p1.m+p2.m))*v2_old

                    p1.move(self.dt)
                    p2.move(self.dt)
        """

    def set_temperature(self, T_desired: float):
        """
        Sets the temperature of the System

        Parameters:
        T_desired (float): The temperature the system should have in kelvin
        """
        if T_desired == -1:
            return # Break the function

        v = np.abs(np.array([p.v for p in self.particles]))

        # Set mass
        mass = 1.6735575*10**(-27)
        boltzmann = 1.380649*10**(-23)
        # First measure the temperature
        T_actual = 2/3 * mass * np.sum(np.mean(v, axis=0)) / boltzmann * 25 * 10**5 
        # Find correction factor
        c = np.sqrt(T_desired / T_actual)

        for p in self.particles:
            p.v *= c

    def save_sim(self, temperature: float = 0):
        """
        Function to save the simulation state in a OVITO compatible file format (LAMMPS dump)
        """

        # Print progress meter
        progress = self.simstep / self.simcount
        decimal, integer = math.modf(progress*self.progress_length) 
        bar = "#" * (int(integer)) 
        bar += "/" if decimal > 0.5 else "."
        bar += "." * max(self.progress_length - int(integer) - 1, 0)
        elapsed = int(time.time() - self.ts_start)

        print(f"\r{self.simstep} {bar} {self.simcount} | {elapsed}s ({round(elapsed / self.simstep,3)}s) : {round(elapsed / self.simstep * self.simcount)}s", end="", flush=True)

        particles = []
        for p in self.particles:
            p_data = {
                "v_x": p.v[0], 
                "v_y": p.v[1], 
                "v_z": p.v[2], 
                "f_x": p.f[0], 
                "f_y": p.f[1], 
                "f_z": p.f[2], 
                "radius": p.r, 
                "mass": p.m,
                "x": p.p[0], 
                "y": p.p[1], 
                "z": p.p[2], 
                "temperature": temperature,
            }

            if isinstance(p, IntraMolecularParticle):
                p_data["q"] = p.q,
                p_data["ColorR"] = "55" if p.q == 1 else "0"   if p.q == -1 else "55",
                p_data["ColorG"] = "0"   if p.q == 1 else "55" if p.q == -1 else "55",
                p_data["ColorB"] = "0"   if p.q == 1 else "0"   if p.q == -1 else "55",

            particles.append(p_data)
        write_data(
            filename=self.filename, 
            ts=self.simstep, 
            bounds=self.size, 
            particles=particles
        )

def generate_random_particles(num, size, radius):
    particles = []

    idx = 0
    while len(particles) < num:
        p = MonoAtomicGasParticle(
            n=idx,
            p=[random.randint(0,size),random.randint(0,size),random.randint(0,size)],
            v=[random.randint(0,200)/100-1,random.randint(0,200)/100-1,random.randint(0,200)/100-1],
            m=1,
            r=radius
        )

        overlap = False
        for i in particles:
            if p.dist(i) + 0.2 <= radius:
                overlap = True 
                break

        if not overlap:
            particles.append(p)
            idx += 1

    return particles

def generate_protine(num):
    particles = []

    for i in range(num):
        particles.append(IntraMolecularParticle(
            n=i,
            connected=[i+1],
            q=np.random.choice([-1,0,1], p=[0.25,0.5,0.25]),
            p=[5,i+1,5],
            v=[random.randint(0,200)/100-1,random.randint(0,200)/100-1,random.randint(0,200)/100-1],
            m=1,
            r=0.4,
        ))

    return particles

if __name__ == "__main__":
    SIZE = 20
    SIMCOUNT = 25000
    PARTICLES = 500
    RADIUS = 0.4


    with open("outputs/output.dump", "w") as file:
        file.write("")


    # Generate randomly positioned particles
    particles = generate_random_particles(PARTICLES, SIZE, RADIUS)

    # Generate a string of particles forming a protine
    #particles = generate_protine(PARTICLES)

    # Custom particle positioning, need to design particle import system
    """
    particles = [
        MonoAtomicGasParticle(
            p=[4,4,4],
            v=[-1]*3,
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[1,1,1],
            v=[1,1,1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[1,1,4],
            v=[1,1,-1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[1,4,1],
            v=[1,-1,1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[4,1,1],
            v=[-1,1,1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[4,4,1],
            v=[-1,-1,1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[1,4,4],
            v=[1,-1,-1],
            m=1,
            r=0.4,
        ),
        MonoAtomicGasParticle(
            p=[4,1,4],
            v=[-1,1,-1],
            m=1,
            r=0.4,
        )
    ]
    """

    print("Startign")

    s = System(
        size=SIZE,
        dt=0.001,
        simcount=SIMCOUNT,
        particles=particles[::1],
        freq=5,
        T_start=200,
        T_stop=200,
        filename="outputs/output.dump"
    )

    for i in range(SIMCOUNT):
        s.step()
