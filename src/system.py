import json
import time
import copy
import numpy as np

from particle import Particle
from video import Video

class System:
    def __init__(
        self, 
        kovelent_lines: bool = False, 
        lennard_jones_overlay: bool = False, 
        frame_rate: int= 30
    ):
        # Initial system parameters, gets overwritten after initialisation
        self.n_particles = 10
        self.width = 20
        self.dt = 0.001
        self.speed_scale = 1
        self.borders = False
        self.T = 90

        self.name = "" # Used in rendering
        self.simcount = 0

        # Rendering based settings
        self.kovelent_lines = kovelent_lines
        self.lennard_jones_overlay = lennard_jones_overlay
        self.frame_rate = frame_rate

        # Variables used internally
        self.particles = []
        self.psudo_particles = []

        self.video = Video()

    def __repr__(self):
        out = ""
        for y in range(self.width + 1):
            for x in range(self.width + 1):
                for p in self.particles:
                    if round(p.x) == x and round(p.y) == y:
                        out += "o"
                        break
                else:
                    out += " "
            out += "|\n|"

        return out

    def create_system(self):
        """Creates a perfect grid of particles in the system"""
        sqrt_npart = int(np.ceil(np.sqrt(self.n_particles)))
        n = 0

        for y in range(sqrt_npart):
            for x in range(sqrt_npart):
                self.particles.append(
                    Particle(
                        system = self,
                        n = n*2,
                        x = x / sqrt_npart * (self.width * 0.9) + self.width * 0.1,
                        y = y / sqrt_npart * (self.width * 0.9) + self.width * 0.1,
                        vx = self.speed_scale * 2 * (np.random.random()-0.5),
                        vy = self.speed_scale * 2 * (np.random.random()-0.5),
                        r = .5
                    )
                )
                n += 1

        self.particles = self.particles[:self.n_particles]
        self.generate_psudo()

    def load_system(self, filename):
        """Import a system from a json file"""
        with open(filename, "r") as file:
            data = json.load(file)

        self.name = data.get("name")
        self.n_steps = data.get("n_steps")
        self.dt = data.get("dt")
        self.width = data.get("size")
        self.T = data.get("T")
        self.borders = data.get("hideBorders")

        for i in data.get("particles") or []:
            self.particles.append(
                Particle(
                    system = self,
                    n         = i.get("n"),
                    x         = i.get("x"),
                    y         = i.get("y"),
                    vx        = i.get("vx"),
                    vy        = i.get("vy"),
                    r         = (i.get("r") or 1) / 2.5,
                    charge    = i.get("charge") or 0,
                    connected = i.get("connected") or []
                )
            )

    def generate_psudo(self):
        """Generate the psudo particles around the system"""
        psudo_particles = self.particles.copy()
        if not self.borders:
            self.psudo_particles = psudo_particles
            return
             
        # Create 8 boxes around main box to calulate everything from
        for y in range(3):
            for x in range(3):
                # Dont copy to center
                if y == 1 and x == 1: continue 

                for p in self.particles:
                    p = copy.copy(p)

                    if x == 0: p.x -= self.width
                    if x == 2: p.x += self.width

                    if y == 0: p.y -= self.width
                    if y == 2: p.y += self.width
                    
                    psudo_particles.append(p)

        self.psudo_particles = psudo_particles

    def collision(self, particle):
        """If two particles colide use newtons third law, this should never happen due to the lennard-jones potential"""
        psudo_particles = self.psudo_particles

        p = particle
        for i in [i for i in psudo_particles if i.n != p.n and p.get_dist(i) < self.width]:
            l = np.sqrt((i.x-p.x)**2 + (i.y-p.y)**2)
            if l < i.r + p.r and i.x < p.x:
                vi = (i.vx, i.vy)
                vp = (p.vx, p.vy)

                i.vx = vp[0]
                i.vy = vp[1]
                p.vx = vi[0]
                p.vy = vi[1]

                i.move(self.dt)
                p.move(self.dt)

    def set_temperature(self, T_desired):
        if T_desired == -1:
            return # Break the function

        vx = np.array([p.vx for p in self.particles])
        vy = np.array([p.vy for p in self.particles])

        # Set mass
        mass = 1.6735575*10**(-27)
        boltzmann = 1.380649*10**(-23)
        #First measure the temperature
        T_actual = 1/2 * mass * (np.mean(vx**2) + np.mean(vy**2)) / boltzmann * 25 * 10**5 

        c = np.sqrt(T_desired / T_actual)

        for p in self.particles:
            p.vx *= c
            p.vy *= c

    def step(self):
        """Apply the velo verlet solver to the system"""
        self.generate_psudo()
        self.simcount += 1

        # Move the particles and save their old forces
        for p in self.particles:
            # Check if two particles have collided
            self.collision(particle = p)
            p.move(self.dt)
            p.save_forces_old()

        # Apply all forces to the particles
        for p1 in self.particles:
            for p2 in self.particles:
                if p1.n > p2.n:
                    p1.lennard_jones(p2)
                    p1.columb_forces(p2)
            p1.spring_forces()

        # Once all the new particles have been formed find the new veloceties
        for p in self.particles:
            p.vx += 0.5 * self.dt * (p.old_forces[0] + p.force_x)
            p.vy += 0.5 * self.dt * (p.old_forces[1] + p.force_y)

            # Check if the particles has reacted one of the bounding borders
            p.check_bounding()

        self.set_temperature(self.T)

    def simulate(self):
        t = time.time()
        n_frame = 0
        for i in range(self.n_steps):
            s = time.time()
            self.step()

            # Save every 40 tick cycle
            if i % 50 == 0:
                # Cool progress bar
                print(
                    "\r#", str(i).zfill(5), 
                    round(s-t), "s |", 
                    "#"*(round(i/self.n_steps*40)), 
                    " "*(40-round(i/self.n_steps*40)), 
                    "|", 
                    round((s-t)/(i+1)*(self.n_steps-i)), "s", 
                    round(i/self.n_steps*100), "%", 
                    end = "", flush = True
                )
                self.video.save(self, f"frames/{n_frame}.png")
                n_frame += 1

