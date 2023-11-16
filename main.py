import numpy as np
import copy
import time
import os
import copy
import cv2
from PIL import Image, ImageDraw
import json


F_fjedder = lambda dist, ang: (
    np.cos(ang) * (-30*(3 - (dist) )), 
    np.sin(ang) * (-30*(3 - (dist) )), 
)

F_lennard_x = lambda r_ij: 48 * ( (1/r_ij)**13 - 0.5 * (1/(r_ij))**7)
F_lennard_y = lambda r_ij: 48 * ( (1/r_ij)**13 - 0.5 * (1/(r_ij))**7)

limit = lambda x, min_val, max_val: max(min(x, min_val), max_val)


class Particle:
    def __init__(
        self, 
        system,
        n: int, 
        x: float, 
        y: float, 
        vx: float, 
        vy: float, 
        r: float, 
        m = 1
    ):
        self.n = n 
        self.system = system

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy 
        self.m = m
        self.r = r

        self.force_x = 0
        self.force_y = 0

    def __repr__(self):
        return f"<Particle x.{self.x} y.{self.y}>"

    def get_dist(self, p):
        """Get the distance between two particles"""
        return np.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

    def get_angle(self, p):
        """Get the angle between two particles"""
        return np.arcsin((p.y-self.y) / self.get_dist(p))

    def move(self, dt: float):
        """Move the particle"""
        self.x += dt*self.vx + 0.5*dt**2*self.force_x #Kommer fra Newtons bevægelsesligninger. s = v*t + 1/2*a*t^2
        self.y += dt*self.vy + 0.5*dt**2*self.force_y #Husk på: a = f*m. Vi sætter bare massen til 1

    def check_bounding(self):
        """If the particle escapes the container teleport it to the other side"""

        # Change the border type when the border is updated
        if self.system.borders:
            if self.x > self.system.width: 
                self.x -= self.system.width
                self.system.collision(self)

            elif self.x < 0: 
                self.x += self.system.width 
                self.system.collision(self)

            if self.y > self.system.width: 
                self.y -= self.system.width
                self.system.collision(self)

            elif self.y < 0: 
                self.y += self.system.width 
                self.system.collision(self)
        else:
            if self.x > self.system.width or self.x < 0:
                self.vx *= -1
            if self.y > self.system.width or self.y < 0:
                self.vy *= -1

    def lennard_jones(self, particles, box_width):
        """Calculate the lennard jones potential for the particle in relation to all other particles"""
        self.force_x = 0
        self.force_y = 0

        # Look at all particles that is not the one itself and within distance of one box width
        for p in [
            i for i in self.system.psudo_particles 
            if i.n != self.n and self.get_dist(i) < box_width
        ]:
            r_ij = self.get_dist(p) 
            if p.n != self.n + 1 and p.n != self.n - 1:
                self.force_x += F_lennard_x(r_ij) * (-1 if self.x > p.x else 1)
                self.force_y += F_lennard_y(r_ij) * (-1 if self.y > p.y else 1)
                p.force_x    += F_lennard_x(r_ij) * (-1 if self.x < p.x else 1)
                p.force_y    += F_lennard_y(r_ij) * (-1 if self.y < p.y else 1)

    def spring_forces(self):
        """Kovelente bonds between molecules, to enable set n to be one higher than the current molecule"""
        for p in self.system.particles:
            if p.n == self.n + 1:
                min_dist = sorted(
                    [
                        (i, i.get_dist(self)) 
                        for i in self.system.psudo_particles 
                        if i.n == p.n
                    ],
                    key=lambda x: x[1]
                )
                
                ang = self.get_angle(min_dist[0][0])
                spring_force = F_fjedder(
                    self.get_dist(min_dist[0][0]), 
                    abs(ang)
                )

                self.force_x += spring_force[0] * (-1 if self.x > p.x else 1)
                self.force_y += spring_force[1] * (-1 if self.y > p.y else 1)
                p.force_x    += spring_force[0] * (-1 if self.x < p.x else 1)
                p.force_y    += spring_force[1] * (-1 if self.y < p.y else 1)

                break

    def velo_verlet(self, dt: float, box_width: int, particles):
        """Apply the velo verlet solver to this particle, this will update everything considerd related to the particle"""
        self.move(dt)

        old_forces = (
            copy.copy(self.force_x), 
            copy.copy(self.force_y)
        )

        self.lennard_jones(particles, box_width)
        self.spring_forces()

        self.vx += 0.5 * dt * (old_forces[0] + self.force_x)
        self.vy += 0.5 * dt * (old_forces[1] + self.force_y)

        self.check_bounding()


class System:
    def __init__(
        self, 
        n_particles, 
        width, 
        dt, 
        speed_scale, 
        borders, 
        kovelent_lines, 
        lennard_jones_overlay, 
        frame_rate
    ):
        self.n_particles = n_particles
        self.width = width
        self.dt = dt
        self.speed_scale = speed_scale
        self.borders = borders

        self.simcount = 0

        self.kovelent_lines = kovelent_lines,
        self.lennard_jones_overlay = lennard_jones_overlay,
        self.frame_rate = frame_rate

        self.particles = []
        self.psudo_particles = []

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
                        n = n,
                        x = x / sqrt_npart * (self.width * 0.9) + self.width * 0.1,
                        y = y / sqrt_npart * (self.width * 0.9) + self.width * 0.1,
                        vx = self.speed_scale * 2 * (np.random.random()-0.5),
                        vy = self.speed_scale * 2 * (np.random.random()-0.5),
                        r = .4
                    )
                )
                n += 1

        self.particles = self.particles[:self.n_particles]
        self.generate_psudo()

    def load_system(self, filename):
        """Import a system from a json file"""
        with open(filename, "r") as file:
            data = json.load(file)

        for i in data:
            self.particles.append(
                Particle(
                    system = self,
                    n = i["n"],
                    x = i["x"]-10,
                    y = i["y"]-10,
                    vx = i["vx"],
                    vy = i["vy"],
                    r = i["r"]
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
            if l < i.r + p.r:
                vi = (i.vx, i.vy)
                vp = (p.vx, p.vy)

                i.vx = vp[0]
                i.vy = vp[1]
                p.vx = vi[0]
                p.vy = vi[1]

                i.move(self.dt)
                p.move(self.dt)

    def step(self):
        """Step the simulation one dt"""
        self.generate_psudo()
        self.simcount += 1

        for p in self.particles:
            self.collision(particle = p)
            p.velo_verlet(dt=self.dt, box_width=self.width, particles=system.particles)

    def save(self, filename='pointplot.png'):
        """Create an image from the position of all particles."""
        scale = 20

        img = Image.new("RGB", (self.width * scale, self.width * scale), "black")
        draw = ImageDraw.Draw(img)
        
        # Lennard-jones petential overlay
        if self.lennard_jones_overlay: 
            for r in range(9*4, 0, -1):
                p = self.particles[1]
                r = r/4 + p.r
                mult = F_lennard_x(r)
                if mult < 0:
                    color = (0, round((abs(mult)*10)), 0)
                else:
                    color = (round((abs(mult))), 0, 0)

                draw.ellipse(
                    [
                        (round(p.x * scale - r * scale), round(p.y * scale - r * scale)),
                        (round(p.x * scale + r * scale), round(p.y * scale + r * scale))
                    ], 
                    fill=color, 
                    outline=color
                )

        particle_collection = self.psudo_particles
        for i, p in enumerate(particle_collection):
            try:
                if particle_collection[i+1].n == p.n + 1:
                    min_dist = sorted(
                        [
                            (x, x.get_dist(p)) 
                            for x in self.psudo_particles 
                            if x.n == particle_collection[i+1].n
                        ], 
                        key=lambda x: x[1]
                    )

                    draw.line(
                        (
                            p.x * scale, p.y * scale, 
                            min_dist[0][0].x * scale, min_dist[0][0].y * scale
                        ), 
                        fill=(0,255,0)
                    )
            except IndexError:
                pass

        for i, p in enumerate(particle_collection):
            color = (0, 155, 155)

            # Draw the circle on the canvas
            draw.ellipse(
                [
                    (p.x * scale - p.r * scale, p.y * scale - p.r * scale),
                    (p.x * scale + p.r * scale, p.y * scale + p.r * scale)
                ], 
                fill=color, 
                outline=color
            )

        draw.text(
            (30, 30),
            f"Iteration {self.simcount}\ndt: {self.dt}\n{self.frame_rate} FPS\nSystem: Protine 1 (test)",
            (200,200,200)
        )

        img.save(filename)

    def animate(self):
        """Combines all pictures from the 'save' method into a playable avi video"""
        print("\nWriting video\n")
        image_folder = 'frames'
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        new_img = []

        for i in range(len(images)):
            new_img.append(f"frames/{i}.png")

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, self.frame_rate, (width,height))

        for i, image in enumerate(new_img):
            video.write(cv2.imread(image))
            os.remove(image)
            print("\rPlease wait", "".join([
                " " if i % 10 != 0 else ".", 
                " " if i % 10 != 1 else ".", 
                " " if i % 10 != 2 else ".", 
                " " if i % 10 != 3 else ".", 
                " " if i % 10 != 4 else ".", 
                " " if i % 10 != 5 else ".", 
                " " if i % 10 != 6 else ".", 
                " " if i % 10 != 7 else ".", 
                " " if i % 10 != 8 else ".", 
                " " if i % 10 != 9 else ".", 
            ]), end = "", flush = True)

        cv2.destroyAllWindows()
        video.release()
        print("\nDone")


if __name__ == "__main__":
    system = System(
        n_particles = 16, 
        width = 45, 
        dt = 0.002, 
        speed_scale = 1,
        borders = False,

        # Drawing based settings
        kovelent_lines = True,
        lennard_jones_overlay = False,
        frame_rate = 60
    )

    # system.create_system()
    system.load_system("examples/protine.json")
    
    t = time.time()
    n_steps = 62000
    n_frame = 0
    for i in range(n_steps):
        s = time.time()
        system.step()

        # Save every 40 tick cycle
        if i % 40 == 0:
            # Cool progress bar
            print(
                "\r#", 
                str(i).zfill(5), 
                round(s-t), 
                "s |", 
                "#"*(round(i/n_steps*40)), 
                " "*(40-round(i/n_steps*40)), 
                "|", round((s-t)/(i+1)*(n_steps-i)), 
                "s", 
                round(i/n_steps*100), 
                "%", 
                end = "", 
                flush = True
            )
            system.save(f"frames/{n_frame}.png")
            n_frame += 1

    system.animate()

