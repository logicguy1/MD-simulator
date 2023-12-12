import numpy as np
import copy
import time
import os
import copy
import cv2
from PIL import Image, ImageDraw
import json



limit = lambda x, min_val, max_val: max(min(x, min_val), max_val)


class System:
    def __init__(
        self, 
        kovelent_lines: bool = False, 
        lennard_jones_overlay: bool = False, 
        frame_rate: int= 30
    ):
        self.n_particles = 10
        self.width = 20
        self.dt = 0.001
        self.speed_scale = 1
        self.borders = False
        self.T = 90

        self.name = "" # Used in rendering
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

        for i in data.get("particles"):
            self.particles.append(
                Particle(
                    system = self,
                    n = i["n"],
                    x = i["x"],
                    y = i["y"],
                    vx = i["vx"],
                    vy = i["vy"],
                    r = i["r"] / 2.5,
                    connected = i["connected"]
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
            p1.spring_forces()

        # Once all the new particles have been formed find the new veloceties
        for p in self.particles:
            p.vx += 0.5 * self.dt * (p.old_forces[0] + p.force_x)
            p.vy += 0.5 * self.dt * (p.old_forces[1] + p.force_y)

            # Check if the particles has reacted one of the bounding borders
            p.check_bounding()

        system.set_temperature(self.T)


    def save(self, filename='pointplot.png'):
        """Create an image from the position of all particles."""
        scale = 20

        img = Image.new("RGB", (self.width * scale, self.width * scale), "black")
        draw = ImageDraw.Draw(img)
        
        # Lennard-jones petential overlay
        if self.lennard_jones_overlay and False: 
            for r in range(9*5, 0, -1):
                p = self.particles[0]
                r = r/5 + p.r
                mult = F_lennard(r, 0)[0]
                if mult < 0:
                    color = (0, round((abs(mult)*20)), 0)
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

        for i in self.particles:
            for j in self.particles:
                if i.n in j.connected or j.n in i.connected:
                    min_dist = sorted(
                        [
                            (x, x.get_dist(i)) 
                            for x in self.psudo_particles 
                            if x.n == j.n
                        ],
                        key=lambda x: x[1]
                    )[0][0]

                    draw.line(
                        (
                            i.x * scale, i.y * scale, 
                            min_dist.x * scale, min_dist.y * scale
                        ), 
                        fill=(0,255,0)
                    )

        for i, p in enumerate(self.psudo_particles):
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
            f"""Iteration {self.simcount}
dt: {self.dt}
{self.frame_rate} FPS 
System: {self.name}
""",
            (200,200,200)
        )

        img.save(filename)

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
                self.save(f"frames/{n_frame}.png")
                n_frame += 1

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

system = System(
    # Drawing based settings
    kovelent_lines = False,
    lennard_jones_overlay = False,
    frame_rate = 60
)

system.load_system("examples/2particles.json")
system.simulate()
system.animate()




if __name__ == "__main__" and False:
    system = System(
        n_particles = 20, 
        width = 45, 
        dt = 0.001, 
        speed_scale = 1,
        borders = True,

        # Drawing based settings
        kovelent_lines = False,
        lennard_jones_overlay = False,
        frame_rate = 60
    )

    #system.create_system()
    #system.load_system("examples/2particles.json")
    system.load_system("examples/protine.json")
    
    t = time.time()
    n_steps = 150000
    n_frame = 0
    for i in range(n_steps):
        s = time.time()
        system.step()

        # Save every 40 tick cycle
        if i % 50 == 0:
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

