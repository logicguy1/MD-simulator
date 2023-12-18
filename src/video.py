import cv2
import os
from PIL import Image, ImageDraw

class Video:
    def save(self, system, filename='test.png'):
        """Create an image from the position of all particles."""
        scale = 20
        rightbar = 170 

        img = Image.new("RGB", (system.width * scale + rightbar, system.width * scale), "white")
        draw = ImageDraw.Draw(img)
        
        for y in range(system.width):
            for x in range(system.width):
                draw.rectangle(
                    [
                        (rightbar + x * scale - 1 + 0.5 * scale, y * scale - 1 + 0.5 * scale),
                        (rightbar + x * scale + 1 + 0.5 * scale, y * scale + 1 + 0.5 * scale),
                    ],
                    fill = (100,100,100)
                )

        # Lennard-jones petential overlay
        if system.lennard_jones_overlay and False: 
            for r in range(9*5, 0, -1):
                p = system.particles[0]
                r = r/5 + p.r
                mult = F_lennard(r, 0)[0]
                if mult < 0:
                    color = (0, round((abs(mult)*20)), 0)
                else:
                    color = (round((abs(mult))), 0, 0)

                draw.ellipse(
                    [
                        (rightbar + round(p.x * scale - r * scale), round(p.y * scale - r * scale)),
                        (rightbar + round(p.x * scale + r * scale), round(p.y * scale + r * scale))
                    ], 
                    fill=color, 
                    outline=color
                )

        for i in system.particles:
            for j in system.particles:
                if i.n in j.connected or j.n in i.connected:
                    min_dist = sorted(
                        [
                            (x, x.get_dist(i)) 
                            for x in system.psudo_particles 
                            if x.n == j.n
                        ],
                        key=lambda x: x[1]
                    )[0][0]

                    draw.line(
                        (
                            rightbar + i.x * scale, i.y * scale, 
                            rightbar + min_dist.x * scale, min_dist.y * scale
                        ), 
                        fill=(0,255,0)
                    )

        for i, p in enumerate(system.psudo_particles):
            charges = {
                "-2": (255, 20, 20),
                "-1": (255, 100, 100),
                "0" : (200, 200, 200),
                "1" : (100, 255, 100),
                "2" : (20, 255, 20),
            }

            # Draw the circle on the canvas
            draw.ellipse(
                [
                    (rightbar + p.x * scale - p.r * scale, p.y * scale - p.r * scale),
                    (rightbar + p.x * scale + p.r * scale, p.y * scale + p.r * scale)
                ], 
                fill    = charges.get(str(p.charge)), 
                outline = charges.get(str(p.charge))
            )

        # Text background
        draw.rectangle(
            [
                (0, 0),
                (rightbar, system.width * scale),
            ],
            fill = "white"
        )

        # Text line seperator
        draw.rectangle(
            [
                (rightbar -1, 0),
                (rightbar + 1, system.width * scale),
            ],
            fill = (200,0,0)
        )

        draw.text(
            (30, 30),
            f"""Iteration {system.simcount}
dt: {system.dt}
{system.frame_rate} FPS 
System: {system.name}
""",
            (20,20,20)
        )

        img.save(filename)

    def animate(self, system):
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

        video = cv2.VideoWriter(video_name, 0, system.frame_rate, (width,height))

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
