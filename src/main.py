import time
import os

from system import System

if __name__ == "__main__":
    system = System(
        # Drawing based settings
        kovelent_lines = False,
        lennard_jones_overlay = False,
        frame_rate = 60
    )

    print("Select a system to load")
    examples = os.listdir("examples")
    for idx, item in enumerate(examples):
        print(f"{idx} : {item}")

    inp = int(input(">> "))

    #system.load_system("examples/2particles.json")
    system.load_system(f"examples/{examples[inp]}")
    system.simulate()
    system.video.animate(system)

