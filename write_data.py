import random

def write_data(filename: str, ts: float, bounds: float, particles):
    """Function to write a timeframe in a ovito supported format

    Arguments:
    filename - The name of the file that should be saved to
    ts - Timestamp of the current timestep
    bounds - The size of the bounding box
    particles - An array of tuples to save with the relevant data, typically in the following format
        v_x v_y v_z radius y x z mass
    """

    with open(filename, "a") as file:
        file.write(f"""
ITEM: TIMESTEP
{ts}
ITEM: NUMBER OF ATOMS
{len(particles)}
ITEM: BOX BOUNDS f f f 
0 {bounds}
0 {bounds}
0 {bounds}
""".strip())
        file.write("\nITEM: ATOMS " + " ".join(particles[0].keys()))
        for p in particles:
            file.write("\n"+ " ".join([str(i) for i in p.values()]))

        file.write("""
ITEM: NUMBER OF BONDS
2
ITEM: BONDS bond_id bond_type atom1 atom2 
1 1 1 2
2 1 2 3""")
        file.write("\n"*1)


if __name__ == "__main__":
    with open("output.dump", "w") as file:
        file.write("")

    for i in range(200):
        particles = [{"v_x": 1, "v_y": 1, "v_z": 1, "radius": 1, "y": random.randint(0,50), "x": random.randint(0,50), "z": random.randint(0,50), "mass": 1} for i in range(200)]
        write_data(
            filename="output.dump", 
            ts=i, 
            bounds=50, 
            particles=particles
        )


