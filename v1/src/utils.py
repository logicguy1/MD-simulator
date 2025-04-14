import numpy as np

F_fjedder = lambda dist, ang: (
    np.cos(ang) * (-300 * (1 - dist)), 
    np.sin(ang) * (-300 * (1 - dist)), 
)

F_lennard = lambda r_ij, ang: (
    np.cos(ang) * (48 * ( (1/r_ij)**13 - 0.5 * (1/(r_ij))**7)), 
    np.sin(ang) * (48 * ( (1/r_ij)**13 - 0.5 * (1/(r_ij))**7)), 
)

F_coulomb = lambda r_ij, ang, q1, q2: (
    np.cos(ang) * (332 * (q1 * q2)/r_ij**2), 
    np.sin(ang) * (332 * (q1 * q2)/r_ij**2), 
)
