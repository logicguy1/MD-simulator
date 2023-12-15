import unittest
import numpy as np
import copy

from particle import Particle
from system import System
from video import Video


class TestParticle(unittest.TestCase):
    def setUp(self):
        """Create an instance of Particle for testing"""
        self.system = System()

        self.particle = Particle(
            system=self.system,
            n=1,
            x=0,
            y=0,
            vx=1,
            vy=1,
            r=1,
            m=1,
            charge=0,
            connected=[]
        )

        self.particle2 = Particle(
            system=self.system,
            n=2,
            x=2,
            y=0,
            vx=-1,
            vy=1,
            r=1,
            m=1,
            charge=0,
            connected=[]
        )

    def test_initialization(self):
        """Test the particle attribudes being set right"""
        self.assertEqual(self.particle.n, 1)
        self.assertEqual(self.particle.x, 0)
        self.assertEqual(self.particle.y, 0)
        self.assertEqual(self.particle.vx, 1)
        self.assertEqual(self.particle.vy, 1)
        self.assertEqual(self.particle.r, 1)
        self.assertEqual(self.particle.m, 1)
        self.assertEqual(self.particle.charge, 0)
        self.assertEqual(self.particle.connected, [])
        self.assertEqual(self.particle.force_x, 0)
        self.assertEqual(self.particle.force_y, 0)
        self.assertEqual(self.particle.old_forces, (0, 0))

    def test_save_forces_old(self):
        """Test if save_forces_old resets the forces correctly"""
        self.particle.force_x = 10
        self.particle.force_y = 5
        self.particle.save_forces_old()
        self.assertEqual(self.particle.old_forces, (10, 5))
        self.assertEqual(self.particle.force_x, 0)
        self.assertEqual(self.particle.force_y, 0)

    def test_check_bounding(self):
        """Test the bounding setup is calculating correctly"""
        self.particle.x = 10
        self.particle.y = 5
        self.particle.vx = 2
        self.particle.vy = 3

        self.system.width = 6

        # As only the X component is outside the border it should be flipped but not the other
        self.particle.check_bounding()
        self.assertEqual(self.particle.vx, -2)
        self.assertEqual(self.particle.vy, 3)

        self.system.width = 3

        # As both now are outside both should be flipped
        self.particle.check_bounding()
        self.assertEqual(self.particle.vx, 2)
        self.assertEqual(self.particle.vy, -3)

    def test_move(self):
        """Test the move function is calculating movements correctly"""
        self.particle.x = 2
        self.particle.y = 2
        self.particle.vx = 3
        self.particle.vy = 3
        self.particle.force_x = 2
        self.particle.force_y = 2

        self.particle.move(1)

        # Calculations showed in the report
        self.assertEqual(self.particle.x, 6)
        self.assertEqual(self.particle.y, 6)

    def test_get_dist(self):
        """Test the get_dist distance function"""
        distance = self.particle.get_dist(self.particle2)
        expected_distance = 2
        self.assertEqual(distance, expected_distance)

        # Test the 345 triangle
        self.particle2.x = 3
        self.particle2.y = 4

        distance = self.particle.get_dist(self.particle2)
        expected_distance = 5
        self.assertEqual(distance, expected_distance)

    def test_get_angle(self):
        """Test the get_angle angle function"""
        angle = self.particle.get_angle(self.particle2)
        expected_angle = 0  # As the two particles are on the same parrelle line
        self.assertEqual(angle, expected_angle)

        self.particle2.x = 3
        self.particle2.y = 4

        angle = self.particle.get_angle(self.particle2)
        expected_angle = 0.927  # As found in geogebra in radients
        self.assertEqual(round(angle, 3), expected_angle)


class TestSystem(unittest.TestCase):
    def setUp(self):
        """Create an instance of System for testing"""
        self.system = System()

        self.particle = Particle(
            system=self.system,
            n=1,
            x=0,
            y=0,
            vx=-1,
            vy=1,
            r=1,
            m=1,
            charge=0,
            connected=[]
        )

        self.particle2 = Particle(
            system=self.system,
            n=2,
            x=2,
            y=0,
            vx=2,
            vy=1.5,
            r=1,
            m=1,
            charge=0,
            connected=[]
        )

        self.system.particles = [self.particle, self.particle2]

    def test_initialization(self):
        """Test whether the System class is initialized correctly with the given parameters"""
        self.assertFalse(self.system.kovelent_lines)
        self.assertFalse(self.system.lennard_jones_overlay)
        self.assertEqual(self.system.frame_rate, 30)

        # Test default values
        self.assertEqual(self.system.n_particles, 10)
        self.assertEqual(self.system.width, 20)
        self.assertAlmostEqual(self.system.dt, 0.001)
        self.assertEqual(self.system.speed_scale, 1)
        self.assertFalse(self.system.borders)
        self.assertEqual(self.system.T, 90)
        self.assertEqual(self.system.name, "")
        self.assertEqual(self.system.simcount, 0)

        # Test rendering settings
        self.assertTrue(isinstance(self.system.video, Video))

        # Test particle lists
        self.assertEqual(len(self.system.particles), 2)
        self.assertEqual(len(self.system.psudo_particles), 0)

    def test_psudo(self):
        """Test is the correct amount of psudo elements are generated"""
        self.system.borders = True
        self.system.generate_psudo()
        expected_amount = 2*9
        self.assertEqual(len(self.system.psudo_particles), expected_amount)

    def test_collision(self):
        """Test the collision system and that the veloceties are flipped"""
        self.system.borders = False
        self.system.generate_psudo()

        self.particle.x = 1
        self.particle.y = 0
        self.particle2.x = 1.1
        self.particle2.y = 0

        self.system.collision(self.particle2)

        self.assertEqual(self.particle.vx, 2)
        self.assertEqual(self.particle.vy, 1.5)
        self.assertEqual(self.particle2.vx, -1)
        self.assertEqual(self.particle2.vy, 1)

    def test_set_temperature(self):
        """Test the temperature function woop"""
        initial_temperature = 300  # Example initial temperature in Kelvin

        # Initialize particles with random velocities
        num_particles = 100  # Choose an appropriate number of particles
        for _ in range(num_particles):
            particle = Particle(self.system, n=0, x=0, y=0, vx=np.random.rand(), vy=np.random.rand(), r=1)
            self.system.particles.append(particle)

        # Set the initial temperature and create a copy of the particles
        self.system.set_temperature(initial_temperature)
        old_particles = copy.deepcopy(self.system.particles)

        # Call the set_temperature method
        desired_temperature = 500  # Example desired temperature in Kelvin
        self.system.set_temperature(desired_temperature)

        # Check whether velocities are correctly scaled
        scaling_factor = np.sqrt(desired_temperature / initial_temperature)
        for old_particle, particle in zip(old_particles, self.system.particles):
            self.assertAlmostEqual(particle.vx, scaling_factor * old_particle.vx)
            self.assertAlmostEqual(particle.vy, scaling_factor * old_particle.vy)


if __name__ == '__main__':
    unittest.main()
