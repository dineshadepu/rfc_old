"""DISCRETE ELEMENT METHOD FOR 3D SIMULATIONS OF MECHANICAL SYSTEMS OF
NON-SPHERICAL GRANULAR MATERIALS by JIAN CHEN

Section 5.4

"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_fluid_coupling import RigidFluidCouplingScheme

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block, rotate


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.008 * 2.
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.fluid_length = 0.5
        # y-dimension (this goes into the paper)
        self.fluid_width = 0.5
        # z-dimension (this is in the plane of the paper going up)
        self.fluid_height = 0.35

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_length = 3.5
        self.tank_width = 0.5
        self.tank_height = 0.5
        self.tank_spacing = spacing
        self.tank_layers = 3

        self.obstacle_length = 0.15
        self.obstacle_width = 0.15
        self.obstacle_height = 0.75
        self.obstacle_spacing = spacing

        self.body_height = 0.2
        self.body_length = 0.2
        self.body_depth = 0.054
        self.body_density = 2000.
        self.body_spacing = spacing
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = -9.81
        self.dim = 2

    def create_particles(self):
        # create the wall which should be lifted for the water to flow
        xw, yw = get_2d_block(self.fluid_spacing, 10., self.tank_layers*self.tank_spacing)
        zw = np.zeros_like(xw)

        # xw, yw, zw = rotate(xw, yw, zw, angle=-45.)

        m = self.body_density * self.body_spacing**self.dim
        rad_s = self.fluid_spacing
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  m=m,
                                  m_fluid=1.,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=rad_s,
                                  name="wall")
        wall.add_property('dem_id', type='int', data=1)

        # Create the rigid body
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        xb[:] -= np.min(xb) - np.min(xw) - self.body_length

        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=1.,
                                  rad_s=rad_s)
        body.y[:] += np.max(wall.y) - np.min(body.y) + self.body_spacing * 2.
        body.x[:] += self.body_height
        body_id = np.zeros(len(xb), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_constant('max_tng_contacts_limit', 30)
        body.add_property('dem_id', type='int', data=0)

        self.scheme.setup_properties([body, wall])

        # body.y[:] += 0.5
        body.m_fsi[:] += self.fluid_density * self.body_spacing**self.dim
        body.rho_fsi[:] = 1000

        self.scheme.scheme.set_linear_velocity(body, [-1, 0., 0.])
        return [body, wall]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=["body"],
                                       fluids=None,
                                       boundaries=['wall'],
                                       dim=2,
                                       en=0.1,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gy=self.gy,
                                       nu=0.,
                                       h=None)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        scheme.configure(h=self.h)

        dt = 5e-4
        print("DT: %s" % dt)
        tf = 1.

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
