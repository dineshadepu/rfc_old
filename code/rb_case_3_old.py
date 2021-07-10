"""
Test the collision of two rigid bodues made of same particle array
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.2
        self.body_length = 0.2
        self.body_density = 2000
        self.body_spacing = spacing / 2.
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gy = 0.
        self.dim = 2

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(
            self.fluid_length, self.fluid_height, self.tank_height,
            self.tank_layers, self.fluid_spacing, self.fluid_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**2.

        xb1, yb1 = get_2d_block(dx=self.body_spacing,
                                length=self.body_length,
                                height=self.body_height)
        m = self.body_density * self.body_spacing**self.dim

        xb2 = xb1 + self.body_length * 2
        yb2 = yb1

        xb = np.concatenate([xb1, xb2])
        yb = np.concatenate([yb1, yb2])

        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.)
        body_id1 = np.zeros(len(xb1), dtype=int)
        body_id2 = np.ones(len(xb2), dtype=int)
        body_id = np.concatenate([body_id1, body_id2])

        dem_id = np.concatenate([body_id1, body_id2])

        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)

        # for dem contact force law
        body.add_constant('max_tng_contacts_limit', 30)

        self.scheme.setup_properties([body])

        self.scheme.scheme.set_linear_velocity(
            body, np.array([
                0.5,
                0.,
                0.,
                -0.5,
                0.,
                0.,
            ]))

        return [body]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['body'],
                                       fluids=[],
                                       boundaries=[],
                                       dim=2,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gy=self.gy)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        print("DT: %s" % dt)
        tf = 0.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
