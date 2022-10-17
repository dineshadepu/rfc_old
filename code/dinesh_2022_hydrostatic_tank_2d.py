"""[1] A 3D simulation of a moving solid in viscous free-surface flows by
coupling SPH and DEM

3.2 Falling solid in water.

"""
from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

# from rigid_body_3d import RigidBody3DScheme
from rigid_fluid_coupling import RigidFluidCouplingScheme
from pysph.sph.equation import Equation, Group
import os
from pysph.tools import geometry as G

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate
from geometry import hydrostatic_tank_2d


class Dinesh2022HydrostaticTank2D(Application):
    def initialize(self):
        self.dim = 2
        spacing = 2. * 1e-3
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.fluid_length = 150. * 1e-3
        # y-dimension (this goes into the paper)
        self.fluid_height = 131 * 1e-3

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_denstiy = 1000
        self.tank_length = 150. * 1e-3
        self.tank_height = 140 * 1e-3
        self.tank_spacing = spacing
        self.tank_layers = 3

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        # fixme: why is it blowing up with higher alpha
        self.alpha = 0.5
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2

        # solver data
        self.tf = 0.5
        self.dt = 0.25 * self.fluid_spacing * self.hdx / (self.c0 * 1.1)

        # Rigid body collision related data
        self.limit = 6
        self.seval = None

    def add_user_options(self, group):
        group.add_argument("--dx",
                           action="store",
                           type=float,
                           dest="dx",
                           default=5. * 1e-3,
                           help="Spacing between particles")

        group.add_argument("--tank-rho",
                           action="store",
                           type=float,
                           dest="tank_rho",
                           default=1000.,
                           help="Density of tank")

    def consume_user_options(self):
        spacing = self.options.dx
        self.fluid_spacing = spacing
        self.tank_spacing = spacing

        self.tank_density = self.options.tank_rho

    def create_particles(self):

        xf, yf, xt, yt = hydrostatic_tank_2d(
            fluid_length=self.fluid_length,
            fluid_height=self.fluid_height,
            tank_height=self.tank_height,
            tank_layers=self.tank_layers,
            fluid_spacing=self.fluid_spacing,
            tank_spacing=self.tank_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")
        # set the initial pressure
        fluid.p[:] = - self.fluid_density * self.gy * (max(fluid.y) - fluid.y[:])

        m_tank = self.tank_density * self.tank_spacing**self.dim
        tank = get_particle_array(x=xt,
                                  y=yt,
                                  m=m_tank,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.tank_density,
                                  rad_s=self.fluid_spacing/2.,
                                  contact_force_is_boundary=1.,
                                  name="tank",
                                  constants={
                                      'E': 21 * 1e10,
                                      'poisson_ratio': 0.3,
                                  })
        tank.add_property('dem_id', type='int', data=1)

        # # Translate the tank and fluid so that fluid starts at 0
        # min_xf = abs(np.min(xf))

        # fluid.x += min_xf
        # tank.x += min_xf

        self.scheme.setup_properties([fluid, tank])

        return [fluid, tank]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=None,
                                       fluids=['fluid'],
                                       rigid_bodies_dynamic=None,
                                       rigid_bodies_static=['tank'],
                                       dim=self.dim,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gx=self.gx,
                                       gy=self.gy,
                                       gz=self.gz,
                                       nu=0.,
                                       alpha=self.alpha,
                                       h=self.h)

        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        tf = self.tf

        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = Dinesh2022HydrostaticTank2D()
    app.run()
    app.post_process(app.info_filename)
