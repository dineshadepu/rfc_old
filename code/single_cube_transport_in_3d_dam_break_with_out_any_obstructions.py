"""Taken from \cite{ji2019coupled}, this is inturn was done in
\citet{canelas2016sph}.


This case models a single cube in the centre of the flume, with a face directly
aligned with the flow direction.
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
from pysph.tools.geometry import get_3d_block


def get_fluid_tank_3d(xf, yf, zf, tank_length, tank_height, tank_layers,
                      tank_spacing):
    """
    Given the fluid block and tank dimensions, return the tank geometry
    here

    tank_length: length in x direction
    tank_height: length in z direction
    """
    # create the left wall
    min_yf = min(yf)
    max_yf = max(yf)

    # width is in y direction
    left_width = max_yf - min_yf

    # create a tank layer on the left
    xt_left, yt_left, zt_left = get_3d_block(dx=tank_spacing,
                                             length=(tank_layers - 1.)*tank_spacing,
                                             height=left_width,
                                             depth=tank_height)
    # adjust the left wall of tank
    xt_left += np.min(xf) - np.max(xt_left) - tank_spacing
    yt_left += np.min(yf) - np.min(yt_left) + 0. * tank_spacing
    zt_left += np.min(zf) - np.min(zt_left) + 0. * tank_spacing

    # create a tank layer on the right
    xt_right, yt_right, zt_right = get_3d_block(dx=tank_spacing,
                                                length=(tank_layers - 1.)*tank_spacing,
                                                height=left_width,
                                                depth=tank_height)
    # adjust the left wall of tank
    xt_right += tank_length - (np.max(xt_right) - np.min(xf))
    yt_right += np.min(yf) - np.min(yt_right) + 0. * tank_spacing
    zt_right += np.min(zf) - np.min(zt_right) + 0. * tank_spacing

    # create the wall in the front
    xt_front, yt_front, zt_front = get_3d_block(
        dx=tank_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_spacing * (tank_layers - 1),
        depth=tank_height)
    xt_front += np.min(xt_left) - np.min(xt_front)
    yt_front += np.min(yf) - np.min(yt_front) + 0. * tank_spacing
    yt_front -= tank_layers * tank_spacing
    zt_front += np.min(zt_left) - np.min(zt_front)

    # create the wall in the back
    xt_back, yt_back, zt_back = get_3d_block(
        dx=tank_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=tank_spacing * (tank_layers - 1),
        depth=tank_height)
    xt_back += np.min(xt_left) - np.min(xt_back)
    yt_back += np.max(yf) - np.min(yt_back) + 1. * tank_spacing
    zt_back += np.min(zt_left) - np.min(zt_back)

    # create the wall in the bottom
    xt_bottom, yt_bottom, zt_bottom = get_3d_block(
        dx=tank_spacing,
        length=np.max(xt_right) - np.min(xt_left),
        height=np.max(yt_back) - np.min(yt_front),
        depth=tank_spacing * (tank_layers - 1))
    xt_bottom += np.min(xt_left) - np.min(xt_bottom)
    yt_bottom += np.min(yt_front) - np.min(yt_front)
    zt_bottom += np.min(zt_left) - np.max(zt_bottom) - tank_spacing * 1

    xt = np.concatenate([xt_left, xt_right, xt_front, xt_back, xt_bottom])
    yt = np.concatenate([yt_left, yt_right, yt_front, yt_back, yt_bottom])
    zt = np.concatenate([zt_left, zt_right, zt_front, zt_back, zt_bottom])

    return xt, yt, zt


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.fluid_length = 3.5
        # y-dimension (this goes into the paper)
        self.fluid_width = 0.7
        # z-dimension (this is in the plane of the paper going up)
        self.fluid_height = 0.4

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_length = 8.
        self.tank_width = 0.7
        self.tank_height = 0.7
        self.tank_spacing = spacing
        self.tank_layers = 3

        self.body_height = 0.15
        self.body_length = 0.15
        self.body_depth = 0.15
        self.body_density = 800.
        self.body_spacing = spacing
        self.body_h = self.hdx * self.body_spacing

        self.h = self.hdx * self.fluid_spacing

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.co = 10 * np.sqrt(2 * 9.81 * self.fluid_height)
        self.p0 = self.fluid_density * self.co**2.
        self.c0 = self.co
        self.alpha = 0.1
        self.gz = -9.81
        self.dim = 3

    def create_particles(self):
        xf, yf, zf = get_3d_block(self.fluid_spacing, self.fluid_length,
                                  self.fluid_width, self.fluid_height)

        xt, yt, zt = get_fluid_tank_3d(xf, yf, zf, self.tank_length,
                                       self.tank_height, self.tank_layers,
                                       self.fluid_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   z=zf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")

        tank = get_particle_array(x=xt,
                                  y=yt,
                                  z=zt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank")
        tank.add_property('dem_id', type='int', data=1)

        # create the wall which should be lifted for the water to flow
        xw, yw, zw = get_3d_block(self.fluid_spacing, 3. * self.fluid_spacing,
                                  self.fluid_width, self.tank_height)
        xw += np.max(xf) - np.min(xw) + self.fluid_spacing
        zw += np.min(zt) - np.min(zw) + self.tank_layers * self.fluid_spacing
        wall = get_particle_array(x=xw,
                                  y=yw,
                                  z=zw,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="wall")
        wall.add_property('dem_id', type='int', data=2)
        # Translate the tank and fluid so that fluid starts at 0
        min_xf = abs(np.min(xf))

        fluid.x += min_xf
        tank.x += min_xf
        wall.x += min_xf

        # Create the rigid body
        print("body spacing", self.body_spacing)
        print("fluid spacing", self.fluid_spacing)
        xb, yb, zb = get_3d_block(dx=self.body_spacing,
                                  length=self.body_length,
                                  height=self.body_height,
                                  depth=self.body_depth)
        xb += np.min(fluid.x) - np.min(xb)
        xb += self.fluid_length + 1.78
        zb += np.min(zt) - np.min(zb) + self.fluid_spacing * self.tank_layers
        # yb += np.max(fluid.y) - np.min(yb) + 1. * self.body_spacing
        # zb += np.max(fluid.z) / 2. + np.min(zb) - 6. * self.body_spacing
        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  z=zb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.)
        body_id = np.zeros(len(xb), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_constant('max_tng_contacts_limit', 30)
        body.add_property('dem_id', type='int', data=0)

        self.scheme.setup_properties([fluid, tank, body])

        # body.y[:] += 0.5
        body.m_fsi[:] += self.fluid_density * self.body_spacing**self.dim
        body.rho_fsi[:] = 1000
        return [fluid, tank, body]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=["body"],
                                       fluids=['fluid'],
                                       boundaries=['tank'],
                                       dim=3,
                                       rho0=self.fluid_density,
                                       p0=self.p0,
                                       c0=self.c0,
                                       gz=self.gz,
                                       nu=0.,
                                       h=None)
        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        scheme.configure(h=self.h)

        dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        print("DT: %s" % dt)
        tf = 1.8

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_step(self, solver):
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'wall':
                pa.z += 0.11 * dt


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
