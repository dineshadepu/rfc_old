"""
Amaro2019Improvement

An improvement of rigid bodies contact for particle-based
non-smooth walls modeling

https://doi.org/10.1007/s40571-019-00233-4

An implementation of collision of three rigid bodies.
"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_body_3d import RigidBody3DScheme
from geometry import hydrostatic_tank_2d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_2d_block


class RigidFluidCoupling(Application):
    def initialize(self):
        spacing = 0.01
        self.hdx = 1.3

        self.fluid_length = 1.0
        self.fluid_height = 1.0
        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 1.5
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.2 - spacing/2.
        self.body_length = 0.2 - spacing/2.
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
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body1_density = 1.5 / 0.2**2.
        m = self.body1_density * self.body_spacing**self.dim
        # left body
        body1 = get_particle_array(name='body1',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body1_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'initial_spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.zeros(len(xb), dtype=int)
        body1.add_property('body_id', type='int', data=body_id)
        body1.add_property('dem_id', type='int', data=dem_id)
        body1.add_constant('total_no_bodies', [3])

        # right bottom body
        # Body 2 starts
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body2_density = 3. / 0.2**2.
        m = self.body2_density * self.body_spacing**self.dim
        xb += max(body1.x) - min(xb) + 0.03
        yb -= min(body1.y) - min(yb) + 0.165 - 1. * self.body_spacing
        body2 = get_particle_array(name='body2',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body2_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'initial_spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int)
        body2.add_property('body_id', type='int', data=body_id)
        body2.add_property('dem_id', type='int', data=dem_id)
        body2.add_constant('total_no_bodies', [3])

        # Body 3 starts
        xb, yb = get_2d_block(dx=self.body_spacing,
                              length=self.body_length,
                              height=self.body_height)
        self.body3_density = 1.2 / 0.2**2.
        m = self.body3_density * self.body_spacing**self.dim
        xb += max(body1.x) - min(xb) + 0.03
        yb += min(body1.y) - min(yb) + 0.165 - 1. * self.body_spacing
        # right top body
        body3 = get_particle_array(name='body3',
                                   x=xb,
                                   y=yb,
                                   h=self.body_h,
                                   m=m,
                                   rho=self.body3_density,
                                   m_fluid=0.,
                                   rad_s=self.body_spacing / 2.,
                                   constants={
                                       'E': 69 * 1e9,
                                       'poisson_ratio': 0.3,
                                       'initial_spacing0': self.body_spacing,
                                   })
        body_id = np.zeros(len(xb), dtype=int)
        dem_id = np.ones(len(xb), dtype=int) * 2
        body3.add_property('body_id', type='int', data=body_id)
        body3.add_property('dem_id', type='int', data=dem_id)
        body3.add_constant('total_no_bodies', [3])

        self.scheme.setup_properties([body1, body2, body3])

        body1.add_property('contact_force_is_boundary')
        body1.contact_force_is_boundary[:] = body1.is_boundary[:]
        body2.add_property('contact_force_is_boundary')
        body2.contact_force_is_boundary[:] = body2.is_boundary[:]
        body3.add_property('contact_force_is_boundary')
        body3.contact_force_is_boundary[:] = body2.is_boundary[:]

        self.scheme.scheme.set_linear_velocity(body1, np.array([1., 0., 0.]))
        # self.scheme.scheme.set_linear_velocity(
        #     body2, np.array([-0.5, 0.0, 0.]))
        self.scheme.scheme.set_linear_velocity(body3, np.array([0.0, 0.0, 0.]))
        # print(body1.total_mass)
        # print(body2.total_mass)
        # print(body3.total_mass)
        # print(body1.inertia_tensor_body_frame[8])
        # print(body2.inertia_tensor_body_frame[8])
        # print(body3.inertia_tensor_body_frame[8])


        # print moment of inertia
        return [body1, body2, body3]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['body1', 'body2', 'body3'],
                                 boundaries=None, dim=2)
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        # dt = 0.125 * self.fluid_spacing * self.hdx / (self.co * 1.1)
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.04

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_step(self, solver):
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'wall':
                pa.z += 0.11 * dt

    def customize_output(self):
        self._mayavi_config('''
        for name in ['fluid']:
            b = particle_arrays[name]
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['tank', 'wall']:
            b = particle_arrays[name]
            b.point_size = 0.1
        ''')


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    # app.post_process(app.info_filename)
