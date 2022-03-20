"""

"""
import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from rigid_fluid_coupling import RigidFluidCouplingScheme
from geometry import hydrostatic_tank_2d, get_fluid_tank_3d

from pysph.examples.solid_mech.impact import add_properties
from pysph.examples.rigid_body.sphere_in_vessel_akinci import (create_boundary,
                                                               create_fluid,
                                                               create_sphere)
from pysph.tools.geometry import get_3d_block


class Amaro2019DamBreakingFlowHittingOneCube3d(Application):
    def initialize(self):
        spacing = 0.05
        self.hdx = 1.3

        # these dimensions are as per the paper Amaro

        # assuming x-axis as length
        self.fluid_length = 4.5
        # assuming y-axis as depth
        self.fluid_depth = 0.7
        # assuming z-axis as height
        self.fluid_height = 0.4

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_height = 0.7
        self.tank_length = 8.
        self.tank_layers = 3
        self.tank_spacing = spacing

        self.body_height = 0.2
        self.body_length = 0.2
        self.body_depth = 0.2
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
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 3

    def create_one_cube(self):
        xb, yb, zb = get_3d_block(dx=self.body_spacing,
                                  length=self.body_length,
                                  height=self.body_height,
                                  depth=self.body_depth)

        body_id = np.zeros(len(xb), dtype=int)

        dem_id = body_id

        return xb, yb, zb, body_id, dem_id

    def get_boundary_particles(self, no_bodies):
        from boundary_particles import (get_boundary_identification_etvf_equations,
                                        add_boundary_identification_properties)
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.base.kernels import (QuinticSpline)
        # create a row of six cylinders
        x, y, z = get_3d_block(dx=self.body_spacing,
                               length=self.body_length,
                               height=self.body_height,
                               depth=self.body_depth)

        m = self.body_density * self.body_spacing**self.dim
        h = self.hdx * self.body_spacing
        rad_s = self.body_spacing / 2.
        pa = get_particle_array(name='foo',
                                x=x,
                                y=y,
                                z=z,
                                rho=self.body_density,
                                h=h,
                                m=m,
                                rad_s=rad_s,
                                constants={
                                    'E': 69 * 1e9,
                                    'poisson_ratio': 0.3,
                                })

        add_boundary_identification_properties(pa)
        # make sure your rho is not zero
        equations = get_boundary_identification_etvf_equations([pa.name],
                                                               [pa.name])

        sph_eval = SPHEvaluator(arrays=[pa],
                                equations=equations,
                                dim=self.dim,
                                kernel=QuinticSpline(dim=self.dim))

        sph_eval.evaluate(dt=0.1)

        tmp = pa.is_boundary
        is_boundary_tmp = np.tile(tmp, no_bodies)
        is_boundary = is_boundary_tmp.ravel()

        return is_boundary

    def create_particles(self):
        zf, yf, xf, zt, yt, xt = get_fluid_tank_3d(
            self.fluid_length, self.fluid_height, self.fluid_depth,
            self.tank_length, self.tank_height, self.tank_layers,
            self.body_spacing, self.body_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   z=zf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")

        xb, yb, zb, body_id, dem_id = self.create_one_cube()

        m = self.body_density * self.body_spacing**self.dim
        body = get_particle_array(name='body',
                                  x=xb,
                                  y=yb,
                                  z=zb,
                                  h=self.body_h,
                                  m=m,
                                  rho=self.body_density,
                                  m_fluid=m_fluid,
                                  rad_s=self.body_spacing / 2.,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                      'spacing0': self.body_spacing,
                                  })
        body.y[:] += self.body_height * 2.

        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        body.add_constant('total_no_bodies', [max(body_id) + 2])

        # ===============================
        # create a tank
        # ===============================
        x, y, z = xt, yt, zt
        dem_id = body_id
        m = self.body_density * self.body_spacing**self.dim
        h = self.body_h
        rad_s = self.body_spacing / 2.

        tank = get_particle_array(name='tank',
                                  x=x,
                                  y=y,
                                  z=z,
                                  h=h,
                                  m=m,
                                  rho=self.body_density,
                                  rad_s=rad_s,
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
        max_dem_id = max(dem_id)
        tank.add_property('dem_id', type='int', data=max_dem_id + 1)

        # ==================================================
        # adjust the rigid body positions on top of the wall
        # ==================================================
        body.y[:] -= min(body.y) - min(tank.y)
        body.y[:] += self.tank_layers * self.body_spacing
        # ==================================================
        # adjust the rigid body positions in front of fluid
        # ==================================================
        body.z[:] += max(fluid.z) - min(body.z) + self.body_spacing
        # now add 1.7 to the bodies
        body.z[:] += 1.7

        body.x[:] -= min(body.x) - min(fluid.x)
        body.x[:] += 0.275

        self.scheme.setup_properties([body, tank, fluid])

        # reset the boundary particles, this is due to improper boundary
        # particle identification by the setup properties
        is_boundary = self.get_boundary_particles(body.total_no_bodies[0] - 1)
        body.is_boundary[:] = is_boundary[:]

        body.add_property('contact_force_is_boundary')
        body.contact_force_is_boundary[:] = body.is_boundary[:]

        tank.add_property('contact_force_is_boundary')
        tank.contact_force_is_boundary[:] = tank.is_boundary[:]
        return [body, tank, fluid]

    def create_scheme(self):
        rfc = RigidFluidCouplingScheme(rigid_bodies=['body'],
                                       fluids=['fluid'],
                                       boundaries=['tank'],
                                       dim=self.dim,
                                       rho0=self.fluid_density,
                                       p0=1.,
                                       c0=1.,
                                       gx=self.gx,
                                       gy=self.gy,
                                       gz=self.gz,
                                       nu=0.,
                                       alpha=self.alpha,
                                       h=self.h)

        s = SchemeChooser(default='rfc', rfc=rfc)
        return s

    def configure_scheme(self):
        dt = 1e-4
        print("DT: %s" % dt)
        tf = 0.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)


if __name__ == '__main__':
    app = Amaro2019DamBreakingFlowHittingOneCube3d()
    app.run()
    # app.post_process(app.info_filename)
