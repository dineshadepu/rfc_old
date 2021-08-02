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


def get_fluid_tank_3d(xf, yf, zf, tank_height, tank_layers,
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
    xt_right += (np.max(xt_right) - np.min(xf)) + 1. * tank_spacing
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
        spacing = 5. * 1e-3
        self.hdx = 1.0

        # the fluid dimensions are
        # x-dimension (this is in the plane of the paper going right)
        self.fluid_length = 150. * 1e-3
        # y-dimension (this goes into the paper)
        self.fluid_width = 140. * 1e-3
        # z-dimension (this is in the plane of the paper going up)
        self.fluid_height = 140. * 1e-3

        self.fluid_density = 1000.0
        self.fluid_spacing = spacing

        self.tank_length = 150. * 1e-3
        self.tank_width = 140. * 1e-3
        self.tank_height = 160. * 1e-3
        self.tank_spacing = spacing
        self.tank_layers = 3

        self.body_length = 20. * 1e-3
        self.body_width = 20. * 1e-3
        self.body_height = 20 * 1e-3
        self.body_density = 2120.
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

        xt, yt, zt = get_fluid_tank_3d(xf, yf, zf, self.tank_height,
                                       self.tank_layers,
                                       self.fluid_spacing)

        m_fluid = self.fluid_density * self.fluid_spacing**self.dim

        fluid = get_particle_array(x=xf,
                                   y=yf,
                                   z=zf,
                                   m=m_fluid,
                                   h=self.h,
                                   rho=self.fluid_density,
                                   name="fluid")
        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_density * self.gz * (max(fluid.z) -
                                                       fluid.z[:])

        tank = get_particle_array(x=xt,
                                  y=yt,
                                  z=zt,
                                  m=m_fluid,
                                  m_fluid=m_fluid,
                                  h=self.h,
                                  rho=self.fluid_density,
                                  rad_s=self.fluid_spacing/2.,
                                  name="tank",
                                  constants={
                                      'E': 69 * 1e9,
                                      'poisson_ratio': 0.3,
                                  })
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
        # print("body spacing", self.body_spacing)
        # print("fluid spacing", self.fluid_spacing)
        xb, yb, zb = get_3d_block(self.body_spacing,
                                  self.body_length,
                                  self.body_width,
                                  self.body_height,)
        xb -= np.min(xb) - np.min(fluid.x)
        xb += 65 * 1e-3 - self.body_spacing/2.
        zb += np.max(zf) - np.min(zb) + self.body_spacing
        zb -= self.body_height/2. + self.body_spacing
        # yb += np.max(fluid.y) - np.min(yb) + 1. * self.body_spacing
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
                                  })
        body_id = np.zeros(len(xb), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_constant('max_tng_contacts_limit', 30)
        body.add_property('dem_id', type='int', data=0)

        # make fluid layers at start at 0
        min_zf = abs(min(fluid.z))
        body.z[:] += min_zf
        fluid.z[:] += min_zf
        tank.z[:] += min_zf

        self.scheme.setup_properties([fluid, tank, body])

        # Remove the fluid particles which are intersecting the gate and
        # gate_support
        # collect the indices which are closer to the stucture
        indices = []
        min_xs = min(body.x)
        max_xs = max(body.x)
        min_ys = min(body.y)
        max_ys = max(body.y)
        min_zs = min(body.z)
        max_zs = max(body.z)

        xf = fluid.x
        yf = fluid.y
        fac = 2. * self.fluid_spacing
        for i in range(len(fluid.x)):
            if xf[i] < max_xs + fac and xf[i] > min_xs - fac:
                if yf[i] < max_ys + fac and yf[i] > min_ys - fac:
                    if zf[i] < max_zs + fac and zf[i] > min_zs - fac:
                        indices.append(i)

        fluid.remove_particles(indices)

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
        tf = 1.5

        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=100)

    def post_step(self, solver):
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'wall':
                pa.z += 0.11 * dt

    def post_process(self, fname):
        """This function will run once per time step after the time step is
        executed. For some time (self.wall_time), we will keep the wall near
        the cylinders such that they settle down to equilibrium and replicate
        the experiment.
        By running the example it becomes much clear.
        """
        from pysph.solver.utils import iter_output
        import os
        if len(self.output_files) == 0:
            return

        files = self.output_files
        print(len(files))
        t = []
        z = []
        for sd, array in iter_output(files[::10], 'body'):
            _t = sd['t']
            t.append(_t)
            # get the system center
            max_z = np.max(array.z)
            z.append(max_z)

        import matplotlib.pyplot as plt
        t = np.asarray(t)
        # t = t - 0.1
        # print(t)

        # data = np.loadtxt('../x_com_zhang.csv', delimiter=',')
        # tx, xcom_zhang = data[:, 0], data[:, 1]

        # plt.plot(tx, xcom_zhang, "s--", label='Simulated PySPH')
        # plt.plot(t, system_x, "s-", label='Experimental')
        # plt.xlabel("time")
        # plt.ylabel("x/L")
        # plt.legend()
        # plt.savefig("xcom", dpi=300)
        # plt.clf()

        # data = np.loadtxt('../y_com_zhang.csv', delimiter=',')
        # ty, ycom_zhang = data[:, 0], data[:, 1]

        # plt.plot(ty, ycom_zhang, "s--", label='Experimental')
        plt.plot(t, z, "s-", label='Simulated PySPH')
        plt.xlabel("time")
        plt.ylabel("z")
        plt.legend()

        fig = os.path.join(os.path.dirname(fname), "max_z.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        particle_arrays['bg'].visible = False
        if 'wake' in particle_arrays:
            particle_arrays['wake'].visible = False
        if 'ghost_inlet' in particle_arrays:
            particle_arrays['ghost_inlet'].visible = False
        for name in ['fluid', 'inlet', 'outlet']:
            b = particle_arrays[name]
            b.scalar = 'p'
            b.range = '-1000, 1000'
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['fluid', 'solid']:
            b = particle_arrays[name]
            b.point_size = 2.0
        ''')


if __name__ == '__main__':
    app = RigidFluidCoupling()
    app.run()
    app.post_process(app.info_filename)
