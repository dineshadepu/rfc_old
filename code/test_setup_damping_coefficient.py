from pysph.base.utils import get_particle_array
from rigid_body_common import setup_damping_coefficient
import numpy as np


def create_particle_array(name, x, y, body_id, dem_id, total_mass, total_no_bodies):
    pa = get_particle_array(x=x, y=y, name=name)

    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('dem_id', type='int', data=dem_id)
    pa.add_constant('total_no_bodies', [total_no_bodies])
    pa.add_constant('min_dem_id', min(pa.dem_id))
    pa.add_constant('max_dem_id', max(pa.dem_id))
    pa.add_constant('total_mass', total_mass)

    nb = int(np.max(pa.body_id) + 1)
    pa.add_constant('nb', nb)
    pa.add_constant('eta', nb * pa.total_no_bodies[0])

    return pa


def test_single_rigid_body():
    x = [1., 2.]
    body_id = np.ones_like(x, dtype=int) * 0
    dem_id = np.ones_like(x, dtype=int) * 0
    total_mass = np.array([2.])
    pa = create_particle_array(x=[1., 2.], y=[0., 0.], body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=1, name="body1")

    coeff_of_rest = np.ones(pa.nb[0]*pa.total_no_bodies[0],
                            dtype=float)
    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])


def test_single_particle_array_with_2_rigid_bodies():
    x = [1., 2., 3., 4.]
    body_id = [0, 0, 1, 1]
    dem_id = [0, 0, 1, 1]
    total_mass = np.array([2., 2.])
    pa = create_particle_array(x=x, y=0, body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=2, name="body1")

    coeff_of_rest = np.ones(pa.nb[0]*pa.total_no_bodies[0],
                            dtype=float)
    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])


def test_single_particle_array_with_5_rigid_bodies():
    x = np.linspace(0., 1., 10)
    body_id = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    dem_id = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    total_mass = np.array([2., 2., 2., 2., 2.])
    pa = create_particle_array(x=x, y=0, body_id=body_id,
                               dem_id=dem_id, total_mass=total_mass,
                               total_no_bodies=5, name="body1")

    coeff_of_rest = np.ones(pa.nb[0]*pa.total_no_bodies[0] * 1.,
                            dtype=float)
    pa.add_constant('coeff_of_rest', coeff_of_rest)

    setup_damping_coefficient(pa, [pa], boundaries=[])


if __name__ == '__main__':
    # test_single_rigid_body()
    test_single_particle_array_with_2_rigid_bodies()
    # test_single_particle_array_with_2_rigid_bodies()
