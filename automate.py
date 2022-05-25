#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores

import numpy as np
import matplotlib
from pysph.solver.utils import load, get_files

matplotlib.use('pdf')

n_core = 24
n_thread = 24 * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class Dinesh2022RigidBodiesCollisionNewtons3rdLawCheck2D(Problem):
    def get_name(self):
        return 'dinesh_2022_rigid_bodies_collision_newtons_3rd_law_check_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_rigid_bodies_collision_newtons_3rd_law_check_2d.py' + backend

        # Base case info
        self.case_info = {
            'Mohseni_angle_0': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=0,
                ), 'Mohseni angle 0'),

            'Mohseni_Vyas_angle_0': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=0,
                ), 'Mohseni Vyas angle 0'),

            'Mohseni_angle_30': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=30,
                ), 'Mohseni angle 30'),

            'Mohseni_Vyas_angle_30': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=30,
                ), 'Mohseni Vyas angle 30'),

            'Mohseni_angle_60': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=60,
                ), 'Mohseni angle 30'),

            'Mohseni_Vyas_angle_60': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=60,
                ), 'Mohseni Vyas angle 60'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Dinesh2022RigidBodiesCollisionNewtons3rdLawCheck3D(Problem):
    def get_name(self):
        return 'dinesh_2022_rigid_bodies_collision_newtons_3rd_law_check_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_rigid_bodies_collision_newtons_3rd_law_check_3d.py' + backend

        # Base case info
        self.case_info = {
            'Mohseni_angle_0': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=0,
                ), 'Mohseni angle 0'),

            'Mohseni_Vyas_angle_0': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=0,
                ), 'Mohseni Vyas angle 0'),

            'Mohseni_angle_30': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=30,
                ), 'Mohseni angle 30'),

            'Mohseni_Vyas_angle_30': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=30,
                ), 'Mohseni Vyas angle 30'),

            'Mohseni_angle_60': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni',
                angle=60,
                ), 'Mohseni angle 30'),

            'Mohseni_Vyas_angle_60': (dict(
                scheme='rb3d',
                pfreq=10,
                kr=1e5,
                fric_coeff=0.4,
                timestep=5e-5,
                tf=0.015,
                contact_force_model='Mohseni_Vyas',
                angle=60,
                ), 'Mohseni Vyas angle 60'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Amaro2019CollisionBetweenThreeRigidCubes(Problem):
    def get_name(self):
        return 'amaro_2019_collision_between_three_rigid_cubes'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_collision_between_three_rigid_cubes.py' + backend

        # Base case info
        self.case_info = {
            'Mohseni_Vyas': (dict(
                kr=1e5,
                fric_coeff=0.0,
                ), 'Mohseni Vyas'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class StackOfCylinders2D(Problem):
    def get_name(self):
        return 'stack_of_cylinders_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/stack_of_cylinders.py' + backend

        # Base case info
        self.case_info = {
            # 'Mohseni': (dict(
            #     pfreq=100,
            #     contact_force_model='Mohseni',
            #     ), 'Mohseni'),

            'Mohseni_Vyas': (dict(
                pfreq=200,
                timestep=5e-5,
                kr=1e5,
                kf=1e3,
                fric_coeff=0.45,
                ), 'Mohseni Vyas'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class WedgeEntry2D(Problem):
    def get_name(self):
        return '2d_wedge_entry'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/2d_wedge_entry.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                nrbc=None,
                pfreq=200,
                tf=5.,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class WaterEntryOfCylinder2D(Problem):
    def get_name(self):
        return 'water_entry_of_cylinder_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/water_entry_of_cylinder_2d.py' + backend

        # Base case info
        self.case_info = {
            'WCSPH': (dict(
                nrbc=None,
                pfreq=200,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class RigidBodyRotatingAndSinking(Problem):
    def get_name(self):
        return 'rigid_body_rotating_and_sinking'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/rigid_body_rotating_and_sinking_in_tank_2d.py' + backend

        # Base case info
        self.case_info = {
            'WCSPH': (dict(
                nrbc=None,
                pfreq=200,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstSingleCube(Problem):
    def get_name(self):
        return 'dam_break_flow_against_a_single_cube'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_a_single_cube.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstThreeCubes(Problem):
    def get_name(self):
        return 'dam_break_flow_against_three_cubes'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_three_cubes.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakFlowAgainstSixCubes(Problem):
    def get_name(self):
        return 'dam_break_flow_against_six_cubes'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_flow_against_six_cubes.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class BlockFloatingInSteadyTank(Problem):
    def get_name(self):
        return 'block_floating_in_steady_tank_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/block_floating_in_steady_tank_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class BlockSinkingInTank(Problem):
    def get_name(self):
        return 'block_sinking_in_tank_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/block_sinking_in_tank_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class CylidersInWaterCollapsedUnderGravity(Problem):
    def get_name(self):
        return 'cyliders_in_water_collapsed_under_gravity'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/cyliders_in_water_collapsed_under_gravity.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakWithBodyTransport(Problem):
    def get_name(self):
        return 'dam_break_with_body_tranport_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_with_body_tranport_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class DamBreakWithMultipleBodiesTransport(Problem):
    def get_name(self):
        return 'dam_break_with_multiple_bodies_tranport_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break_with_multiple_bodies_tranport_3d.py' + backend

        # Base case info
        self.case_info = {
            'wcsph': (dict(
                pfreq=200,
                dem="canelas",
                # tf=2.0,
                ), 'WCSPH'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022BouncingCube3D(Problem):
    def get_name(self):
        return 'dinesh_2022_bouncing_cube_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_bouncing_cube_3d.py' + backend

        # Base case info
        self.case_info = {
            'coeff_of_restitution_0_2': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.4,
                tf=3.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=3.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0.6': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.6,
                tf=3.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Mohseni2021FreeSlidingOnASlope2D(Problem):
    """
    For pure rigid body problems we use RigidBody3DScheme.
    Scheme used: RigidBody3DScheme
    """
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope_2d.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.2,
                tf=1.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.4,
                tf=1.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=1.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0_6': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.6,
                tf=1.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Mohseni2021FreeSlidingOnASlope3D(Problem):
    def get_name(self):
        return 'mohseni_2021_free_sliding_on_a_slope_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_free_sliding_on_a_slope_3d.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_2': (dict(
                scheme='rb3d',
                pfreq=500,
                kr=1e5,
                fric_coeff=0.2,
                tf=1.,
                ), r'$\mu=$0.2'),

            'fric_coeff_0_4': (dict(
                scheme='rb3d',
                pfreq=500,
                kr=1e5,
                fric_coeff=0.4,
                tf=1.,
                ), r'$\mu=$0.4'),

            'fric_coeff_tan_30': (dict(
                scheme='rb3d',
                pfreq=500,
                kr=1e5,
                fric_coeff=np.tan(np.pi/6),
                tf=1.,
                ), r'$\mu=$tan(30)'),

            'fric_coeff_0_6': (dict(
                scheme='rb3d',
                pfreq=500,
                kr=1e5,
                fric_coeff=0.6,
                tf=1.,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self._plot_particles()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def _plot_particles(self):
        from pysph.solver.utils import get_files
        from mayavi import mlab
        mlab.options.offscreen = True

        name = 'fric_coeff_0_2'
        fname = 'mohseni_2021_free_sliding_on_a_slope_3d'
        print(self.input_path(name))
        output_files = get_files(self.input_path(name), fname)
        print(output_files)
        output_times = np.array([0., 0.5, 1.])
        logfile = os.path.join(self.input_path(name), 'mohseni_2021_free_sliding_on_a_slope_3d.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)
        print(to_plot)

        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        view = None

        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_depth = 0.2

        for i, f in enumerate(to_plot):
            mlab.clf()
            print(i, f)
            data = load(f)
            t = data['solver_data']['t']
            body = data['arrays']['rigid_body']
            wall = data['arrays']['wall']

            bg = mlab.points3d(
                body.x, body.y, body.z, body.m, mode='point',
                colormap='viridis', vmin=-2, vmax=5
            )
            bg.actor.property.render_points_as_spheres = True
            bg.actor.property.point_size = 10

            # get the maximum and minimum of the geometry
            x_min = min(body.x) - self.rigid_body_height
            x_max = max(body.x) + 3. * self.rigid_body_height
            y_min = min(body.y) - 4. * self.rigid_body_height
            y_max = max(body.y) + 1. * self.rigid_body_height

            filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                (wall.y >= y_min) & (wall.y <= y_max))
            wall_x = wall.x[filtr_1]
            wall_y = wall.y[filtr_1]
            wall_z = wall.z[filtr_1]
            wall_m = wall.m[filtr_1]

            bg = mlab.points3d(
                wall_x, wall_y, wall_z, wall_m, mode='point',
                colormap='viridis', vmin=-2, vmax=5
            )
            bg.actor.property.render_points_as_spheres = True
            bg.actor.property.point_size = 10

            mlab.axes()
            cc = mlab.gcf().scene.camera

            if i == 0:
                cc.position = [0.586053487183876, 0.004303849033652898, 1.473259753499654]
                cc.focal_point = [0.06312874772169114, 0.005215998279429021, -0.021841839056437724]
                cc.view_angle = 30.0
                cc.view_up = [0.001971909716611141, 0.9999980445816752, -0.00014968264907690208]
                cc.clipping_range = [0.0020735062077402457, 2.0735062077402455]
                cc.compute_view_plane_normal()
                mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # if i == 1:
            #     cc.position = [1.4627436855170233, -0.03557451059089654, 1.27846398456091]
            #     cc.focal_point = [0.43620275332077596, -0.22339306989490337, -0.04562500946907658]
            #     cc.view_angle = 30.0
            #     cc.view_up = [-0.05918534564203035, 0.9937096256748714, -0.09506984118183866]
            #     cc.clipping_range = [0.008589597475446954, 8.589597475446954]
            #     cc.compute_view_plane_normal()
            #     mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # if i == 2:
            #     cc.position = [2.4030753343216613, -0.7945668802440728, 1.3660548264119496]
            #     cc.focal_point = [1.465669800084762, -0.8684062077241502, -0.03328938156400192]
            #     cc.view_angle = 30.0
            #     cc.view_up = [-0.03405769647694373, 0.9989725637264626, -0.029897997131297732]
            #     cc.clipping_range = [0.006581481386146972, 6.581481386146972]
            #     cc.compute_view_plane_normal()

            mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # save the figure
            figname = os.path.join(self.input_path(name), "time" + str(i) + ".png")
            mlab.savefig(figname)
            # plt.show()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Mohseni2021ControlledSlidingOnAFlatSurface2D(Problem):
    """
    For pure rigid body problems we use RigidBody3DScheme.
    Scheme used: RigidBody3DScheme
    """
    def get_name(self):
        return 'mohseni_2021_controlled_sliding_on_a_flat_surface_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/mohseni_2021_controlled_sliding_on_a_flat_surface_2d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rb3d',
                pfreq=100,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.5,
                tf=1.,
                detailed=None
                ), 'Case 1'),

            'rfc': (dict(
                scheme='rfc',
                pfreq=100,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.5,
                tf=1.,
                detailed=None
                ), 'rfc'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class De2021CylinderRollingOnAnInclinedPlane2d(Problem):
    """
    For pure rigid body problems we use RigidBody3DScheme.
    Scheme used: RigidBody3DScheme
    """
    def get_name(self):
        return 'de_2021_cylinder_rolling_on_an_inclined_plane_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/de_2021_cylinder_rolling_on_an_inclined_plane_2d.py' + backend

        # Base case info
        self.case_info = {
            'fric_coeff_0_3': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.3,
                tf=0.6,
                ), r'$\mu=$0.3'),

            'fric_coeff_0_6': (dict(
                scheme='rb3d',
                pfreq=300,
                kr=1e7,
                kf=1e5,
                fric_coeff=0.6,
                tf=0.6,
                ), r'$\mu=$0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            x_analytical = data[name]['x_analytical']

            t = data[name]['t']
            x_com = data[name]['x_com']

            plt.plot(t_analytical, x_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, x_com, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('xcom_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Dinesh2022MultipleCubesColliding3D(Problem):
    def get_name(self):
        return 'dinesh_2022_multiple_cubes_colliding_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_multiple_cubes_colliding.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes_just_touching_no_gravity': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'two_cubes_overlapping_blow_up': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'multiple_cubes_colliding': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),

            'many_cubes_resting_on_a_big_rigid_cube': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.2,
                tf=3.,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022HydrostaticTank2D(Problem):
    def get_name(self):
        return 'dinesh_2022_hydrostatic_tank_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_hydrostatic_tank_2d.py' + backend

        # Base case info
        self.case_info = {
            # 'dx_0_005': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=5*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.005 m'),

            # 'dx_0_003': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     dx=3*1e-3,
            #     kr=1e5,
            #     fric_coeff=0.0,
            #     tf=0.5,
            #     ), 'dx=0.003 m'),

            'dx_0_002': (dict(
                scheme='rfc',
                pfreq=300,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.002 m'),

            'dx_0_001': (dict(
                scheme='rfc',
                pfreq=300,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.001 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Qiu2017FallingSolidInWater2D(Problem):
    def get_name(self):
        return 'qiu_2017_falling_solid_in_water_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_falling_solid_in_water_2d.py' + backend

        # Base case info
        self.case_info = {
            'dx_0_002': (dict(
                scheme='rfc',
                no_edac=None,
                pfreq=300,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.002 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_experimental = data[name]['t_experimental']
            y_cm_experimental = data[name]['y_cm_experimental']

            t = data[name]['t']
            y_cm_simulated = data[name]['y_cm_simulated']

            plt.scatter(t, y_cm_simulated, s=1, label=self.case_info[name][1])

        # experimental plot should be only once plotted
        plt.plot(t_experimental, y_cm_experimental, label=self.case_info[name][1] + ' Experimental')

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_cm_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Qiu2017FallingSolidInWater3D(Problem):
    def get_name(self):
        return 'qiu_2017_falling_solid_in_water_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_falling_solid_in_water_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                no_edac=None,
                pfreq=300,
                dx=5*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=0.5,
                ), 'dx=0.002 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()
        self._plot_particles()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_experimental = data[name]['t_experimental']
            y_cm_experimental = data[name]['y_cm_experimental']

            t = data[name]['t']
            y_cm_simulated = data[name]['y_cm_simulated']

            plt.scatter(t, y_cm_simulated, s=1, label=self.case_info[name][1])

        # experimental plot should be only once plotted
        plt.plot(t_experimental, y_cm_experimental, label=self.case_info[name][1] + ' Experimental')

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_cm_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)

    def _plot_particles(self):
        from pysph.solver.utils import get_files
        from mayavi import mlab
        mlab.options.offscreen = True

        name = 'case_1'
        fname = 'qiu_2017_falling_solid_in_water_3d'
        output_files = get_files(self.input_path(name), fname)
        output_times = np.array([0., 0.2, 0.3, 0.4])
        logfile = os.path.join(self.input_path(name), 'qiu_2017_falling_solid_in_water_3d.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)

        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
        view = None

        self.rigid_body_length = 0.1
        self.rigid_body_height = 0.1
        self.rigid_body_depth = 0.2

        for i, f in enumerate(to_plot):
            mlab.clf()
            print(i, f)
            data = load(f)
            t = data['solver_data']['t']
            body = data['arrays']['rigid_body']
            wall = data['arrays']['wall']

            bg = mlab.points3d(
                body.x, body.y, body.z, body.m, mode='point',
                colormap='viridis', vmin=-2, vmax=5
            )
            bg.actor.property.render_points_as_spheres = True
            bg.actor.property.point_size = 10

            # get the maximum and minimum of the geometry
            x_min = min(body.x) - self.rigid_body_height
            x_max = max(body.x) + 3. * self.rigid_body_height
            y_min = min(body.y) - 4. * self.rigid_body_height
            y_max = max(body.y) + 1. * self.rigid_body_height

            filtr_1 = ((wall.x >= x_min) & (wall.x <= x_max)) & (
                (wall.y >= y_min) & (wall.y <= y_max))
            wall_x = wall.x[filtr_1]
            wall_y = wall.y[filtr_1]
            wall_z = wall.z[filtr_1]
            wall_m = wall.m[filtr_1]

            bg = mlab.points3d(
                wall_x, wall_y, wall_z, wall_m, mode='point',
                colormap='viridis', vmin=-2, vmax=5
            )
            bg.actor.property.render_points_as_spheres = True
            bg.actor.property.point_size = 10

            mlab.axes()
            cc = mlab.gcf().scene.camera

            if i == 0:
                cc.position = [0.586053487183876, 0.004303849033652898, 1.473259753499654]
                cc.focal_point = [0.06312874772169114, 0.005215998279429021, -0.021841839056437724]
                cc.view_angle = 30.0
                cc.view_up = [0.001971909716611141, 0.9999980445816752, -0.00014968264907690208]
                cc.clipping_range = [0.0020735062077402457, 2.0735062077402455]
                cc.compute_view_plane_normal()
                mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # if i == 1:
            #     cc.position = [1.4627436855170233, -0.03557451059089654, 1.27846398456091]
            #     cc.focal_point = [0.43620275332077596, -0.22339306989490337, -0.04562500946907658]
            #     cc.view_angle = 30.0
            #     cc.view_up = [-0.05918534564203035, 0.9937096256748714, -0.09506984118183866]
            #     cc.clipping_range = [0.008589597475446954, 8.589597475446954]
            #     cc.compute_view_plane_normal()
            #     mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # if i == 2:
            #     cc.position = [2.4030753343216613, -0.7945668802440728, 1.3660548264119496]
            #     cc.focal_point = [1.465669800084762, -0.8684062077241502, -0.03328938156400192]
            #     cc.view_angle = 30.0
            #     cc.view_up = [-0.03405769647694373, 0.9989725637264626, -0.029897997131297732]
            #     cc.clipping_range = [0.006581481386146972, 6.581481386146972]
            #     cc.compute_view_plane_normal()

            mlab.text(0.7, 0.85, f"T = {output_times[i]} sec", width=0.2)

            # save the figure
            figname = os.path.join(self.input_path(name), "time" + str(i) + ".png")
            mlab.savefig(figname)
            # plt.show()


class Qiu2017FloatingSolidInWater2D(Problem):
    def get_name(self):
        return 'qiu_2017_floating_solid_in_water_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_floating_solid_in_water_2d.py' + backend

        # Base case info
        self.case_info = {
            'dx_0_002_edac': (dict(
                scheme='rfc',
                pfreq=300,
                edac=None,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'edac dx=0.002 m'),

            'dx_0_001_edac': (dict(
                scheme='rfc',
                pfreq=300,
                edac=None,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'edac dx=0.001 m'),

            'dx_0_002_no_edac': (dict(
                scheme='rfc',
                pfreq=300,
                no_edac=None,
                dx=2*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'dx=0.002 m'),

            'dx_0_001_no_edac': (dict(
                scheme='rfc',
                pfreq=300,
                no_edac=None,
                dx=1*1e-3,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=1.5,
                ), 'dx=0.001 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_experimental = data[name]['t_experimental']
            y_cm_experimental = data[name]['y_cm_experimental']

            t = data[name]['t']
            y_cm_simulated = data[name]['y_cm_simulated']

            plt.scatter(t, y_cm_simulated, s=1, label=self.case_info[name][1])

        # experimental plot should be only once plotted
        plt.plot(t_experimental, y_cm_experimental, label=self.case_info[name][1] + ' Experimental')

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_cm_vs_time.pdf'))
        plt.clf()
        plt.close()


class Qiu2017FloatingSolidInWater3D(Problem):
    def get_name(self):
        return 'qiu_2017_floating_solid_in_water_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/qiu_2017_floating_solid_in_water_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                dx=5*1e-3,
                kr=1e5,
                fric_coeff=0.2,
                tf=0.5,
                ), r'$\mu=$0.2'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022SteadyCubesOnAWall2D(Problem):
    def get_name(self):
        return 'dinesh_2022_steady_cubes_on_a_wall_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_steady_cubes_on_a_wall_2d.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=1.8,
                ), 'Two cubes'),

            'three_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=2.,
                ), 'Three cubes'),

            'six_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=2.,
                ), 'Six cubes'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022SteadyCubesOnAWall3D(Problem):
    def get_name(self):
        return 'dinesh_2022_steady_cubes_on_a_wall_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_steady_cubes_on_a_wall_3d.py' + backend

        # Base case info
        self.case_info = {
            'two_cubes': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.2,
                tf=1.8,
                ), 'Two cubes'),

            # 'three_cubes': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     kr=1e5,
            #     # This has to be fixed. Friction coefficient between different bodies
            #     fric_coeff=0.2,
            #     tf=2.,
            #     ), 'Three cubes'),

            # 'six_cubes': (dict(
            #     scheme='rfc',
            #     pfreq=300,
            #     kr=1e5,
            #     # This has to be fixed. Friction coefficient between different bodies
            #     fric_coeff=0.2,
            #     tf=2.,
            #     ), 'Six cubes'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022DamBreak3d(Problem):
    def get_name(self):
        return 'dinesh_2022_dam_break_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2022_3d_dam_break.py' + backend

        # Base case info
        self.case_info = {
            'dx_0_002': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.0,
                fluid_alpha=0.2,
                tf=2.,
                ), 'dx=0.002 m'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingOneCube3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_one_cube_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_one_cube_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.05,
                tf=1.8,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingThreeStackedCubes3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_three_stacked_cubes_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_three_stacked_cubes_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                # This has to be fixed. Friction coefficient between different bodies
                fric_coeff=0.05,
                tf=1.8,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Amaro2019DamBreakingFlowHittingSixStackedCubes3d(Problem):
    def get_name(self):
        return 'amaro_2019_dam_breaking_flow_hitting_six_stacked_cubes_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/amaro_2019_dam_breaking_flow_hitting_six_stacked_cubes_3d.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                scheme='rfc',
                pfreq=300,
                kr=1e5,
                fric_coeff=0.05,
                tf=1.8,
                ), 'case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Dinesh2022ParticleBouncingOnAWall2D(Problem):
    def get_name(self):
        return 'dinesh_particle_bouncing_on_a_wall_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_particle_bouncing_on_a_wall_2d.py' + backend

        # Base case info
        self.case_info = {
            'cor_0_2': (dict(
                scheme='rb3d',
                pfreq=100,
                kr=1e7,
                en=0.2,
                gy=0.0,
                tf=0.3,
                ), 'cor=0.2'),

            'cor_0_4': (dict(
                scheme='rb3d',
                pfreq=100,
                kr=1e7,
                en=0.4,
                gy=0.0,
                tf=0.3,
                ), 'cor=0.4'),

            'cor_0_6': (dict(
                scheme='rb3d',
                pfreq=100,
                kr=1e7,
                en=0.6,
                gy=0.0,
                tf=0.3,
                ), 'cor=0.6'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_velocity()
        self.move_figures()

    def plot_velocity(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t_analytical = data[name]['t_analytical']
            v_analytical = data[name]['v_analytical']

            t = data[name]['t']
            velocity_rbd = data[name]['velocity_rbd']

            plt.plot(t_analytical, v_analytical, label=self.case_info[name][1] + ' analytical')
            plt.scatter(t, velocity_rbd, s=1, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_time.pdf'))
        plt.clf()
        plt.close()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class Vyas2021ReboundKinematics(Problem):
    def get_name(self):
        return 'vyas_2021_rebound_kinematics'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/vyas_2021_rebound_kinematics.py' + backend

        spacing = 1e-3
        E = 70 * 1e9
        nu = 0.3
        G = E / (2. * (1. + nu))
        E_star = E / (2. * (1. - nu**2.))
        kr = 1.7 * E_star
        kf = (1. - nu) / (1. - nu/2.) * kr

        kr = 1e9
        kf = kr / (2 * (1. + 0.3))

        fric_coeff = 0.1

        dt = 1e-5
        # Base case info
        self.case_info = {
            'angle_2': (dict(
                spacing=spacing,
                velocity=5.,
                angle=2.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=2.'),

            'angle_5': (dict(
                spacing=spacing,
                velocity=5.,
                angle=5.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=5.'),

            'angle_10': (dict(
                spacing=spacing,
                velocity=5.,
                angle=10.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=10.'),

            'angle_15': (dict(
                spacing=spacing,
                velocity=5.,
                angle=15.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=15.'),

            'angle_20': (dict(
                spacing=spacing,
                velocity=5.,
                angle=20.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=20.'),

            'angle_25': (dict(
                spacing=spacing,
                velocity=5.,
                angle=25.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=25.'),

            'angle_30': (dict(
                spacing=spacing,
                velocity=5.,
                angle=30.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=30.'),

            'angle_35': (dict(
                spacing=spacing,
                velocity=5.,
                angle=35.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=35.'),

            'angle_40': (dict(
                spacing=spacing,
                velocity=5.,
                angle=40.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=40.'),

            'angle_45': (dict(
                spacing=spacing,
                velocity=5.,
                angle=45.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=45.'),

            # 'angle_45_kr_10': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=45.,
            #     kr=10e9,
            #     kf=10e9 / (1. + 0.3),
            #     fric_coeff=fric_coeff,
            #     timestep=dt,
            #     ), 'kr=10'),

            # 'angle_45_kr_20': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=45.,
            #     kr=20e9,
            #     kf=20e9 / (1. + 0.3),
            #     fric_coeff=fric_coeff,
            #     timestep=dt,
            #     ), 'kr=20'),

            # 'angle_45_kr_30': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=45.,
            #     kr=30e9,
            #     kf=30e9 / (1. + 0.3),
            #     fric_coeff=fric_coeff,
            #     timestep=dt,
            #     ), 'kr=30.'),

            'angle_45_kr_40': (dict(
                spacing=spacing,
                velocity=5.,
                angle=45.,
                kr=40e9,
                kf=40e9 / (1. + 0.3),
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'kr=40.'),

            'angle_45_kr_35': (dict(
                spacing=spacing,
                velocity=5.,
                angle=45.,
                kr=35e9,
                kf=35e9 / (1. + 0.3),
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'kr=35.'),

            # 'angle_45_kr_50': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=45.,
            #     kr=50e9,
            #     kf=50e9 / (1. + 0.3),
            #     fric_coeff=fric_coeff,
            #     timestep=dt,
            #     ), 'kr=50.'),

            # 'angle_45_kr_60': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=45.,
            #     kr=60e9,
            #     kf=60e9 / (1. + 0.3),
            #     fric_coeff=fric_coeff,
            #     timestep=dt,
            #     ), 'kr=60.'),

            # 'angle_60': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=60.,
            #     kr=1e8,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=60.'),

            # 'angle_70': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=70.,
            #     kr=1e8,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=70.'),

            # 'angle_80': (dict(
            #     spacing=spacing,
            #     velocity=5.,
            #     angle=80.,
            #     kr=1e8,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=80.'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_theta_vs_omega()

    def plot_theta_vs_omega(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))
            theta_exp = data[name]['theta_exp']
            omega_exp = data[name]['omega_exp']

        non_dim_theta = []
        non_dim_omega = []

        for name in self.case_info:
            non_dim_theta.append(data[name]['non_dim_theta'])
            non_dim_omega.append(data[name]['non_dim_omega'])

        plt.plot(non_dim_theta, non_dim_omega, '^-', label='Simulated')
        plt.plot(theta_exp, omega_exp, 'v-', label='Thornton')
        plt.xlabel('non dimensional theta')
        plt.ylabel('non dimensional Omega')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('theta_vs_omega.pdf'))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('pdf')

    PROBLEMS = [
        # ========================
        # Only rigid body problems
        # ========================
        # Current paper problem
        Dinesh2022RigidBodiesCollisionNewtons3rdLawCheck2D,  # DEM
        Dinesh2022RigidBodiesCollisionNewtons3rdLawCheck3D,  # DEM
        Amaro2019CollisionBetweenThreeRigidCubes,  # DEM
        Mohseni2021FreeSlidingOnASlope2D,  # DEM
        Mohseni2021FreeSlidingOnASlope3D,  # DEM
        Mohseni2021ControlledSlidingOnAFlatSurface2D,  # DEM
        # Mohseni2021ControlledSlidingOnAFlatSurface3D,  # DEM
        De2021CylinderRollingOnAnInclinedPlane2d,  # DEM
        StackOfCylinders2D,  # Experimental validation
        Qiu2017FallingSolidInWater2D,  # RFC
        Qiu2017FallingSolidInWater3D,  # RFC
        # Qiu2017FloatingSolidInWater2D,  # RFC
        # Qiu2017FloatingSolidInWater3D,
        Amaro2019DamBreakingFlowHittingOneCube3d,  # RFC
        Amaro2019DamBreakingFlowHittingThreeStackedCubes3d,  # RFC
        Amaro2019DamBreakingFlowHittingSixStackedCubes3d,  # RFC

        # Current paper problem
        # DineshBouncingParticleOnAWall,

        # Current paper problem
        Vyas2021ReboundKinematics

        # These are test problems for body transport under dam break 3d
        # Dinesh2022DamBreak3d,
        # Dinesh2022HydrostaticTank2D,
        # Dinesh2022MultipleCubesColliding3D,
        # Dinesh2022SteadyCubesOnAWall2D,
        # Dinesh2022SteadyCubesOnAWall3D
        # Dinesh2022BouncingCube3D,  # tests the coefficient of restitution
    ]

    automator = Automator(simulation_dir='outputs',
                          output_dir=os.path.join('manuscript', 'figures'),
                          all_problems=PROBLEMS)

    automator.run()
