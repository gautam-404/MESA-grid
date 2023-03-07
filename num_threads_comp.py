from MESAcontroller import ProjectOps, MesaAccess
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool
import helper
from rich import print, progress
import humanize


def evo(num_threads, dirw, n=0, gyre=False):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    v_surf_init = 2
    initial_mass = 1.32
    mass = initial_mass
    Zinit = 0.004
    metallicity = Zinit
    name = f"{dirw}/work_{n}"
    proj = ProjectOps(name)     
    proj.create(overwrite=True) 
    star = MesaAccess(name)
    star.load_HistoryColumns("./templates/history_columns.list")
    star.load_ProfileColumns("./templates/profile_columns.list")

    rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                            'new_rotation_flag': True,
                            'change_initial_rotation_flag': True,
                            'set_initial_surface_rotation_v': True,
                            'set_surface_rotation_v': True,
                            'new_surface_rotation_v': v_surf_init,
                            'relax_surface_rotation_v' : True,
                            'num_steps_to_relax_rotation' : 100, ## Default value is 100
                            'set_uniform_am_nu_non_rot': True}
    convergence_helper = {"convergence_ignore_equL_residuals" : True}  

    inlist_template = "./templates/inlist_template"
    failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
    retry = -1
    dM = [0, 1e-3, -1e-3, 2e-3, -2e-3]
    while retry<len(dM) and failed:
        proj.clean()
        proj.make(silent=True)
        phases_params = helper.phases_params(initial_mass, Zinit)     
        phases_names = phases_params.keys()
        terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
        phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]         ## 1E7 is the age when we switch to a coarser timestep
        for phase_name in phases_names:
            try:
                ## Run from inlist template by setting parameters for each phase
                star.load_InlistProject(inlist_template)
                print(phase_name)
                star.set(phases_params[phase_name], force=True)
                star.set('max_age', phase_max_age.pop(0), force=True)
                if phase_name == "Pre-MS Evolution":
                    ## Initiate rotation
                    star.set(rotation_init_params, force=True)
                    if retry>=0:
                        star.set(convergence_helper, force=True)
                    proj.run(logging=True, parallel=True)
                else:
                    proj.resume(logging=True, parallel=True)
            except Exception as e:
                failed = True
                print(e)
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            else:
                failed = False
        if failed:
            retry += 1
            initial_mass = mass + dM[retry]
            with open(f"{name}/run.log", "a+") as f:
                if retry == len(dM)-1:
                    f.write(f"Max retries reached. Model skipped!\n")
                    break
                f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                f.write(f"Failed at phase: {phase_name}\n")
                f.write(f"Retrying with dM = {dM[retry]}\n")
                f.write(f"New initial mass: {initial_mass}\n")
    if not failed:
        if gyre:   ## Optional, GYRE can berun separately using the run_gyre function    
            os.environ['OMP_NUM_THREADS'] = '8'
            ## Run GYRE on multiple profile files parallely
            proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", files='all', data_format="FGONG", 
                                logging=False, parallel=True, n_cores=num_threads)


def measure_parallel():
    total = 1
    ncpus = 128
    num_threads_list = [1, 2, 4, 8, 16, 24, 32]
    for num_threads in num_threads_list:
        name = "test"
        if os.path.exists(name):
            os.system(f"rm -rf {name}")
        os.mkdir(name)
        n_processes = ncpus // num_threads
        print("\nTotal tracks to be run: ", total)
        print("No. of CPUs available:", ncpus)
        print("OMP_NUM_THREADS:", num_threads)
        print("No. of parallel tracks:", min([n_processes, total]))
        args = zip(repeat(num_threads), repeat(name), range(1, total+1))
        p_start = time.time()
        with Pool(n_processes, initializer=helper.mute) as p:
            with progress.Progress(*helper.progress_columns()) as progressbar:
                task = progressbar.add_task("[b i green]Running...", total=total)
                times = []
                for result in p.istarmap(evo, args):
                    times.append(result)
                    progressbar.advance(task)
        p_end = time.time()
        total_time = (p_end - p_start)  ## seconds
        print(f"Time taken: {total_time/60:.2f} minutes")
        plt.bar(num_threads, total_time, label=str(num_threads))
    plt.legend(title="num threads")
    plt.xlabel("num threads")
    plt.ylabel("time taken (s)")
    plt.savefig("num_threads_parallel.png")

measure_parallel()



        


