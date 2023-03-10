from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
from rich import print, progress
import os, shutil
from itertools import repeat
from multiprocessing import Pool
import glob


import helper



def evo_star(name, mass, metallicity, v_surf_init, logging, parallel, cpu_this_process):
    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    ## Create working directory
    proj = ProjectOps(name)     
    proj.create(overwrite=True) 
    star = MesaAccess(name)
    star.load_HistoryColumns("../templates/history_columns.list")
    star.load_ProfileColumns("../templates/profile_columns.list")

    initial_mass = mass
    Zinit = metallicity
    rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                            'new_rotation_flag': True,
                            'change_initial_rotation_flag': True,
                            'set_initial_surface_rotation_v': True,
                            'set_surface_rotation_v': True,
                            'new_surface_rotation_v': v_surf_init,
                            'relax_surface_rotation_v' : True,
                            'num_steps_to_relax_rotation' : 100, ## Default value is 100
                            'set_uniform_am_nu_non_rot': True}

    convergence_helper = {"convergence_ignore_equL_residuals" : True}  ## Uses max resid dlnE_dt instead

    inlist_template = "../templates/inlist_template"
    failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
    retry = 0
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
                    proj.run(logging=logging, parallel=parallel)
                else:
                    proj.resume(logging=logging, parallel=parallel)
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
    highest_density_profile = sorted(glob.glob(f"{name}/LOGS/profile*.data.FGONG"), 
                                key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1]))[-1]
    highest_density_profile = highest_density_profile.split('/')[-1]
    proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", files=highest_density_profile, data_format="FGONG", 
                        logging=True, parallel=False)
    os.mkdir(f"tests_here/test_star/V_{v_surf_init}")
    shutil.copy(f"{name}/LOGS/{highest_density_profile}", f"tests_here/test_star/V_{v_surf_init}/")
    shutil.copy(f"{name}/LOGS/{highest_density_profile.split('.')[0]}-freqs.dat", f"tests_here/test_star/V_{v_surf_init}/")
    shutil.rmtree(name)

if __name__ == "__main__":
    V = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    M = 1.32
    Z = 0.02
    length = len(V)
    n_cores = os.cpu_count()
    n_procs = length
    cpu_per_process = n_cores//n_procs
    os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(n_procs, initializer=helper.mute) as pool:
            args = zip([f"tests_here/track{i}" for i in range(1, length+1)], repeat(M), repeat(Z), V,
                         repeat(True), repeat(True), repeat(cpu_per_process))
            for _ in pool.istarmap(evo_star, args):
                progressbar.advance(task)