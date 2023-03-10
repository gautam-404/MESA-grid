from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
from rich import print, progress
import os
from itertools import repeat
from multiprocessing import Pool


import helper



def evo_star(name, mass, metallicity, v_surf_init, rotation, logging, parallel, convergence_help):
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
    proj.clean()
    proj.make(silent=True)
    phases_params = helper.phases_params(initial_mass, Zinit)     
    phases_names = phases_params.keys()
    terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
    phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]         ## 1E7 is the age when we switch to a coarser timestep
    for phase_name in phases_names:
        star.load_InlistProject(inlist_template)
        print(phase_name)
        star.set(phases_params[phase_name], force=True)
        star.set('max_age', phase_max_age.pop(0), force=True)
        if phase_name == "Pre-MS Evolution":
            if rotation:
                star.set(rotation_init_params, force=True)
            if convergence_help:
                star.set(convergence_helper, force=True)
            proj.run(logging=logging, parallel=parallel)
        else:
            proj.resume(logging=logging, parallel=parallel)

if __name__ == "__main__":
    V = 0
    M = 1.5
    Z = 0.022
    os.environ["OMP_NUM_THREADS"] = "16"
    name = f"test_here/preMS_M{M}_Z{Z}_V{V}"
    evo_star(name, M, Z, V, rotation=False, logging=True, parallel=False, convergence_help=False)
