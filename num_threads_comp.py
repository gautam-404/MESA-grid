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
import tracemalloc

def evo(num_threads, dirw, n=0):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    v_surf_init = 2
    initial_mass = 1.3
    initial_z = 0.02
    rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                            'new_rotation_flag': True,
                            'change_initial_rotation_flag': True,
                            'set_initial_surface_rotation_v': True,
                            'set_surface_rotation_v': True,
                            'new_surface_rotation_v': v_surf_init,
                            'relax_surface_rotation_v' : True,
                            'num_steps_to_relax_rotation' : 100, ## Default value is 100
                            'set_uniform_am_nu_non_rot': True}
    params = helper.phases_params(initial_mass, initial_z)["Pre-MS Evolution"]     
    terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
    proj = ProjectOps(f"{dirw}/test{n}")
    proj.create(overwrite=True)
    star = MesaAccess(f"{dirw}/test{n}")
    star.load_InlistProject("./templates/inlist_template")
    star.set(params, force=True)
    star.set(rotation_init_params, force=True)
    star.set("max_age", terminal_age, force=True)
    star.set({"convergence_ignore_equL_residuals" : True}, force=True)
    proj.make()
    proj.run()
    return 


def measure_parallel():
    total = 64
    ncpus = 64
    num_threads_list = [1, 8, 16, 24, 32]
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
        tracemalloc.start()
        with Pool(n_processes, initializer=helper.mute) as p:
            with progress.Progress(*helper.progress_columns()) as progressbar:
                task = progressbar.add_task("[b i green]Running...", total=total)
                times = []
                for result in p.istarmap(evo, args):
                    times.append(result)
                    progressbar.advance(task)
        max_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        p_end = time.time()
        total_time = (p_end - p_start)  ## seconds
        print(f"Time taken: {humanize.naturaltime(total_time)}")
        print(f"Max memory used: {humanize.naturalsize(max_mem)}")
        plt.bar(num_threads, total_time, label=str(num_threads))
    plt.legend(title="num threads")
    plt.xlabel("num threads")
    plt.ylabel("time taken (s)")
    plt.savefig("num_threads_parallel.png")

measure_parallel()



        


