import glob
import multiprocessing as mp
import threading
import time
import os, sys
import shutil
import tarfile
from itertools import repeat

import numpy as np
from MESAcontroller import MesaAccess, ProjectOps
from rich import print, progress, prompt, live, console, panel

import helper

def progress_columns():
    progress_columns = (progress.SpinnerColumn(),
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(bar_width=60),
                progress.MofNCompleteColumn(),
                progress.TaskProgressColumn(),
                progress.TimeElapsedColumn())
    return progress_columns

def live_display(n):
    ## Progress bar
    progressbar = progress.Progress(*progress_columns(), disable=False)
    group = console.Group(panel.Panel(progressbar, expand=False), panel.Panel(helper.scrap_age(n), expand=False))
    return live.Live(group), progressbar, group

def update_live_display(live_disp, progressbar, group, n, stop=False):
    try:
        while True:
            group = console.Group(panel.Panel(progressbar, expand=False), panel.Panel(helper.scrap_age(n), expand=False))
            time.sleep(0.1)
            live_disp.update(group, refresh=True)
            if stop is True:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt


    
def evo_star(mass, metallicity, coarse_age, v_surf_init=0, model=0, rotation=True, 
            save_model=False, logging=True, loadInlists=False, parallel=False, silent=False):
    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    ## Create working directory
    name = f"gridwork/work_{model}"
    proj = ProjectOps(name)     
    proj.create(overwrite=True) 
    proj.make(silent=silent)
    star = MesaAccess(name)
    star.load_HistoryColumns("./templates/history_columns.list")
    star.load_ProfileColumns("./templates/profile_columns.list")

    initial_mass = mass
    Zinit = metallicity

    ## Get Parameters
    terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
    phases_params = helper.phases_params(initial_mass, Zinit)     
    if rotation:
        templates = sorted(glob.glob("./urot/*inlist*"))
        # phase_max_age = [1.0E-3, 0.25E6, 1E6, coarse_age, 4.0E7, terminal_age]
        phase_max_age = [1.0E-3, 2E6, coarse_age, 4.0E7, terminal_age]
        rotation_init_params = {'change_v_flag': True,
                                'new_v_flag': True,
                                'change_rotation_flag': True,
                                'new_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': v_surf_init,
                                'set_uniform_am_nu_non_rot': True}
    else:
        templates = sorted(glob.glob("./inlists/*inlist*"))
        phase_max_age = [1.0E-3, 2E6, coarse_age, 4.0E7, terminal_age]


    inlist_template = "./templates/inlist_template"
    continue_forwards = True
    for phase_name in phases_params.keys():
        try:
            if loadInlists:         ## Run from pre-made inlists
                star.load_InlistProject(templates.pop(0))
            else:                   ## Run from inlist template by setting parameters for each phase
                star.load_InlistProject(inlist_template)
                star.set(phases_params[phase_name], force=True)
            print(phase_name)
            star.set('initial_mass', initial_mass, force=True)
            star.set('initial_z', Zinit, force=True)
            star.set('max_age', phase_max_age.pop(0), force=True)
            if phase_name == "Initial Contraction":
                if rotation:
                    ## Initiate rotation
                    star.set(rotation_init_params, force=True)
                proj.run(logging=logging, parallel=parallel)
            else:
                proj.resume(logging=logging, parallel=parallel)
        except (ValueError, FileNotFoundError) as e:
            continue_forwards = False
            print(e)
            break
        except Exception as e:
            continue_forwards = False
            print(e)
            print(f"[i red]{phase_name} run failed. Check run log for details.")
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    if continue_forwards:
        # Run GYRE
        proj = ProjectOps(name)
        proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", data_format="FGONG", files='all', logging=True, parallel=False)
        # proj.runGyre(gyre_in="templates/gyre_rot_template_l2.in", data_format="FGONG", files='all', logging=True, parallel=False)
        # proj.runGyre(gyre_in="templates/gyre_rot_template_all_modes.in", data_format="FGONG", files='all', logging=True, parallel=False)

        ## Archive LOGS
        os.mkdir(f"grid_archive/gyre/freqs_{model}")
        shutil.copy(f"{name}/LOGS/history.data", f"grid_archive/histories/history_{model}.data")
        shutil.copy(f"{name}/LOGS/profiles.index", f"grid_archive/profiles/profiles_{model}.index")
        for file in glob.glob(f"{name}/LOGS/*-freqs.dat"):
            shutil.copy(file, f"grid_archive/gyre/freqs_{model}")
        if save_model:
            compressed_file = f"grid_archive/models/model_{model}.tar.gz"
            with tarfile.open(compressed_file,"w:gz") as tarhandle:
                tarhandle.add(name, arcname=os.path.basename(name))
        shutil.rmtree(name)
    else:
        exit()
        # shutil.rmtree(name)





def run_grid(parallel=False, show_progress=True, testrun=False, create_grid=True,
            rotation=True, save_model=True, loadInlists=False, logging=True, overwrite=None):
    ## Initialize grid
    masses, metallicities, coarse_age_list, v_surf_init_list = init_grid(testrun=testrun, create_grid=create_grid)

    ## Create archive directories
    if os.path.exists("grid_archive"):
        if overwrite:
            shutil.rmtree("grid_archive")
        else:
            if overwrite is None:
                if prompt.Confirm.ask("Overwrite existing grid_archive?"):
                    shutil.rmtree("grid_archive")
            if os.path.exists("grid_archive"):
                print("Moving old grid_archive(s) to grid_archive_old(:)")
                old = 0
                while os.path.exists(f"grid_archive_old{old}"):
                    old += 1
                    if old >= 3:
                        break
                while old > 0:
                    shutil.move(f"grid_archive_old{old-1}", f"grid_archive_old{old}")
                    old -= 1
                shutil.move("grid_archive", f"grid_archive_old{old}")    
    os.mkdir("grid_archive")
    os.mkdir("grid_archive/models")
    os.mkdir("grid_archive/histories")
    os.mkdir("grid_archive/profiles")
    os.mkdir("grid_archive/gyre")
    ## Create work directory
    if os.path.exists("gridwork"):
        shutil.rmtree("gridwork")
    os.mkdir("gridwork")

    ## Run grid ##
    if parallel:
        ## Run grid in parallel
        ## OMP_NUM_THREADS x n_processes = Total cores available
        n_processes = -(-os.cpu_count() // int(os.environ['OMP_NUM_THREADS']))  ## round up
        n_processes -= 1   ## leave some breathing room
        length = len(masses)
        args = zip(masses, metallicities, coarse_age_list, v_surf_init_list,
                        range(length), repeat(rotation), repeat(save_model), 
                        repeat(logging), repeat(loadInlists), repeat(parallel), repeat(True))
        if show_progress:
            live_disp, progressbar, group = live_display(n_processes)
            with live_disp:
                task = progressbar.add_task("[b i green]Running...", total=length)
                try:
                    stop_thread = False
                    thread = threading.Thread(target=update_live_display, 
                                args=(live_disp, progressbar, group, n_processes, lambda : stop_thread,))
                    thread.start()
                    with mp.Pool(n_processes, initializer=helper.mute) as pool:
                        for proc in pool.istarmap(evo_star, args):
                            progressbar.advance(task)
                    stop_thread = True
                    thread.join()
                except KeyboardInterrupt:
                    os.system("echo && echo KeyboardInterrupt && echo")
                    os._exit(1)
        else:
            with progress.Progress(*progress_columns()) as progressbar,\
                 mp.Pool(n_processes, initializer=helper.mute) as pool:
                task = progressbar.add_task("[b i green]Running...", total=length)
                for proc in pool.istarmap(evo_star, args):
                    progressbar.advance(task)
    else:
        # Run grid in serial
        model = 1
        for mass, metallicity, v_surf_init, coarse_age in zip(masses, metallicities, v_surf_init_list, coarse_age_list):
            print(f"[b i yellow]Running model {model} of {len(masses)}")
            evo_star(mass, metallicity, coarse_age, v_surf_init, model=model, 
                        rotation=rotation, save_model=save_model, logging=logging, loadInlists=loadInlists)
            model += 1
            print(f"[b i green]Done with model {model-1} of {len(masses)}")
            # os.system("clear")




def init_grid(testrun=None, create_grid=True):
    def get_grid(sample_masses, sample_metallicities, sample_v_init):
        ## Metallicities: repeat from sample_metallicities for each mass and v_init
        metallicities = np.repeat(sample_metallicities, len(sample_masses)*len(sample_v_init)).astype(float)      
        ## Masses: repeat from sample_masses for each Z and v_init
        masses = np.tile(np.repeat(sample_masses, len(sample_v_init)), len(sample_metallicities)).astype(float)    
        ## v_init: repeat from sample_v_init for each mass and Z
        v_surf_init_list = np.tile(sample_v_init, len(sample_masses)*len(sample_metallicities)).astype(float) 
        return masses, metallicities, v_surf_init_list

    if testrun is not None:
        if testrun == "single":
            masses = [1.7]
            metallicities = [0.017]
            coarse_age_list = [1E7]
            v_surf_init_list = [0.05]
        if testrun == "grid":
            sample_masses = np.arange(1.30,1.51,0.02)                  ## 1.30 - 1.50 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.0010,0.0101,0.0010)     ## 0.001 - 0.010 (0.001 step)
            sample_v_init = np.append(0.2, np.arange(2, 20, 2))        ## 0.2 and 2 - 18 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)
            coarse_age_list = 1E7 * np.ones(len(masses)).astype(float)               ## 1E6 yr
    else:
        if create_grid:
            ## Create grid
            sample_masses = np.arange(1.36, 2.22, 0.02)                ## 1.36 - 2.20 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.001, 0.0101, 0.0001)    ## 0.001 - 0.010 (0.0001 step)
            sample_v_init = np.append(0.2, np.arange(2, 20, 2))                        ## 0.2 and 2 - 18 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)   
            coarse_age_list = 1E7 * np.ones(len(masses)).astype(float)               ## 1E6 yr
        else:
            ## Load grid
            arr = np.genfromtxt("./templates/coarse_age_map.csv",
                            delimiter=",", dtype=str, skip_header=1)
            masses = arr[:,0].astype(float)
            metallicities = arr[:,1].astype(float)
            coarse_age_list = [age*1E6 if age != 0 else 20*1E6 for age in arr[:,2].astype(float)]
            v_surf_init_list = np.random.randint(1, 10, len(masses)).astype(float) * 30
    return masses, metallicities, coarse_age_list, v_surf_init_list



if __name__ == "__main__":
    # run grid
    run_grid(parallel=False, overwrite=True, testrun="grid")

    

