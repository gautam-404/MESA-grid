import glob
import os
import shutil
import tarfile
import threading
import time
from itertools import repeat
import subprocess

import numpy as np
from MESAcontroller import MesaAccess, ProjectOps
from rich import console, print, progress, prompt
from ray.util.multiprocessing import Pool
import ray
from ray.runtime_env import RuntimeEnv

import helper


def evo_star(args):
    '''Evolve a star
    Args:   mass (optional, float): mass of star in solar masses
            metallicity (optional, float): metallicity of star
            v_surf_init (optional, float): initial surface rotation velocity in km/s
            model (optional, int): model number, used by the code to name the model when parallelizing
            gyre (optional, bool): whether to run gyre after evolution
            save_model (optional, bool): whether to save the model
            logging (optional, bool): whether to log the evolution
            parallel (optional, bool): whether to parallelize the evolution
            silent (optional, bool): whether to suppress output
    '''
    mass, metallicity, v_surf_init, model, gyre, save_model, logging, parallel, silent = args

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

    phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]         ## 1E7 is the age when we switch to a coarser timestep
    phases_names = phases_params.keys()
    rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                            'new_rotation_flag': True,
                            'change_initial_rotation_flag': True,
                            'set_initial_surface_rotation_v': True,
                            'set_surface_rotation_v': True,
                            'new_surface_rotation_v': v_surf_init,
                            'relax_surface_rotation_v' : True,
                            'num_steps_to_relax_rotation' : 100, ## Default value is 100
                            'set_uniform_am_nu_non_rot': True}


    inlist_template = "./templates/inlist_template"
    continue_forwards = True
    for phase_name in phases_names:
        try:                 ## Run from inlist template by setting parameters for each phase
            star.load_InlistProject(inlist_template)
            star.set(phases_params[phase_name], force=True)
            print(phase_name)
            star.set('initial_mass', initial_mass, force=True)
            star.set('initial_z', Zinit, force=True)
            star.set('max_age', phase_max_age.pop(0), force=True)
            if phase_name == "Pre-MS Evolution":
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
        if gyre:   ## Optional, GYRE can berun separately using the run_gyre function    
            run_gyre(dir_name=name, gyre_in=os.path.abspath("templates/gyre_rot_template_dipole.in"), parallel = not parallel)   
            ## Run GYRE on the model. If grid is run in parallel, GYRE is run in serial and vice versa
        
        ## Archive LOGS
        shutil.copy(f"{name}/LOGS/history.data", f"grid_archive/histories/history_{model}.data")
        shutil.copy(f"{name}/LOGS/profiles.index", f"grid_archive/profiles/profiles_{model}.index")
        if save_model:
            compressed_file = f"grid_archive/models/model_{model}.tar.gz"
            with tarfile.open(compressed_file,"w:gz") as tarhandle:
                tarhandle.add(name, arcname=os.path.basename(name))
        shutil.rmtree(name)
    else:
        if logging:         ## If the run failed, archive the log file
            shutil.copy(f"{name}/run.log", f"grid_archive/failed/failed_{model}.log")
            with open(f"grid_archive/failed/failed_{model}.log", "a") as f:
                f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                f.write(f"Failed at phase: {phase_name}")
        shutil.rmtree(name)




def run_gyre(dir_name, gyre_in, parallel=True):
    '''
    Run GYRE on all models in the archive. OR run GYRE on a single model.
    Args:       
        dir_name (str): archive directory name
        parallel (optional, bool): whether to parallelize the evolution
    '''
    models_dir = f"{dir_name}/models/"
    gyre_in = os.path.abspath(gyre_in)
    if not os.path.exists(models_dir):
        # Run GYRE
        proj = ProjectOps(dir_name)
        proj.runGyre(gyre_in=gyre_in, data_format="FGONG", files='all', logging=True, parallel=parallel)
    else:
        os.chdir(models_dir)
        models = sorted(glob.glob(f"*tar.gz"))
        for model in models:
            tarfile.open(model, "r:gz").extractall()
            name = model.split(".")[0].replace("model", "work")
            print(f"[b][blue]Running GYRE on[/blue] {name}")
            # Run GYRE
            proj = ProjectOps(name)
            proj.runGyre(gyre_in=gyre_in, data_format="FGONG", files='all', logging=True, parallel=parallel)

            ## Archive GYRE output
            os.mkdir(f"{dir_name}/gyre/freqs_{model}")
            for file in glob.glob(f"{name}/LOGS/*-freqs.dat"):
                shutil.copy(file, f"{dir_name}/gyre/freqs_{model}")
            shutil.rmtree(name)
        os.chdir("../..")
        



def run_grid(masses, metallicities, v_surf_init_list, gyre=False, 
            save_model=True, logging=True, overwrite=None):
    '''
    Run the grid of models.
    Args:
        parallel (optional, bool): whether to parallelize the evolution
        show_progress (optional, bool): whether to show a progress bar
        testrun (optional, bool): whether to run a test run
        create_grid (optional, bool): whether to create the grid
        save_model (optional, bool): whether to save the model
        logging (optional, bool): whether to log the evolution
        overwrite (optional, bool): whether to overwrite the grid_archive
    '''

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
    os.mkdir("grid_archive/failed")
    ## Create work directory
    if os.path.exists("gridwork"):
        shutil.rmtree("gridwork")
    os.mkdir("gridwork")

    ## Run grid ##
    workers = int(ray.cluster_resources()["CPU"])
    cpu_per_worker = 32
    runtime_env = RuntimeEnv(env_vars={"OMP_NUM_THREADS": str(cpu_per_worker), 
                                        "MKL_NUM_THREADS": str(cpu_per_worker)})
    ray_remote_args = {"num_cpus": cpu_per_worker, "runtime_env": runtime_env, 
                        "scheduling_strategy" : "SPREAD", 
                        "max_restarts" : -1, "max_task_retries" : -1}
    n_processes = (workers // cpu_per_worker)
    # print(workers, cpu_per_worker, n_processes)
    length = len(masses)
    args = zip(masses, metallicities, v_surf_init_list,
                    range(1, length+1), repeat(gyre), repeat(save_model), repeat(logging), 
                    repeat(True), repeat(True))
       
    print(f"[b i][blue]Evolving total {length} stellar models with {n_processes} processes running in parallel.[/blue]")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(processes=n_processes, initializer=helper.mute, ray_remote_args=ray_remote_args) as executor:
            results = []
            for i, res in enumerate(executor.imap_unordered(evo_star, args)):
                results.append(res)
                progressbar.advance(task)

        

def init_grid(testrun=None, create_grid=True):
    '''
    Initialize the grid
    Args:   
        testrun (optional, str): whether to make grid for a test run
        create_grid (optional, bool): whether to create the grid from scratch
    Returns:
            masses (list): list of masses
            metallicities (list): list of metallicities
            v_surf_init_list (list): list of initial surface velocities
    '''
    def get_grid(sample_masses, sample_metallicities, sample_v_init):
        '''
        Get the grid from the sample lists
        '''
        masses = np.repeat(sample_masses, len(sample_metallicities)*len(sample_v_init)).astype(float) 
        metallicities = np.tile(np.repeat(sample_metallicities, len(sample_v_init)), len(sample_masses)).astype(float)
        v_surf_init_list = np.tile(sample_v_init, len(sample_masses)*len(sample_metallicities)).astype(float) 
        ### Uncomment to print grid
        # print(list(map(tuple, np.dstack(np.array([masses, metallicities, v_surf_init_list]))[0]))) 
        # exit()    
        return masses, metallicities, v_surf_init_list

    if testrun is not None:
        if testrun == "single":
            v_surf_init_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
            masses = [1.7]*len(v_surf_init_list)
            metallicities = [0.017]*len(v_surf_init_list)
        if testrun == "grid":
            sample_masses = np.arange(1.30, 1.56, 0.02)                  ## 1.30 - 1.54 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.001, 0.013, 0.001)     ## 0.001 - 0.012 (0.001 step)
            sample_v_init = np.arange(0, 20, 2)                          ## 0 - 18 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)
    elif create_grid:
        ## Create grid
        sample_masses = np.arange(1.36, 2.22, 0.02)                ## 1.36 - 2.20 Msun (0.02 Msun step)
        sample_metallicities = np.arange(0.001, 0.0101, 0.0001)    ## 0.001 - 0.010 (0.0001 step)
        sample_v_init = np.append(0.2, np.arange(2, 20, 2))        ## 0.2 and 2 - 18 km/s (2 km/s step)
        masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)                
    else:
        ## Load grid
        arr = np.genfromtxt("./templates/coarse_age_map.csv",
                        delimiter=",", dtype=str, skip_header=1)
        masses = arr[:,0].astype(float)
        metallicities = arr[:,1].astype(float)
        v_surf_init_list = np.random.randint(1, 10, len(masses)).astype(float) * 30
    return masses, metallicities, v_surf_init_list



if __name__ == "__main__":
    try:
        ray.init("auto")
    except:
        ## Start the ray cluster
        try:
            with console.Console().status("[b i][blue]Starting ray cluster...[/blue]") as status:
                res = subprocess.call("./rayCluster/ray-cluster.sh", stdout=subprocess.DEVNULL)
                status.update("[b i][green]Ray cluster started.[/green]")
                subprocess.call(["clear"])
                ray.init("auto")
        except KeyboardInterrupt:
            subprocess.call("ray stop --force".split(" "))
            subprocess.call("killall -9 pbs_tmrsh".split(" "))
            print("[b i][red]Ray cluster setup aborted.[/red]")
            raise KeyboardInterrupt
        
    try:
        ## Initialize grid
        masses, metallicities, v_surf_init_list = init_grid(testrun="grid")

        # run grid
        run_grid(masses, metallicities, v_surf_init_list, overwrite=True)

        # # run gyre
        # run_gyre(dir_name="grid_archive_old", gyre_in="templates/gyre_rot_template_dipole.in")
    except KeyboardInterrupt:
        subprocess.call("ray stop --force".split(" "))
        subprocess.call("killall -9 pbs_tmrsh".split(" "))
        print("[b i][red]Grid run aborted.[/red]")
        print("[b i][red]Stopping ray cluster[/red]")
    except Exception as e:
        import traceback
        import logging
        logging.error(traceback.format_exc())
        subprocess.call("ray stop --force".split(" "))
        subprocess.call("killall -9 pbs_tmrsh".split(" "))
        print("[b i][red]Stopping ray cluster[/red]")

    

