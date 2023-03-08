import glob
import os
import shutil
import tarfile
from itertools import repeat
import subprocess
import traceback
import logging
import time

import numpy as np
from MESAcontroller import MesaAccess, ProjectOps
from rich import console, print, progress
from ray.util.multiprocessing import Pool
import ray
from ray.runtime_env import RuntimeEnv

import helper


def evo_star(args):
    '''
    Run MESA evolution for a single star.
    Args:
        args (tuple): tuple of arguments
            args[0] (float): initial mass
            args[1] (float): metallicity
            args[2] (float): initial surface rotation velocity
            args[3] (str): model number
            args[4] (bool): whether to run GYRE
            args[5] (bool): whether to save the model
            args[6] (bool): whether to log the evolution in a run.log file
            args[7] (bool): whether this function is being run in parallel with ray
    '''
    mass, metallicity, v_surf_init, model, gyre, save_model, logging, parallel, cpu_this_process = args

    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    ## Create working directory
    name = f"gridwork/work_{model}"
    proj = ProjectOps(name)     
    proj.create(overwrite=True) 
    star = MesaAccess(name)
    star.load_HistoryColumns("./templates/history_columns.list")
    star.load_ProfileColumns("./templates/profile_columns.list")

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

    ## Failed Models, CONVERGENCE ISSUE: 
    #   max_residual > tol_max_residual           2    4.1542802224140270D-05    1.0000000000000001D-05
    #   max_residual > tol_max_residual           2    3.3535355297261565D-05    1.0000000000000001D-05
    # Then later: terminated evolution: hydro_failed

    ## What worked:
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

    if not failed:
        if gyre:   ## Optional, GYRE can berun separately using the run_gyre function    
            os.environ['OMP_NUM_THREADS'] = '6'
            ## Run GYRE on multiple profile files parallely
            proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", files='all', data_format="FGONG", 
                        logging=False, parallel=True, n_cores=cpu_this_process)
        
        ## Archive LOGS
        helper.archive_LOGS(name, model, save_model, gyre)
    else:
        if logging:         ## If the run failed, archive the log file
            shutil.copy(f"{name}/run.log", f"grid_archive/failed/failed_{model}.log")
        shutil.rmtree(name)





def run_grid(masses, metallicities, v_surf_init_list, models_list=None, cpu_per_process=16, gyre=False, 
            save_model=True, logging=True, overwrite=None):
    '''
    Run the grid of models.
    Args:
        masses (list): list of initial masses
        metallicities (list): list of metallicities
        v_surf_init_list (list): list of initial surface velocities
        models_list (optional, list): model numbers corresponding to the grid points. 
                                    If None, model numbers will be automatically assigned.
        cpu_per_process (optional, int): number of CPUs to use per process
        gyre (optional, bool): whether to run GYRE on the models
        save_model (optional, bool): whether to save the model after the run
        logging (optional, bool): whether to log the run
        overwrite (optional, bool): whether to overwrite existing "gridwork" and "gridwork" directory. 
                                    If False, the existing "grid_archive" directory will be renamed to "grid_archive_old".
                                    If False, the existing "gridwork" will be overwritten nonetheless.
    '''

    ## Create archive directories
    helper.create_grid_dirs(overwrite=overwrite)

    ## Run grid ##
    length = len(masses)
    if models_list is None:
        models_list = range(1, length+1)
    args = zip(masses, metallicities, v_surf_init_list, models_list,
                    repeat(gyre), repeat(save_model), repeat(logging), 
                    repeat(True), repeat(cpu_per_process))
    ray_pool(evo_star, args, length, cpu_per_process=cpu_per_process)




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


def gyre_parallel(args):
    '''Run GYRE on a tar.gz model'''
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    model, dir_name, gyre_in, cpu_per_process = args
    model = model.split("/")[-1]
    models_archive = os.path.abspath(f"{dir_name}/models")
    gyre_archive = os.path.abspath(f"{dir_name}/gyre/freqs_{model.split('.')[0]}")
    try:
        with helper.cwd(models_archive):
            with tarfile.open(model, "r:gz") as tar:
                tar.extractall()
            name = model.split(".")[0].replace("model", "work")
            work_dir = os.path.abspath(os.path.join(models_archive, name))
            print(f"[b][blue]Running GYRE on[/blue] {name}")
            # Run GYRE
            proj = ProjectOps(name)
            os.environ['OMP_NUM_THREADS'] = '8'

            ## Run GYRE on multiple profile files parallely
            proj.runGyre(gyre_in=gyre_in, files='all', data_format="FGONG", logging=False, parallel=True, n_cores=cpu_per_process)
            
            # ## Run GYRE on each profile file sequentially
            # proj.runGyre(gyre_in=gyre_in, files='all', data_format="FGONG", logging=True, parallel=False, n_cores=cpu_per_process)
    except Exception as e:
        print(f"[b][red]Error running GYRE on[/red] {name}")
        logging.error(traceback.format_exc())
        raise e
    finally:
        ## Archive GYRE output
        os.mkdir(gyre_archive)
        for file in glob.glob(os.path.join(work_dir, "LOGS/*-freqs.dat")):
            shutil.copy(file, gyre_archive)
        ## Compress GYRE output
        compressed_file = f"{gyre_archive}.tar.gz"
        with tarfile.open(compressed_file, "w:gz") as tarhandle:
            tarhandle.add(gyre_archive, arcname=os.path.basename(gyre_archive))
        ## Remove GYRE output
        shutil.rmtree(gyre_archive)
        ## Remove work directory
        for i in range(5):              ## Try 5 times, then give up. NFS is weird. Gotta wait and retry.
            os.system(f"rm -rf {work_dir} > /dev/null 2>&1")
            time.sleep(0.5)             ## Wait for the process, that has the nfs files open, to die/diconnect



def run_gyre(dir_name, gyre_in, cpu_per_process=16):
    '''
    Run GYRE on all models in the archive. OR run GYRE on a single model.
    Args:       
        dir_name (str): archive directory name
        parallel (optional, bool): whether to parallelize the evolution
    '''
    models_archive = os.path.abspath(f"{dir_name}/models/")
    gyre_archive = os.path.abspath(f"{dir_name}/gyre/")
    gyre_in = os.path.abspath(gyre_in)
    models = glob.glob(os.path.join(models_archive, "*.tar.gz"))
    args = zip(models, repeat(dir_name), repeat(gyre_in), repeat(cpu_per_process))
    length = len(models)
    try:
        ray_pool(gyre_parallel, args, length, cpu_per_process=cpu_per_process)
    except KeyboardInterrupt:

        print("[b][red]GYRE interrupted. Cleaning up.")
        [shutil.rmtree(f, ignore_errors=True) for f in glob.glob(os.path.join(gyre_archive, "*"))]
        tmp = os.path.join(models_archive, "work*")
        os.system(f"rm -rf {tmp} > /dev/null 2>&1") ## one of the folders might not be deleted... -_-
        print("[b][red]GYRE stopped.")
        raise KeyboardInterrupt
    except Exception as e:
        print("[b][red]GYRE run failed. Check run logs.")
        raise e

        

def ray_pool(func, args, length, cpu_per_process=16):
    processors = int(ray.cluster_resources()["CPU"])
    runtime_env = RuntimeEnv(env_vars={"OMP_NUM_THREADS": str(cpu_per_process), 
                                        "MKL_NUM_THREADS": str(cpu_per_process)})
    ray_remote_args = {"num_cpus": cpu_per_process, "runtime_env": runtime_env, 
                        "scheduling_strategy" : "DEFAULT", 
                        "max_restarts" : -1, "max_task_retries" : -1}
    n_processes = (processors // cpu_per_process)
    print(f"[b][blue]Running {min([n_processes, length])} parallel processes on {processors} cores.[/blue]")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(ray_address="auto", processes=n_processes, initializer=helper.mute, ray_remote_args=ray_remote_args) as pool:
            for i, res in enumerate(pool.imap_unordered(func, args)):
                progressbar.advance(task)

def stop_ray():
    subprocess.call("ray stop --force".split(" "))
    subprocess.call("killall -9 pbs_tmrsh".split(" "))

def start_ray():
    ## this shell script stops all ray processing before starting new cluster
    subprocess.call("./rayCluster/ray-cluster.sh".split(" "), stdout=subprocess.DEVNULL)

    

if __name__ == "__main__":
    try:
        try:
            ray.init(address="auto")
        except:
            ## Start the ray cluster
            with console.Console().status("[b i][blue]Starting ray cluster...[/blue]") as status:
                start_ray()
                # subprocess.call(["clear"])
            print("[b i][green]Ray cluster started.[/green]\n")
            ray.init(address="auto")
        print("\n[b i][blue]Ray cluster resources:[/blue]")
        print("CPUs: ", ray.cluster_resources()["CPU"])
        print("Memory: ", ray.cluster_resources()["memory"]/1e9, "GB")
        print("Object Store Memory: ", ray.cluster_resources()["object_store_memory"]/1e9, "GB\n")

        ## Initialize grid
        masses, metallicities, v_surf_init_list = init_grid(testrun="grid")

        # ## Run grid
        # run_grid(masses, metallicities, v_surf_init_list, cpu_per_process=24, overwrite=True)

        ## Run gyre
        run_gyre(dir_name="grid_archive_run1", gyre_in="templates/gyre_rot_template_dipole.in", cpu_per_process=48)
    except KeyboardInterrupt:
        print("[b i][red]Grid run aborted.[/red]\n")
        stop_ray()
        print("[b i][red]Ray cluster stopped.[/red]\n")
    except Exception as e:
        logging.error(traceback.format_exc())
        print("[b i][red]Encountered an error.[/red]\n")
        stop_ray()
        print("[b i][red]Ray cluster stopped.[/red]\n")

    

