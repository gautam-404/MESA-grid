import glob
import os
import shutil
import tarfile
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
    mass, metallicity, v_surf_init, model, gyre, save_model, logging, parallel = args

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

    ## Get Parameters
    terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
    phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]         ## 1E7 is the age when we switch to a coarser timestep
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
    failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
    retry = -1
    dM = [1e-3, 2e-3, -1e-3, -2e-3]
    while retry<len(dM) and failed is True:
        proj.clean()
        proj.make(silent=True)
        phases_params = helper.phases_params(initial_mass, Zinit)     
        phases_names = phases_params.keys()
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
                    proj.run(logging=logging, parallel=parallel)
                else:
                    proj.resume(logging=logging, parallel=parallel)
                failed = False
            except (ValueError, FileNotFoundError) as e:
                failed = True
                print(e)
                break
            except Exception as e:
                failed = True
                print(e)
                print(f"[i red]{phase_name} run failed. Check run log for details.")
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                failed = True
                break
        if failed is True:
            retry += 1
            initial_mass += dM[retry]
            with open(f"{name}/run.log", "a+") as f:
                if retry == len(dM):
                    f.write(f"Max retries reached. Model skipped!\n")
                    break
                f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                f.write(f"Failed at phase: {phase_name}\n")
                f.write(f"Retrying with dM = {dM[retry]}\n")
                f.write(f"New initial mass: {initial_mass}\n")

    if not failed:
        if gyre:   ## Optional, GYRE can berun separately using the run_gyre function    
            run_gyre(dir_name=name, gyre_in=os.path.abspath("templates/gyre_rot_template_dipole.in"), parallel = not parallel)   
            ## Run GYRE on the model. If grid is run in parallel, GYRE is run in serial and vice versa
        
        ## Archive LOGS
        helper.archive_LOGS(name, model, save_model)
    else:
        if logging:         ## If the run failed, archive the log file
            shutil.copy(f"{name}/run.log", f"grid_archive/failed/failed_{model}.log")
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
    processors = int(ray.cluster_resources()["CPU"])
    runtime_env = RuntimeEnv(env_vars={"OMP_NUM_THREADS": str(cpu_per_process), 
                                        "MKL_NUM_THREADS": str(cpu_per_process)})
    ray_remote_args = {"num_cpus": cpu_per_process, "runtime_env": runtime_env, 
                        "scheduling_strategy" : "DEFAULT", 
                        "max_restarts" : -1, "max_task_retries" : -1}
    n_processes = (processors // cpu_per_process)
    # print(workers, cpu_per_process, n_processes)
    length = len(masses)
    if models_list is None:
        models_list = range(1, length+1)
    args = zip(masses, metallicities, v_surf_init_list, models_list,
                    repeat(gyre), repeat(save_model), repeat(logging), 
                    repeat(True))
       
    print(f"[b i][blue]Evolving total {length} stellar models with {n_processes} processes running in parallel.[/blue]")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(ray_address="auto", processes=n_processes, initializer=helper.mute, ray_remote_args=ray_remote_args) as pool:
            for i, res in enumerate(pool.imap_unordered(evo_star, args)):
                progressbar.advance(task)




def rerun_failed(grid_archive, masses, metallicities, v_surf_init_list, cpu_per_process=16,
                gyre=False, save_model=True, logging=True, overwrite=None):
    '''
    Retry failed models
    Args:
        grid_archive (str): grid archive directory
        gyre (optional, bool): whether to run GYRE on the models
        save_model (optional, bool): whether to save the model
        logging (optional, bool): whether to log the evolution
        overwrite (optional, bool): whether to overwrite the grid_archive
    '''
    failed_models = glob.glob(f"{grid_archive}/failed/*")
    print(f"[b i][blue]Running {len(failed_models)} previously failed models.[/blue]")
    failed_masses = []
    failed_metallicities = []
    failed_v_surf_init_list = []
    models_list = []
    for model in failed_models:
        i = int(model.split("/")[-1].split(".")[0].split("_")[-1])
        failed_masses.append(masses[i-1])
        failed_metallicities.append(metallicities[i-1])
        failed_v_surf_init_list.append(v_surf_init_list[i-1])
        models_list.append(i)
        
    run_grid(failed_masses, failed_metallicities, failed_v_surf_init_list, 
            models_list=models_list, cpu_per_process=cpu_per_process, gyre=gyre, 
            save_model=save_model, logging=logging, overwrite=overwrite)
        



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
        # run_grid(masses, metallicities, v_surf_init_list, cpu_per_process=16, overwrite=True)

        ## Run only failed models
        rerun_failed("grid_archive_run1", masses, metallicities, v_surf_init_list, cpu_per_process=16, overwrite=True)

        # ## Run gyre
        # run_gyre(dir_name="grid_archive_old", gyre_in="templates/gyre_rot_template_dipole.in")
    except KeyboardInterrupt:
        print("[b i][red]Grid run aborted.[/red]\n")
        stop_ray()
        print("[b i][red]Ray cluster stopped.[/red]\n")
    except Exception as e:
        import traceback
        import logging
        logging.error(traceback.format_exc())
        print("[b i][red]Encountered an error.[/red]\n")
        stop_ray()
        print("[b i][red]Ray cluster stopped.[/red]\n")

    

