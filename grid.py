import glob
import multiprocessing as mp
import threading
import time
import os
import shutil
import tarfile
from itertools import repeat

import numpy as np
from MESAcontroller import MesaAccess, ProjectOps
from rich import print, progress, prompt, live, console, panel

import helper

def progress_columns():
    '''Define progress bar columns'''
    progress_columns = (progress.SpinnerColumn(),
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(bar_width=60),
                progress.MofNCompleteColumn(),
                progress.TaskProgressColumn(),
                progress.TimeElapsedColumn())
    return progress_columns

def live_display(n):
    '''Define live display
    Args:   n (int): number of models
    Returns:    live_disp (rich.live.Live): live display
                progressbar (rich.progress.Progress): progress bar
                group (rich.console.Group): group of panels
    '''
    ## Progress bar
    progressbar = progress.Progress(*progress_columns(), disable=False)
    group = console.Group(panel.Panel(progressbar, expand=False), panel.Panel(helper.scrap_age(n), expand=False))
    return live.Live(group), progressbar, group

def update_live_display(live_disp, progressbar, group, n):
    '''Update live display
    Args:   live_disp (rich.live.Live): live display
            progressbar (rich.progress.Progress): progress bar
            group (rich.console.Group): group of panels
            n (int): number of models
            stop (bool): stop live display
    '''
    try:
        while True:
            group = console.Group(panel.Panel(progressbar, expand=False), panel.Panel(helper.scrap_age(n), expand=False))
            time.sleep(0.1)
            live_disp.update(group, refresh=True)
            if stop_thread is True:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt


    
def evo_star(mass, metallicity, coarse_age, v_surf_init=0, model=0, gyre=False,
            save_model=True, logging=True, loadInlists=False, parallel=False, silent=False):
    '''Evolve a star
    Args:   mass (optional, float): mass of star in solar masses
            metallicity (optional, float): metallicity of star
            coarse_age (optional, float): age of star when we switch to a coarser timestep
            v_surf_init (optional, float): initial surface rotation velocity in km/s
            model (optional, int): model number, used by the code to name the model when parallelizing
            gyre (optional, bool): whether to run gyre after evolution
            save_model (optional, bool): whether to save the model
            logging (optional, bool): whether to log the evolution
            loadInlists (optional, bool): whether to load pre-made inlists 
            parallel (optional, bool): whether to parallelize the evolution
            silent (optional, bool): whether to suppress output
    '''
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
    if v_surf_init != 0:
        templates = sorted(glob.glob("./urot/*inlist*"))
        if loadInlists:
            phase_max_age = [1.0E-3, 0.25E6, 1E6, coarse_age, 4.0E7, terminal_age]
            phases_names = ['Initial Contraction', 'Pre-Main Sequence', 'Start Rotation', 'Hi-Res Evolution', 'Low-Res Evolution', 'Late Main Sequence Evolution']
        else:
            phase_max_age = [1.0E-3, 2E6, coarse_age, 4.0E7, terminal_age]
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
    else:
        templates = sorted(glob.glob("./urot/*inlist*"))
        phase_max_age = [1.0E-3, 2E6, coarse_age, 4.0E7, terminal_age]
        phases_names = phases_params.keys()


    inlist_template = "./templates/inlist_template"
    continue_forwards = True
    for phase_name in phases_names:
        try:
            if loadInlists:         ## Run from pre-made inlists
                star.load_InlistProject(templates.pop(0))
                if phase_name == "Start Rotation":
                    continue        ## Skip the start rotation phase, because we started rotation in the contraction phase
            else:                   ## Run from inlist template by setting parameters for each phase
                star.load_InlistProject(inlist_template)
                star.set(phases_params[phase_name], force=True)
            print(phase_name)
            star.set('initial_mass', initial_mass, force=True)
            star.set('initial_z', Zinit, force=True)
            star.set('max_age', phase_max_age.pop(0), force=True)
            if phase_name == "Initial Contraction":
                if v_surf_init != 0:
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
            run_gyre(dir_name=name, gyre_in= os.path.abspath("templates/gyre_rot_template_dipole.in"), parallel=not parallel)   
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
        if logging:
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
        



def run_grid(parallel=False, show_progress=True, testrun=False, create_grid=True,
            gyre=False, save_model=True, loadInlists=False, logging=True, overwrite=None):
    '''
    Run the grid of models.
    Args:
        parallel (optional, bool): whether to parallelize the evolution
        show_progress (optional, bool): whether to show a progress bar
        testrun (optional, bool): whether to run a test run
        create_grid (optional, bool): whether to create the grid
        save_model (optional, bool): whether to save the model
        loadInlists (optional, bool): whether to load inlists from the inlists directory
        logging (optional, bool): whether to log the evolution
        overwrite (optional, bool): whether to overwrite the grid_archive
    '''

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
    os.mkdir("grid_archive/failed")
    ## Create work directory
    if os.path.exists("gridwork"):
        shutil.rmtree("gridwork")
    os.mkdir("gridwork")

    ## Run grid ##
    if parallel:
        ## Run grid in parallel
        ## OMP_NUM_THREADS x n_processes = Total cores available
        n_processes = -(-os.cpu_count() // int(os.environ['OMP_NUM_THREADS']))  ## round up
        # n_processes -= 1   ## leave some breathing room
        length = len(masses)
        args = zip(masses, metallicities, coarse_age_list, v_surf_init_list,
                        range(1, length+1), repeat(gyre), repeat(save_model), repeat(logging), 
                        repeat(loadInlists), repeat(parallel), repeat(True))
        if show_progress:
            live_disp, progressbar, group = live_display(n_processes)
            with live_disp:
                task = progressbar.add_task("[b i green]Running...", total=length)
                try:
                    global stop_thread
                    stop_thread = False
                    thread = threading.Thread(target=update_live_display, 
                                args=(live_disp, progressbar, group, n_processes))
                    thread.start()
                    with mp.Pool(n_processes, initializer=helper.mute) as pool:
                        for proc in pool.starmap(evo_star, args):
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
                for proc in pool.starmap(evo_star, args):
                    progressbar.advance(task)
    else:
        # Run grid in serial
        model = 1
        for mass, metallicity, v_surf_init, coarse_age in zip(masses, metallicities, v_surf_init_list, coarse_age_list):
            print(f"[b i yellow]Running model {model} of {len(masses)}")
            evo_star(mass, metallicity, coarse_age, v_surf_init, model=model, gyre=gyre,
                    save_model=save_model, logging=logging, loadInlists=loadInlists)
            model += 1
            print(f"[b i green]Done with model {model-1} of {len(masses)}")
            # os.system("clear")




def init_grid(testrun=None, create_grid=True):
    '''
    Initialize the grid
    Args:   
        testrun (optional, str): whether to make grid for a test run
        create_grid (optional, bool): whether to create the grid from scratch
    Returns:
            masses (list): list of masses
            metallicities (list): list of metallicities
            coarse_age_list (list): list of coarse ages
            v_surf_init_list (list): list of initial surface velocities
    '''
    def get_grid(sample_masses, sample_metallicities, sample_v_init):
        '''
        Get the grid from the sample lists
        '''
        ## Metallicities: repeat from sample_metallicities for each mass and v_init
        metallicities = np.repeat(sample_metallicities, len(sample_masses)*len(sample_v_init)).astype(float)      
        ## Masses: repeat from sample_masses for each Z and v_init
        masses = np.tile(np.repeat(sample_masses, len(sample_v_init)), len(sample_metallicities)).astype(float)    
        ## v_init: repeat from sample_v_init for each mass and Z
        v_surf_init_list = np.tile(sample_v_init, len(sample_masses)*len(sample_metallicities)).astype(float) 
        return masses, metallicities, v_surf_init_list

    if testrun is not None:
        if testrun == "single":
            v_surf_init_list = [2, 4, 6, 8, 10, 12, 14, 16, 18]
            masses = [1.7]*len(v_surf_init_list)
            metallicities = [0.017]*len(v_surf_init_list)
            coarse_age_list = [1e7]*len(v_surf_init_list)
        if testrun == "grid":
            sample_masses = np.arange(1.30,1.51,0.02)                  ## 1.30 - 1.50 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.0010,0.0101,0.0010)     ## 0.001 - 0.010 (0.001 step)
            sample_v_init = np.arange(2, 20, 2)                        ## 2 - 18 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)
            coarse_age_list = 1E7 * np.ones(len(masses)).astype(float)               
    else:
        if create_grid:
            ## Create grid
            sample_masses = np.arange(1.36, 2.22, 0.02)                ## 1.36 - 2.20 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.001, 0.0101, 0.0001)    ## 0.001 - 0.010 (0.0001 step)
            sample_v_init = np.append(0.2, np.arange(2, 20, 2))                        ## 0.2 and 2 - 18 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)   
            coarse_age_list = 1E7 * np.ones(len(masses)).astype(float)               
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
    parallel = True
    if parallel:
        os.environ['OMP_NUM_THREADS'] = "8"     
        ## Uses 8 logical cores per evolution process, works best for a machine with 16 logical cores i.e. 2 parallel processes.
        ## An optimal balance between OMP_NUM_THREADS and n_processes is required for best performance.
    else:
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())     ## Uses all available logical cores.

    # run grid
    run_grid(parallel=parallel, overwrite=True, testrun="single")

    # # run gyre
    # run_gyre(dir_name="grid_archive_old", gyre_in="templates/gyre_rot_template_dipole.in")

    

