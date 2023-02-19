import os
import sys

from rich import print
import numpy as np

Y_sun_phot = 0.2485 # Asplund+2009
Y_sun_bulk = 0.2703 # Asplund+2009
Z_sun_phot = 0.0134 # Asplund+2009
Z_sun_bulk = 0.0142 # Asplund+2009
Y_recommended = 0.28 # typical acceptable value, according to Joel Ong TSC2 talk.
dY_by_dZ = 1.4
h2_to_h1_ratio = 2.0E-05
he3_to_he4_ratio = 1.66E-04

dt_limit_values = ['burn steps', 'Lnuc', 'Lnuc_cat', 'Lnuc_H', 'Lnuc_He', 'lgL_power_phot', 'Lnuc_z', 'bad_X_sum',
                  'dH', 'dH/H', 'dHe', 'dHe/He', 'dHe3', 'dHe3/He3', 'dL/L', 'dX', 'dX/X', 'dX_nuc_drop', 'delta mdot',
                  'delta total J', 'delta_HR', 'delta_mstar', 'diff iters', 'diff steps', 'min_dr_div_cs', 'dt_collapse',
                  'eps_nuc_cntr', 'error rate', 'highT del Ye', 'hold', 'lgL', 'lgP', 'lgP_cntr', 'lgR', 'lgRho', 'lgRho_cntr',
                  'lgT', 'lgT_cntr', 'lgT_max', 'lgT_max_hi_T', 'lgTeff', 'dX_div_X_cntr', 'lg_XC_cntr', 'lg_XH_cntr', 
                  'lg_XHe_cntr', 'lg_XNe_cntr', 'lg_XO_cntr', 'lg_XSi_cntr', 'XC_cntr', 'XH_cntr', 'XHe_cntr', 'XNe_cntr',
                  'XO_cntr', 'XSi_cntr', 'log_eps_nuc', 'max_dt', 'neg_mass_frac', 'adjust_J_q', 'solver iters', 'rel_E_err',
                  'varcontrol', 'max increase', 'max decrease', 'retry', 'b_****']

def initial_abundances(Zinit):
    """
    Input: Zinit
    Output: Yinit, initial_h1, initial_h2, initial_he3, initial_he4
    """
    dZ = np.round(Zinit - Z_sun_bulk,4)
    dY = dY_by_dZ * dZ
    Yinit = np.round(Y_recommended + dY,4)
    Xinit = 1 - Yinit - Zinit

    initial_h2 = h2_to_h1_ratio * Xinit
    initial_he3= he3_to_he4_ratio * Yinit
    initial_h1 = (1 - initial_h2) * Xinit
    initial_he4= (1 - initial_he3) * Yinit

    return Yinit, initial_h1, initial_h2, initial_he3, initial_he4

def phases_params(initial_mass, Zinit):
    '''
    Input: initial_mass, Zinit
    Output: phases_params
    '''
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = initial_abundances(Zinit)

    params = { 'Initial Contraction':
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit,
                    'initial_h1': initial_h1,'initial_h2': initial_h2, 
                    'initial_he3': initial_he3, 'initial_he4': initial_he4,
                    'create_pre_main_sequence_model': True, 'pre_ms_T_c': 9e5,
                    'set_initial_model_number' : True, 'initial_model_number' : 0,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'set_rates_preference' : True, 'new_rates_preference' : 2,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'max_model_number': 1, 'max_timestep' : 3.15e13,
                    'delta_lgTeff_limit' : 0.005, 'delta_lgTeff_hard_limit' : 0.01,
                    'delta_lgL_limit' : 0.02, 'delta_lgL_hard_limit' : 0.05,
                    'okay_to_reduce_gradT_excess' : True, 'scale_max_correction' : 0.1},
                    
                    'Pre-Main Sequence' :
                    {'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 1.3e4, 'max_model_number': -1,
                    'delta_lgTeff_limit' : 0.005, 'delta_lgTeff_hard_limit' : 0.01,
                    'delta_lgL_limit' : 0.02, 'delta_lgL_hard_limit' : 0.05,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 50,
                    'relax_dlnZ' : 5.0e-3, 'relax_dY' : 1.0e-2},

                    'Hi-Res Evolution' :
                    {'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 1.25e4, 'max_model_number': -1,
                    'delta_lgTeff_limit' : 0.005, 'delta_lgTeff_hard_limit' : 0.01,
                    'delta_lgL_limit' : 0.02, 'delta_lgL_hard_limit' : 0.05,
                    'write_header_frequency': 10, 'history_interval': 15, 'terminal_interval': 15, 'profile_interval': 15},

                    'Low-Res Evolution' :
                    {'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 0.75e6, 'max_model_number': -1,
                    'delta_lgTeff_limit' : 0.00015, 'delta_lgTeff_hard_limit' : 0.0015,
                    'delta_lgL_limit' : 0.0005, 'delta_lgL_hard_limit' : 0.005,
                    'write_header_frequency': 4, 'history_interval': 4, 'terminal_interval': 4, 'profile_interval': 4},

                    'Late Main Sequence Evolution' :
                    {'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 1e8, 'max_model_number': -1,
                    'delta_lgTeff_limit' : 0.0006, 'delta_lgTeff_hard_limit' : 0.006,
                    'delta_lgL_limit' : 0.002, 'delta_lgL_hard_limit' : 0.02,
                    'write_header_frequency': 1, 'history_interval': 1, 'terminal_interval': 1, 'profile_interval': 1}
    }

    return params

def mute():
    sys.stdout = open(os.devnull, 'w') 

def unmute():
    sys.stdout = sys.__stdout__

def scrap_age(n):
    text = "\n"
    for i in range(n):
        logfile = f"gridwork/work_{i}/run.log"
        age = None
        old_age = 0
        if os.path.exists(logfile):
            with open(logfile, "r") as f:
                for outline in f:
                    try:
                        if outline.split()[-1] in dt_limit_values:
                            age = float(outline.split()[0])
                    except:
                        pass
        if age is not None:
            if age != old_age:
                old_age = age
                if age < 1/365:
                    age_str = f"[b]Age: [cyan]{age*365*24:.4f}[/cyan] hours"
                elif 1/365 < age < 1:
                    age_str = f"[b]Age: [cyan]{age*365:.4f}[/cyan] days"
                elif 1 < age < 1000:
                    age_str = f"[b]Age: [cyan]{age:.3f}[/cyan] years"
                else:
                    age_str = f"[b]Age: [cyan]{age:.3e}[/cyan] years"
                text += f"[b][i]Model[/i] [magenta]{i}[/magenta] [yellow]----->[/yellow] {age_str}\n"
        else:
            text += f"[b][i]Model[/i] [magenta]{i}[/magenta] [yellow]----->[/yellow] Running...\n"
    # print(text)
    return text