import warnings
import argparse
import traceback

import os
NP_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = NP_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NP_THREADS
os.environ['MKL_NUM_THREADS'] = NP_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NP_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NP_THREADS
warnings.filterwarnings("error")

from envs.beacons import BeaconsLightDarkConfig
from solvers.pftdpw import PFTDPWConfig
from mes_simp.mes_simp import MeasurementSimplificationConfig
from pomdp_simulation import POMDPSimulation

from datetime import datetime
from utils.json_numpy import save_dataclass
import pickle
import numpy as np


def timestamp():
    return datetime.now().strftime("%d-%m-%Y_%H:%M:%S")


def create_save_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def pickle_obj(obj, file_name='l', dir_path=''):
    pkl_path = os.path.join(dir_path, file_name + '.pkl')
    create_save_dir(dir_path=dir_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(obj, f)


def run_sim(pomdp, iter_num, do_gif, do_viz, save_fig_pdf, sim_gif_prefix, dir_path):
    np.random.seed(iter_num)
    print("Simulation number: ", str(iter_num))
    pomdp.do_sim(do_gif=do_gif, do_viz=do_viz, save_fig_pdf=save_fig_pdf, file_prefix=sim_gif_prefix, save_dir=dir_path)
    rs_and_phis = pomdp.get_rewards_and_bounds()
    pickle_obj(obj=rs_and_phis, file_name=f'rs_and_phis_{iter_num}', dir_path=dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-G", "--GIF", help="Save gif of simulation",
                        action='store_true')
    parser.add_argument("--Dir", help="Directory to save gif", nargs='?', default=None)
    parser.add_argument("-V", "--Visualize", help="Visualize with matplotlib",
                        action='store_true')
    parser.add_argument("-D", "--Debug", help="Debug mode", action='store_true')
    parser.add_argument("-S", "--Simplified", help="Use simplified measurement model for planner",
                        action='store_true')
    parser.add_argument("-R", "--Resample", help="Whether to use resampling in particle filters", action='store_true')
    parser.add_argument("-I", "--Iterations", help="Number of simulations to run", type=int)
    parser.add_argument("-F", "--Figures", help="Number of figures to save", type=int, default=0)
    parser.add_argument("-P", "--Policy", help="Type of policy: 'expected_value' or 'expected_lb' or 'expected_ub'", type=str, default='expected_value')

    # Read arguments from command line
    args = parser.parse_args()
    do_viz = args.Visualize
    do_gif = args.GIF
    save_figs = args.Figures
    save_dir = args.Dir
    simplified = args.Simplified
    num_iters = args.Iterations
    debug_mode = args.Debug
    policy_type = args.Policy
    do_resample = args.Resample
    
    print(f"Running for {num_iters} simulations")
    
    env_cfg = BeaconsLightDarkConfig()
    solver_cfg = PFTDPWConfig(pf_resample=do_resample)
    mes_simp_cfg = MeasurementSimplificationConfig()
    
    create_save_dir(save_dir)
    create_save_dir(os.path.join(save_dir, "pdfs"))
    
    curr_time = timestamp()
    save_dataclass(env_cfg, f"env_cfg_{curr_time}", save_dir)
    save_dataclass(solver_cfg, f"solver_cfg_{curr_time}", save_dir)
    save_dataclass(mes_simp_cfg, f"mes_simp_cfg_{curr_time}", save_dir)

    pomdp_sim = POMDPSimulation(env_cfg=env_cfg, solver_cfg=solver_cfg, mes_simp_cfg=mes_simp_cfg, 
                                do_simp_obs_planning=simplified, do_resample=do_resample, policy_type=policy_type)
    for i in range(num_iters):
        gif_prefix = '_'.join(filter(bool, [str(i), 'Simp' if simplified else 'Orig', 'V' if do_viz else '', 'G' if do_gif else '']))
        if debug_mode:
            run_sim(pomdp_sim, i, do_gif, do_viz, i < save_figs, gif_prefix, save_dir)
        else:
            try:
                run_sim(pomdp_sim, i, do_gif, do_viz, i < save_figs, gif_prefix, save_dir)
            except Exception as e:
                print("Simulation number:", i)
                print(traceback.format_exc())
                print(e)
    
    
