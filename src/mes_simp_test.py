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

# Doesn't seem to actually work
# mkl.set_num_threads(8)

from envs.beacons import BeaconsLightDark, BeaconsLightDarkConfig, BeaconsLightDarkPlotting
from solvers.pftdpw import PFTDPW, PFTDPWConfig, BeliefTreePlotting
from mes_simp.mes_simp import MeasurementSimplification, MeasurementSimplificationConfig, MeasurementSimplificationPlotting
from pomdp_simulation import POMDPSimulation

from inference.particle_filter import ParticleBelief, ParticleFilter
from models.probability_utils import MultivariateNormalDistribution
import numpy as np

from datetime import datetime
from utils.json_numpy import save_dataclass
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt


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


def run_sim(pomdp, iter_num, do_gif, do_viz, sim_gif_prefix, dir_path):
    print("Iteration number: ", str(iter_num))
    pomdp.do_sim(do_gif=do_gif, do_viz=do_viz, file_prefix=sim_gif_prefix, save_dir=dir_path)
    rs_and_phis = pomdp.get_rewards_and_bounds()
    pickle_obj(obj=rs_and_phis, file_name=f'rs_and_phis_{iter_num}', dir_path=dir_path)


if __name__ == "__main__":
    do_viz = True
    do_gif = False
    simplified = True
    
    env_cfg = BeaconsLightDarkConfig()
    solver_cfg = PFTDPWConfig()
    mes_simp_cfg = MeasurementSimplificationConfig()
    
    pomdp_sim = POMDPSimulation(env_cfg=env_cfg, solver_cfg=solver_cfg, mes_simp_cfg=mes_simp_cfg, 
                                do_simp_obs_planning=simplified, do_resample=True)
    
    beacons_env = BeaconsLightDark(cfg=env_cfg)
    beacons_viz = BeaconsLightDarkPlotting(beacons_env)    
    mes_simp = MeasurementSimplification(beacons_env, mes_simp_cfg)
    mes_simp_viz = MeasurementSimplificationPlotting(mes_simp=mes_simp)
    mes_simp.setup_mes_simp()
    
    init_y = np.linspace(2, 7, num=1000)
    x = 5
    m_i = np.zeros(len(init_y))
    
    for j in tqdm(range(len(init_y))):
        y = init_y[j]
        init_state = np.array([x, y])
        num_states = 200
        action = np.array([[0, -1]])
        
        normal_dist_cov_transition = MultivariateNormalDistribution(mean=init_state, cov=beacons_env.cfg.transition_model_cov)
        states = normal_dist_cov_transition.sample(num=num_states).reshape(num_states, 2)
        pb = ParticleBelief(states=states, weights=np.ones(num_states), time=0)
        m_i[j] = mes_simp.estimate_m_i_belief(pb, action)
        
    plt.plot(init_y, m_i)
    plt.show()
    
    
    
