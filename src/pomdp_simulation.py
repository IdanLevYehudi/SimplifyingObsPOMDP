import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import imageio
import os
from pathlib import Path
from copy import copy
import time
import cv2
import warnings

from envs.beacons import BeaconsLightDark, BeaconsLightDarkPlotting
from mes_simp.mes_simp import MeasurementSimplification, MeasurementSimplificationPlotting
from solvers.pftdpw import PFTDPW, PFTDPWConfig, BeliefTreePlotting
from inference.particle_filter import ParticleBelief, ParticleFilter

from utils.visualization import confidence_ellipse


class POMDPSimulation:
    def __init__(self, env_cfg, solver_cfg, mes_simp_cfg, do_simp_obs_planning=True, do_resample=False, policy_type='expected_value') -> None:
        self.env_cfg = env_cfg
        self.solver_cfg = solver_cfg
        self.mes_simp_cfg = mes_simp_cfg
        
        self.do_simp_obs_planning = do_simp_obs_planning
        self.do_resample = do_resample
        self.policy_type = policy_type
        
        self.beacons_env = BeaconsLightDark(cfg=self.env_cfg)
        self.beacons_viz = BeaconsLightDarkPlotting(self.beacons_env)
            
        self.mes_simp = MeasurementSimplification(self.beacons_env, self.mes_simp_cfg)
        self.mes_simp_viz = MeasurementSimplificationPlotting(mes_simp=self.mes_simp)
        self.init_mes_simp()
        
        planning_obs_model = self.beacons_env.get_simp_obs_model() if self.do_simp_obs_planning else self.beacons_env.get_obs_model()
        self.solver = PFTDPW(self.solver_cfg, self.beacons_env, planning_obs_model, do_mes_simp=do_simp_obs_planning, mes_simp=self.mes_simp)
        self.pb = ParticleFilter(self.beacons_env.transition, self.beacons_env.obs_model)
        
        # Every time step we will append the ith bound over the return
        self.rewards_pZ = list()  # Empirical rewards obtained at the ith time index
        self.q_qZ = list()  # Estimated q value for chosen action at the ith time index
        self.phi = list()  # Estimated phi bound for chosen action at the ith time index
        self.plan_times = list()
        self.ws = list()
        self.policy_diff_lb = list()  # Whether max simplified policy indicates an action different than max lower bound policy
        self.policy_diff_ub = list()  # Same but for upper bound policy
    
    def init_mes_simp(self):
        self.mes_simp.setup_mes_simp()
    
    def draw_figure_to_buffer(self, fig, plot_images, visualize=True):
        fig.canvas.draw()
        if visualize:
            plt.pause(0.05)  # Continue with simulation
        # plt.pause(0)  # Pause for user
        plot_images.append(np.array(fig.canvas.renderer._renderer))

    def save_fig_pdf(self, fig, file_prefix='', save_dir=''):
        plan_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        file_name = '_'.join(filter(bool, ['beacons', file_prefix, plan_time, '.pdf']))
        file_path = os.path.join(save_dir, "pdfs", file_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(file_path, format="pdf", dpi=600, bbox_inches="tight")
    
    def save_gif(self, plot_images, file_prefix='', save_dir=''):
        height, width, layers = plot_images[0].shape
        plan_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        file_name = '_'.join(filter(bool, ['beacons', file_prefix, plan_time, '.avi']))
        file_path = os.path.join(save_dir, file_name)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = 1
        video = cv2.VideoWriter(file_path, fourcc, fps, (width,height))
        for image in plot_images:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
    
    def do_sim(self, do_gif=True, do_viz=True, save_fig_pdf=False, file_prefix='', save_dir=''):
        self.beacons_env.init_state()
        self.init_rewards_and_bounds()
        b = ParticleBelief.belief_from_prior(self.beacons_env.get_prior_model(),
                                         self.beacons_env.get_curr_time(),
                                         self.solver_cfg.num_par)
        
        tree_viz = BeliefTreePlotting()
        if do_gif or do_viz:
            env_ax_width, env_ax_height = self.beacons_viz.env_size_ratio(10)
            fig_width = env_ax_width * 1.33
            fig_height = env_ax_height
            fig1 = plt.figure(1, figsize=(fig_width, fig_height), dpi=600, constrained_layout=True)
            fig1.clf()
            ax_env, ax_actions = fig1.subplots(1, 2, gridspec_kw={'width_ratios':[3, 1]})
            plot_images = []
        if do_viz:
            plt.ion()
            fig2 = plt.figure(2)
            fig2.clf()
            ax2 = fig2.gca()
            
            fig_obs_model = plt.figure(3)
            ax_obs = fig_obs_model.gca()
            obs_model_close = self.beacons_env.obs_model.g_close
            for i in range(len(obs_model_close.gmm.weights)):
                confidence_ellipse(obs_model_close.comp_params[i][0],
                                   obs_model_close.comp_params[i][1],
                                   ax_obs, facecolor='red', 
                                   alpha=(obs_model_close.gmm.weights[i] ** 0.17) * 0.8)
            plt.show(block=False)
            plt.pause(0.05)
            # plt.pause(0.0)
        
        o = None
        while not self.beacons_env.terminated() and not b.is_empty():
            
            b_copy = copy(b)
            # Plan, calculate action, visualize
            plan_time_start = time.time()
            a_id = self.solver.solve(b_copy)
            plan_time_finish = time.time()
            plan_time = plan_time_finish - plan_time_start
            # actions = self.solver.extract_action_sequence()
            # a = actions[0]
            
            a_expt = self.solver.extract_action_at_root(policy='expected_value')
            a_lb = self.solver.extract_action_at_root(policy='expected_lb')
            a_ub = self.solver.extract_action_at_root(policy='expected_ub')
            
            curr_policy_diff_lb = np.any(a_expt != a_lb)
            curr_policy_diff_ub = np.any(a_expt != a_ub)
            
            a = a_expt if self.policy_type == 'expected_value' else a_lb if self.policy_type == 'expected_lb' else a_ub
            
            self.q_qZ.append(self.solver.tree.root.q[a_id] if a_id is not None else 0)
            self.phi.append(self.solver.tree.root.phi[a_id] if a_id is not None else 0)
            self.plan_times.append(plan_time)
            self.ws.append(self.solver.tree.max_unnorm_w())
            self.policy_diff_lb.append(curr_policy_diff_lb)
            self.policy_diff_ub.append(curr_policy_diff_ub)
            if do_gif or do_viz:
                ax_env.cla()
                ax_actions.cla()
                self.beacons_viz.plot_env(ax_env)
                self.mes_simp_viz.plot_delta_states(ax_env)
                self.beacons_viz.plot_states_covs(ax_env, [(b_copy.emp_mean(), b_copy.emp_cov())])
                self.beacons_viz.plot_states(ax_env, b_copy.states, weights=b_copy.weights, color='purple')
                # self.beacons_viz.plot_action_sequence(ax_env, b_copy.emp_mean(), np.vstack(actions))
                self.beacons_viz.plot_action_sequence(ax_env, b_copy.emp_mean(), a.reshape(1, 2))
                # self.beacons_viz.plot_action(ax_env, states=b_copy.states, action=a, weights=b_copy.weights)
                if o is not None:
                    self.beacons_viz.plot_observations(ax_env, o)
                tree_viz.plot_actions_bounds_at_root(ax_actions, self.solver)
            if save_fig_pdf:
                self.save_fig_pdf(fig1, file_prefix, save_dir)
            if do_gif:
                self.draw_figure_to_buffer(fig1, plot_images, visualize=do_viz)
            if do_viz:
                ax2.cla()
                tree_viz.plot_tree(ax2, self.solver.tree)
                plt.show(block=False)
                plt.pause(0.05)
                
            # Apply action, visualize observation and updated pf
            b_copy, _, _ = self.pb.transition(b_copy, a)
            o, r, t = self.beacons_env.step(a)
            self.rewards_pZ.append(r)
            print("cumulative reward:", sum(self.rewards_pZ))
            print("average plan time:", sum(self.plan_times) / len(self.plan_times))
            b_copy = self.pb.observation(b_copy, o, resample=self.do_resample, n_resample=self.solver_cfg.num_par)
            b_copy.clean_frozen()
            b_copy.clean_particles()
            b = b_copy
            if t:
                break
        if do_gif:
            self.save_gif(plot_images, file_prefix, save_dir)
        if do_viz or do_gif:
            plt.close(fig1)
        if do_viz:
            plt.close(fig2)
    
    def init_rewards_and_bounds(self):
        self.rewards_pZ = list()
        self.q_qZ = list()
        self.phi = list()
        self.plan_times = list()
        self.ws = list()
        self.policy_diff_lb = list()
        self.policy_diff_ub = list()
    
    def get_rewards_and_bounds(self):
        return self.rewards_pZ, self.q_qZ, self.phi, self.plan_times, self.ws, self.policy_diff_lb, self.policy_diff_ub
    