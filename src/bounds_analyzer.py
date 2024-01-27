import warnings
import argparse
import traceback
import os
import glob
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def safe_index_list(list_of_lists, i):
    try:
        return list_of_lists[i]
    except IndexError:
        list_of_lists.append(list())
        return list_of_lists[i]

class BatchBeaconPOMDPData:
    def __init__(self) -> None:
        self.empirical_Q = list()
        self.expected_Q = list()
        self.expected_Phi = list()
        self.plan_durations = list()
        self.max_ws = list()
        self.policy_diff_lb = list()
        self.policy_diff_ub = list()
        
        self.empirical_Q_per_time = list()
        self.expected_Q_per_time = list()
        self.expected_Phi_per_time = list()
        self.plan_durations_per_time = list()
        self.max_ws_per_time = list()
        self.policy_diff_lb_per_time = list()
        self.policy_diff_ub_per_time = list()
    
    def populate_data(self, data):
        ws = None
        policy_diff_lb = None
        policy_diff_ub = None
        
        if len(data) == 4:
            empirical_rewards, estimated_returns, estimated_phi, plan_times = data
        elif len(data) == 5:
            empirical_rewards, estimated_returns, estimated_phi, plan_times, ws = data
        elif len(data) == 6:
            empirical_rewards, estimated_returns, estimated_phi, plan_times, ws, policy_diff_lb = data
        elif len(data) == 7:
            empirical_rewards, estimated_returns, estimated_phi, plan_times, ws, policy_diff_lb, policy_diff_ub = data
            
        empirical_returns = np.cumsum(np.array(empirical_rewards)[::-1])[::-1]
        
        self.empirical_Q.append(empirical_returns)
        self.expected_Q.append(estimated_returns)
        self.expected_Phi.append(estimated_phi)
        self.plan_durations.append(plan_times)
        
        if ws is not None:
            self.max_ws.append(ws)
        if policy_diff_lb is not None:
            self.policy_diff_lb.append(policy_diff_lb)
        if policy_diff_ub is not None:
            self.policy_diff_ub.append(policy_diff_ub)
        
        for i in range(len(empirical_returns)):
            safe_index_list(self.empirical_Q_per_time, i).append(empirical_returns[i])
            safe_index_list(self.expected_Q_per_time, i).append(estimated_returns[i])
            safe_index_list(self.expected_Phi_per_time, i).append(estimated_phi[i])
            safe_index_list(self.plan_durations_per_time, i).append(plan_times[i])
            if ws is not None:
                safe_index_list(self.max_ws_per_time, i).append(ws[i])
            if policy_diff_lb is not None:                    
                safe_index_list(self.policy_diff_lb_per_time, i).append(policy_diff_lb[i])
            if policy_diff_ub is not None:                    
                safe_index_list(self.policy_diff_ub_per_time, i).append(policy_diff_ub[i])
                
    def read_files(self, dir_path, file_pattern='rs_and_phis_*.pkl'):
        file_paths = glob.glob(os.path.join(dir_path, file_pattern))
        for file_path in file_paths:
            data = None
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.populate_data(data)

class BatchBeaconPOMDPAnalyzer:
    simp_pi_qz: BatchBeaconPOMDPData
    simp_pi_lb: BatchBeaconPOMDPData
    simp_pi_ub: BatchBeaconPOMDPData
    orig_pi_pz: BatchBeaconPOMDPData
    
    def __init__(self, data_holder_simp_pi_qz, data_holder_simp_pi_lb, data_holder_simp_pi_ub, data_holder_orig_pi_pz) -> None:
        self.simp_pi_qz = data_holder_simp_pi_qz
        self.simp_pi_lb = data_holder_simp_pi_lb
        self.simp_pi_ub = data_holder_simp_pi_ub
        self.orig_pi_pz = data_holder_orig_pi_pz

    def max_w(self):
        # Prints the maximum unnormalized particle weights
        print("Max weight for q:", max([max(arr) for arr in self.simp_pi_qz.max_ws]))
        print("Max weight for q lb:", max([max(arr) for arr in self.simp_pi_lb.max_ws]))
        print("Max weight for q ub:", max([max(arr) for arr in self.simp_pi_ub.max_ws]))
        print("Max weight for p:", max([max(arr) for arr in self.orig_pi_pz.max_ws]))
        
    def policy_performances(self):
        # Prints the mean and std of each policy performance
        returns_simp_pi_qz = np.array(self.simp_pi_qz.empirical_Q_per_time[0])
        returns_simp_pi_lb = np.array(self.simp_pi_lb.empirical_Q_per_time[0])
        returns_simp_pi_ub = np.array(self.simp_pi_ub.empirical_Q_per_time[0])
        returns_orig_pi_pz = np.array(self.orig_pi_pz.empirical_Q_per_time[0])
        
        bugs_simp_pi_qz = np.logical_and(returns_simp_pi_qz < 85, returns_simp_pi_qz > -50)
        bugs_simp_pi_lb = np.logical_and(returns_simp_pi_lb < 85, returns_simp_pi_lb > -50)
        bugs_simp_pi_ub = np.logical_and(returns_simp_pi_ub < 85, returns_simp_pi_ub > -50)
        bugs_orig_pi_pz = np.logical_and(returns_orig_pi_pz < 85, returns_orig_pi_pz > -50)

        print(f"Num bugs qz: {np.count_nonzero(bugs_simp_pi_qz)}")
        print(f"Num bugs lb: {np.count_nonzero(bugs_simp_pi_lb)}")
        print(f"Num bugs ub: {np.count_nonzero(bugs_simp_pi_ub)}")
        print(f"Num bugs pz: {np.count_nonzero(bugs_orig_pi_pz)}")
        
        # Assume bugs to be unterminated scenarios, therefore set them to -64 which is maximal miss penalty without collision
        # returns_simp_pi_qz[bugs_simp_pi_qz] = -64
        # returns_simp_pi_lb[bugs_simp_pi_lb] = -64
        # returns_simp_pi_ub[bugs_simp_pi_ub] = -64
        # returns_orig_pi_pz[bugs_orig_pi_pz] = -64
        
        # We choose to omit the bugged scenarios, to not skew the distributions
        returns_simp_pi_qz = returns_simp_pi_qz[~bugs_simp_pi_qz]
        returns_simp_pi_lb = returns_simp_pi_lb[~bugs_simp_pi_lb]
        returns_simp_pi_ub = returns_simp_pi_ub[~bugs_simp_pi_ub]
        returns_orig_pi_pz = returns_orig_pi_pz[~bugs_orig_pi_pz]

        print(f"Pi qz mean: {np.mean(returns_simp_pi_qz)}, std: {np.std(returns_simp_pi_qz, ddof=1)}")
        print(f"Pi lb mean: {np.mean(returns_simp_pi_lb)}, std: {np.std(returns_simp_pi_lb, ddof=1)}")
        print(f"Pi ub mean: {np.mean(returns_simp_pi_ub)}, std: {np.std(returns_simp_pi_ub, ddof=1)}")
        print(f"Pi pz mean: {np.mean(returns_orig_pi_pz)}, std: {np.std(returns_orig_pi_pz, ddof=1)}")
        
        successes_simp_pi_qz = returns_simp_pi_qz > 85
        successes_simp_pi_lb = returns_simp_pi_lb > 85
        successes_simp_pi_ub = returns_simp_pi_ub > 85
        successes_orig_pi_pz = returns_orig_pi_pz > 85
        
        print(f"Pi qz success rate: {np.count_nonzero(successes_simp_pi_qz)/len(returns_simp_pi_qz)}")
        print(f"Pi lb success rate: {np.count_nonzero(successes_simp_pi_lb)/len(returns_simp_pi_lb)}")
        print(f"Pi ub success rate: {np.count_nonzero(successes_simp_pi_ub)/len(returns_simp_pi_ub)}")
        print(f"Pi pz success rate: {np.count_nonzero(successes_orig_pi_pz)/len(returns_orig_pi_pz)}")
        
        lengths_simp_pi_qz = np.array([len(arr) for arr in self.simp_pi_qz.empirical_Q])
        lengths_simp_pi_lb = np.array([len(arr) for arr in self.simp_pi_lb.empirical_Q])
        lengths_simp_pi_ub = np.array([len(arr) for arr in self.simp_pi_ub.empirical_Q])
        lengths_orig_pi_pz = np.array([len(arr) for arr in self.orig_pi_pz.empirical_Q])
        
        print(f"Pi qz mean length: {np.mean(lengths_simp_pi_qz)}")
        print(f"Pi lb mean length: {np.mean(lengths_simp_pi_lb)}")
        print(f"Pi ub mean length: {np.mean(lengths_simp_pi_ub)}")
        print(f"Pi pz mean length: {np.mean(lengths_orig_pi_pz)}")
        
        
    def visualize_bounds_fill_between(self, savefig):
        # The bounds (in theory) should satisfy |Qp - Qq| <= Phi (+ eps which we don't know)
        # Therefore -Phi <= Qp - Qq <= Phi
        alpha_plot = 0.8
        line_width = 0.5
        alpha_error = 0.1
        alpha_error_epsilon = 0.08
        epsilon = 30
        
        max_horizon = max([len(arr) for arr in self.simp_pi_qz.empirical_Q])
        
        fig_fill_between = plt.figure(figsize=(3,1.5), dpi=300)
        ax_fill_between = fig_fill_between.gca()
        for i in range(len(self.simp_pi_qz.empirical_Q)):
            sim_length_i = len(self.simp_pi_qz.empirical_Q[i])
            time_steps = np.arange(sim_length_i)
            empirical_diff = np.array(self.simp_pi_qz.empirical_Q[i]) - np.array(self.simp_pi_qz.expected_Q[i])
            phis = np.array(self.simp_pi_qz.expected_Phi[i])
            ax_fill_between.plot(time_steps, empirical_diff, '-', color='blue', linewidth=line_width, alpha=alpha_plot, zorder=5)
            ax_fill_between.fill_between(time_steps, -phis, phis, color='lightblue', alpha=alpha_error, zorder=10)
            ax_fill_between.fill_between(time_steps, -phis - epsilon, phis + epsilon, color='lightgray', alpha=alpha_error_epsilon, zorder=0)
        
        ax_fill_between.set_xlim(0, max_horizon)
        ax_fill_between.set_ylim(-20, 70)
        ind = np.arange(max_horizon)
        
        # ax_fill_between.set_title('Simulation Empirical Value Function vs. Bounds')
        ax_fill_between.set_xticks(ind, [str(i) for i in ind])
        ax_fill_between.set_xlabel('Time Step')
        ax_fill_between.set_ylabel('Value Diff. and Bounds')
        legend = ax_fill_between.legend(labels=['Qp-Qq', 'Phi', 'Phi+eps'], loc='upper right')
        for handle in legend.legendHandles:
            handle._alpha = 1
        if savefig:
            fig_fill_between.savefig('bounds_fill_between.pdf', format='pdf', bbox_inches='tight')
        
    def visualize_bounds_binary_bars(self, savefig):
        epsilon = 30
        
        max_horizon = len(self.simp_pi_qz.empirical_Q_per_time)        
        num_sims_per_time = np.array([len(arr) for arr in self.simp_pi_qz.empirical_Q_per_time])
        sims_in_bounds_per_time = [[]] * max_horizon
        sims_in_bounds_plus_epsilon_per_time = [[]] * max_horizon
        for i in range(len(self.simp_pi_qz.empirical_Q_per_time)):
            empirical_returns_i = np.array(self.simp_pi_qz.empirical_Q_per_time[i])
            expected_returns_i = np.array(self.simp_pi_qz.expected_Q_per_time[i])
            expected_phi_i = np.array(self.simp_pi_qz.expected_Phi_per_time[i])
            sims_in_bounds_per_time[i] = np.count_nonzero(np.logical_and(empirical_returns_i - expected_returns_i <= expected_phi_i, 
                                                                         empirical_returns_i + expected_returns_i >= -expected_phi_i))
            sims_in_bounds_plus_epsilon_per_time[i] = np.count_nonzero(np.logical_and(empirical_returns_i - expected_returns_i <= expected_phi_i + epsilon, 
                                                                                      empirical_returns_i + expected_returns_i >= -expected_phi_i - epsilon))
    
        fig_bounds_bars = plt.figure(figsize=(3,1.5), dpi=300)
        ax_bounds_bars = fig_bounds_bars.gca()
        # width = 0.3
        ind = np.arange(max_horizon)
        
        # ax_bounds_bars.bar(ind, sims_in_bounds_plus_epsilon_per_time, width, color='skyblue', label=r'$|Q_{t}^{p_Z} - Q_{t}^{q_Z}| \leq \Phi_t + \varepsilon$ \text{Holding}', zorder=5)
        # ax_bounds_bars.bar(ind, num_sims_per_time, width, color='orangered', label='Total Simulations Number', zorder=0)
        # ax_bounds_bars.bar(ind, sims_in_bounds_plus_epsilon_per_time, width, color='skyblue', label='Bounds + Eps=30 Holding', zorder=5)
        # ax_bounds_bars.bar(ind, sims_in_bounds_per_time, width, color='palegreen', label='Bounds Holding', zorder=10)
        
        ax_bounds_bars.bar(ind, num_sims_per_time, color='orangered', label='Not in', zorder=0)
        ax_bounds_bars.bar(ind, sims_in_bounds_plus_epsilon_per_time, color='skyblue', label='Phi+eps', zorder=5)
        ax_bounds_bars.bar(ind, sims_in_bounds_per_time, color='palegreen', label='Phi', zorder=10)
        
        # ax_bounds_bars.set_title('Bounds Holding per Time Step')
        ax_bounds_bars.set_xticks(ind, [str(i) for i in ind])
        ax_bounds_bars.set_xlabel('Time Step')
        ax_bounds_bars.set_ylabel('No. of Simulations')
        ax_bounds_bars.legend(loc='best')
        if savefig:
            fig_bounds_bars.savefig('bounds_bars.pdf', format='pdf', bbox_inches='tight')
    
    def visualize_time_statistics(self, savefig):
        avg_durations_q = list()
        std_durations_q = list()
        avg_durations_p = list()
        std_durations_p = list()
        for durations_list in self.simp_pi_qz.plan_durations_per_time:
            avg_durations_q.append(np.mean(np.array(durations_list)))
            std_durations_q.append(np.std(np.array(durations_list), ddof=1))
        for durations_list in self.orig_pi_pz.plan_durations_per_time:
            avg_durations_p.append(np.mean(np.array(durations_list)))
            std_durations_p.append(np.std(np.array(durations_list), ddof=1))
        
        avg_durations_q = np.array(avg_durations_q)
        avg_durations_p = np.array(avg_durations_p)
        
        num_bars = max(len(avg_durations_q), len(avg_durations_p))
        ind = np.arange(num_bars)
        
        fig_time = plt.figure(figsize=(3,1.7), dpi=600)
        ax_time = fig_time.gca()
        
        width = 0.5
        ax_time.bar(np.arange(len(avg_durations_q)), avg_durations_q, width=0.9*width, yerr=std_durations_q, color='deepskyblue', label='Simplified $q_Z$')
        ax_time.bar(np.arange(len(avg_durations_p)) + width, avg_durations_p, width=0.9*width, yerr=std_durations_p, color='orange', label='Original $p_Z$')
        # ax_time.set_title('Mean Planning Durations per Time Step')
        ax_time.set_xticks(ind + width/2, [str(i) if i % 2 == 0 else '' for i in ind])
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Plan Time (secs)')
        ax_time.legend(loc='best')
        if savefig:
            fig_time.savefig('time_statistics.pdf', format='pdf', bbox_inches='tight')
            
    def visualize_order_statistics(self, savefig, data: BatchBeaconPOMDPData, save_name='order_statistics.pdf'):
        num_sims_per_time = np.array([len(arr) for arr in data.policy_diff_lb_per_time], dtype=float)
        lb_order_invs_per_time = np.array([np.count_nonzero(np.array(arr)) for arr in data.policy_diff_lb_per_time], dtype=float)
        ub_order_invs_per_time = np.array([np.count_nonzero(np.array(arr)) for arr in data.policy_diff_ub_per_time], dtype=float)
        precent_invs_lb = lb_order_invs_per_time / num_sims_per_time
        precent_invs_ub = ub_order_invs_per_time / num_sims_per_time
        ind = np.arange(len(lb_order_invs_per_time))
        
        fig_order = plt.figure(figsize=(3,1.7), dpi=600)
        ax_order = fig_order.gca()
        ax_order.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        
        width = 0.5
        ax_order.bar(ind, precent_invs_lb, width=width*0.9, color='skyblue', label="$\pi^{\mathcal{LB}}$" )
        ax_order.bar(ind + width, precent_invs_ub, width=width*0.9, color='orangered', label="$\pi^{\mathcal{UB}}$")
        ax_order.set_xticks(ind + width/2, [str(i) if i % 2 == 0 else '' for i in ind])
        ax_order.set_xlabel('Time Step')
        ax_order.set_ylabel('\% Action Inconsistency')
        ax_order.legend(loc='best')
        if savefig:
                fig_order.savefig(save_name, format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--Dir-Simp", help="Directory of simplified planning", nargs='?', default=None)
    # parser.add_argument("--Dir-Orig", help="Directory of simplified planning", nargs='?', default=None)
    # parser.add_argument("-V", "--Visualize", help="Visualize with matplotlib", action='store_true')
    # # Read arguments from command line
    # args = parser.parse_args()
    # do_viz = args.Visualize
    # dir_simp = args.Dir_Simp
    # dir_orig = args.Dir_Orig
    
    save = True
    
    # dir_simp = os.path.join('gifs', '23-08-09-stats-simp')
    # dir_orig = os.path.join('gifs', '23-08-09-stats-orig')
    
    dir_simp = os.path.join('gifs', '23-08-15-stats-resample-simp')
    dir_simp_lb = os.path.join('gifs', '23-08-15-stats-resample-lb')
    dir_simp_ub = os.path.join('gifs', '23-08-15-stats-resample-ub')
    dir_orig = os.path.join('gifs', '23-08-15-stats-resample-orig')
    
    plt.rcParams['text.usetex'] = True
    SMALLER_SIZE = 9
    SMALL_SIZE = 10
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    data_simp = BatchBeaconPOMDPData()
    data_simp.read_files(dir_simp)
    data_simp_lb = BatchBeaconPOMDPData()
    data_simp_lb.read_files(dir_simp_lb)
    data_simp_ub = BatchBeaconPOMDPData()
    data_simp_ub.read_files(dir_simp_ub)
    data_orig = BatchBeaconPOMDPData()
    data_orig.read_files(dir_orig)
        
    analyzer = BatchBeaconPOMDPAnalyzer(data_holder_simp_pi_qz=data_simp, 
                                        data_holder_simp_pi_lb=data_simp_lb, 
                                        data_holder_simp_pi_ub=data_simp_ub, 
                                        data_holder_orig_pi_pz=data_orig)
    
    analyzer.max_w()
    analyzer.policy_performances()
    
    analyzer.visualize_time_statistics(savefig=save)
    # analyzer.visualize_bounds_fill_between(savefig=save)
    # analyzer.visualize_bounds_binary_bars(savefig=save)
    
    analyzer.visualize_order_statistics(savefig=save, data=analyzer.simp_pi_qz, save_name='order_statistics_qz.pdf')
    analyzer.visualize_order_statistics(savefig=save, data=analyzer.simp_pi_lb, save_name='order_statistics_lb.pdf')
    
    # plt.show()