from dataclasses import dataclass

from envs.abstract_environment import AbstractEnvironment
from envs.beacons import in_axis_aligned_rectangle
from models.probability_utils import MixtureDistribution
from inference.particle_filter import ParticleBelief
import numpy as np

from pykdtree.kdtree import KDTree

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

import os
import pickle
from tqdm import tqdm


@dataclass
class MeasurementSimplificationConfig:
    num_delta_points: int = 2 ** 11
    num_observations_per_delta: int = 2 ** 8
    k_sigma: float = 4.0  # Distance at which to take nearest neighbors to query in KDTree
    tree_search_eps: float = 0.1  # More efficient search?
    clean_threshold: float = 1e-4  # States with delta less than this value will be removed for efficiency
    use_bb_optimization: bool = True  # Whether to test for inclusion in bounding box containing delta states
    use_kdtree_in_m_i: bool = True # Whether to use kdtree for m_i estimation
    use_file: bool = True  # Whether to try to read from file instead of computing
    monte_carlo_state_samples = True  # Whether to do Monte Carlo estimation with particles from belief
    num_state_samples = 30  # Number of particles sampled from belief
    sample_seed: int = 0
    dir_path = "data/mes_simp_light_petal_dark_simple"
    file_name: str = "delta_and_states"
    file_extension: str = ".pkl"


def sum_delta_over_states(states, action, delta_states, delta, transition_model, vmax, delta_normalizer):
    """ Currently unused """
    n_D = len(delta)  #  should be equal to len(delta_states)
    n_S = len(states)
    
    repeated_states = np.repeat(states, n_D, axis=0)
    repeated_delta_states = np.tile(delta_states, reps=(n_S, 1))
    states_and_action = np.stack([repeated_states, np.repeat(action, n_S * n_D, axis=0)], axis=1)
    
    probabilities = transition_model.query_density(x=repeated_delta_states, theta=states_and_action)
    delta_estimates = np.matmul(probabilities.reshape(n_S, n_D), delta)
    return vmax * delta_normalizer * delta_estimates 


class MeasurementSimplification:
    def __init__(self, env: AbstractEnvironment, cfg: MeasurementSimplificationConfig, verbose=True) -> None:
        self.env = env
        self.cfg = cfg
        self.verbose = verbose

        self.transition_model = self.env.get_transition_model()
        self.obs_model = self.env.get_obs_model()
        self.simp_obs_model = self.env.get_simp_obs_model()

        self.proposal_distribution = MixtureDistribution(ds=[self.obs_model, self.simp_obs_model],
                                                         weights=np.ones(shape=2))

        # Could be a kd-tree or some spatially optimized data structure.
        self.N_delta = self.cfg.num_delta_points
        self.Q0 = None  # We assume uniform Q0
        self.delta_normalizer = None  # Factor that multiplies all of delta calculations
        self.N_delta_kept = None  # Number of delta points kept after cleaning
        self.delta_states = None
        self.kdtree = None  # KDTree holding all delta states
        self.delta_states_bounding_box = None  # 2x2 array where [0, :] is the bottom-left corner, [1, :] is the top-right, of bounding box of all delta states
        self.delta = None  # A mapping from delta state to the discrepancy function
        self.neighbor_distance = self.cfg.k_sigma * self.env.cfg.transition_sigma
        
    def deltas_file_name(self):
        return f"{self.cfg.file_name}_{self.cfg.num_delta_points}_{self.cfg.num_observations_per_delta}_{round(1/self.Q0)}_{self.cfg.file_extension}"
        
    def read_deltas_from_file(self):
        file_name = self.deltas_file_name()
        full_path = os.path.join(self.cfg.dir_path, file_name)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as file:
                delta_states, delta, Q0 = pickle.load(file)
                return delta_states, delta, Q0
        else:
            return None
            
    def save_deltas_to_file(self):
        if not os.path.isdir(self.cfg.dir_path):
            os.makedirs(self.cfg.dir_path, exist_ok=True)
        file_name = self.deltas_file_name()
        full_path = os.path.join(self.cfg.dir_path, file_name)
        with open(full_path, 'wb') as file:
                pickle.dump((self.delta_states, self.delta, self.Q0), file)

    def estimate_delta(self):
        ret = None
        if self.cfg.use_file:
            ret = self.read_deltas_from_file()
            if ret is not None:
                self.delta_states = ret[0]
                self.delta = ret[1]
                # self.Q0 = ret[2]  # For some reason in some of the files we saved 1/self.Q0 accidentally
        if ret is None:
            np.random.seed(self.cfg.sample_seed)
            self.delta_states = self.env.sample_states(self.N_delta)        
            # We don't want to keep too many observations in memory simultaneously(num_delta_points*num_observations_per_delta), 
            # so we sample observations for each state individually.
            self.delta = np.zeros(shape=self.N_delta)
            for i in tqdm(range(self.N_delta)):
                delta_state = self.delta_states[i, :][np.newaxis, :]
                obs = np.squeeze(self.proposal_distribution.sample(
                    delta_state, num=self.cfg.num_observations_per_delta))
                p_densities = self.obs_model.query_density(obs, theta=delta_state)
                q_densities = self.simp_obs_model.query_density(obs, theta=delta_state)
                # No need to calculate this as this is directly (p+q)/2
                # proposal_densities = self.proposal_distribution.query_density(obs, theta=delta_state)

                # The following is the importance sampling estimator for delta[i] when samples are from proposal of (p+q)/2
                self.delta[i] = 2 * np.sum(np.abs(p_densities - q_densities) / (p_densities + q_densities)) / \
                    self.cfg.num_observations_per_delta
            self.save_deltas_to_file()
                
    def setup_mes_simp(self):
        self.Q0 = 1 / (self.env.env_size[0] * self.env.env_size[1])  # We assume uniform Q0
        self.estimate_delta()
        self.delta_normalizer = 1 / (self.N_delta * self.Q0)  # Factor that multiplies all of delta calculations
        
        idx_to_clean = self.delta < self.cfg.clean_threshold
        self.delta_states = self.delta_states[~idx_to_clean]
        self.delta = self.delta[~idx_to_clean]
        self.N_delta_kept = len(self.delta)
        self.delta_states_bounding_box = np.zeros((2, 2))
        self.delta_states_bounding_box[1, :] = np.amax(self.delta_states, axis=0)
        self.delta_states_bounding_box[0, :] = np.amin(self.delta_states, axis=0)
        self.delta_states_bounding_box += np.array([[-self.neighbor_distance, -self.neighbor_distance],
                                               [self.neighbor_distance, self.neighbor_distance]])
        self.kdtree = KDTree(self.delta_states)

    def _estimate_m_i_states(self, time_index, state_samples, action, delta_states_idx=None):
        # delta_states is of shape n_Dx2
        # state_samples is of shape Nx2
        # Action is of shape 1x2
        # Returns the estimate of m_i in shape (N,) for each state_sample given the 
        if delta_states_idx is None:
            delta_states = self.delta_states
            delta = self.delta
            n_D = self.N_delta_kept
        else:
            delta_states = self.delta_states[delta_states_idx]
            delta = self.delta[delta_states_idx]
            n_D = len(delta)
        
        if n_D == 0:
            return np.zeros(len(state_samples))
        
        if len(state_samples.shape) == 1:
            state_samples = state_samples.reshape(1, -1)
        action = action.reshape(1, -1)
        
        repeated_states = np.repeat(state_samples, n_D, axis=0)
        repeated_delta_states = np.tile(delta_states, reps=(len(state_samples), 1))
        states_and_action = np.stack([repeated_states, np.repeat(action, len(repeated_states), axis=0)], axis=1)
        
        probabilities = self.transition_model.query_density(x=repeated_delta_states, theta=states_and_action)
        delta_estimates = np.matmul(probabilities.reshape(len(state_samples), n_D), delta) * self.delta_normalizer
        m_i = self.env.max_value(time_index) * delta_estimates
        return m_i
    
    def _estimate_m_i_states_bbox_kdtree(self, time_index, state_samples, action):
        transitioned = state_samples + action.reshape(1, -1)
        states_in_bb_idx = in_axis_aligned_rectangle(transitioned, self.delta_states_bounding_box)
        m_i = np.zeros(len(state_samples))
        transitioned_in_bb = transitioned[states_in_bb_idx]
        if transitioned_in_bb.size > 0:
            if self.cfg.use_kdtree_in_m_i:
                _, neighbors_transitioned_idx = self.kdtree.query(transitioned_in_bb, self.N_delta_kept, self.cfg.tree_search_eps, self.neighbor_distance)
                states_to_update_idx = np.copy(states_in_bb_idx)
                states_near_kdtree_idx = neighbors_transitioned_idx[:, 0] < self.N_delta_kept
                states_to_update_idx[states_in_bb_idx] = states_near_kdtree_idx
                states_near_kdtree = state_samples[states_to_update_idx]
                if states_near_kdtree.size > 0:
                    relevant_delta_states_ids = np.unique(neighbors_transitioned_idx)
                    relevant_delta_states_ids = relevant_delta_states_ids[relevant_delta_states_ids < self.N_delta_kept]
                    relevant_delta_states_idx = np.full(self.N_delta_kept, False)
                    relevant_delta_states_idx[relevant_delta_states_ids] = True
                    m_i[states_to_update_idx] = self._estimate_m_i_states(time_index, states_near_kdtree, action, relevant_delta_states_idx)
            else:
                m_i[states_in_bb_idx] = self._estimate_m_i_states(time_index, state_samples[states_in_bb_idx], action, None)
        return m_i
    
    def _estimate_m_i_states_kdtree(self, time_index, state_samples, action):
        transitioned = state_samples + action.reshape(1, -1)
        m_i = np.zeros(len(state_samples))
        _, neighbors_transitioned_idx = self.kdtree.query(transitioned, self.N_delta_kept, self.cfg.tree_search_eps, self.neighbor_distance)
        states_near_kdtree_idx = neighbors_transitioned_idx[:, 0] < self.N_delta_kept
        states_near_kdtree = state_samples[states_near_kdtree_idx]
        if states_near_kdtree.size > 0:
            relevant_delta_states_ids = np.unique(neighbors_transitioned_idx)
            relevant_delta_states_ids = relevant_delta_states_ids[relevant_delta_states_ids < self.N_delta_kept]
            relevant_delta_states_idx = np.full(self.N_delta_kept, False)
            relevant_delta_states_idx[relevant_delta_states_ids] = True
            m_i[states_near_kdtree_idx] = self._estimate_m_i_states(time_index, states_near_kdtree, action, relevant_delta_states_idx)
        return m_i
    
    def estimate_m_i_states(self, time_index, state_samples, action):
        if self.cfg.use_bb_optimization:
            return self._estimate_m_i_states_bbox_kdtree(time_index, state_samples, action)
        elif self.cfg.use_kdtree_in_m_i:
            return self._estimate_m_i_states_kdtree(time_index, state_samples, action)
        else:
            return self._estimate_m_i_states(time_index, state_samples, action)
        
    def estimate_m_i_belief(self, b: ParticleBelief, action):
        # We need to multiply by (1/N_delta) and multiply the likelihood by 1/Q0(x) for each state sample. 
        # However, when taking a uniform distribution, 1/Q0(x)=N_delta and both terms cancel out.
        if self.cfg.monte_carlo_state_samples:
            estimation_states = b.random_states(self.cfg.num_state_samples)
            weights = np.ones(self.cfg.num_state_samples) * (np.sum(b.weights) / self.cfg.num_state_samples)
            m_i_states = self.estimate_m_i_states(b.time, estimation_states, action)
            return np.dot(weights, m_i_states)
        else:
            m_i_states = self.estimate_m_i_states(b.time, b.states, action)
            return np.dot(b.weights, m_i_states)


class MeasurementSimplificationPlotting:
    def __init__(self, mes_simp: MeasurementSimplification) -> None:
        self.mes_simp = mes_simp
        self.cmap = plt.get_cmap("Reds")

    def delta_color(self, deltas, method=5.0):
        if method == 'sigmoid':
            deltas_norm = deltas - np.mean(deltas)
            deltas_norm = deltas_norm / np.std(deltas_norm)
        elif method == 'absolute':
            deltas_norm = 0.5 * deltas
        elif type(method) == float:
            deltas_norm = method * deltas
        return self.cmap(deltas_norm)

    def plot_delta_states(self, ax: Axes):
        ax.scatter(x=self.mes_simp.delta_states[:, 0], y=self.mes_simp.delta_states[:, 1], c=self.delta_color(
            self.mes_simp.delta), marker="1")
