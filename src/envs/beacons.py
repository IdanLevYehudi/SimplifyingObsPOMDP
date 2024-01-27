import numpy as np
from dataclasses import dataclass, field

from scipy.stats import qmc
from copy import copy

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import to_rgb, to_rgba

from configs.envs.beacons_params import BeaconsLightDarkConfig
from envs.abstract_environment import AbstractEnvironment
from models.probability_utils import AbstractDistribution, MultivariateNormalDistribution, GMMDistribution
from inference.particle_filter import ParticleBelief
from utils.visualization import confidence_ellipse
from utils.quasirandom import rd_sequence


def in_axis_aligned_rectangle(xs, rect):
    # rect is assumed to be 2x2, where beacons_area[0] is the bottom-left point
    # and rect[1] is the top-right point.
    xs = fix1d(xs)
    return np.logical_and(np.all(xs >= rect[0], axis=1),
                          np.all(xs <= rect[1], axis=1))


def rectangle_dims(rect):
    # Returns width, height of an axis-aligned rectangle.
    width = rect[1, 0] - rect[0, 0]
    height = rect[1, 1] - rect[0, 1]
    return width, height


def fix1d(arr):
    if len(arr.shape) == 1:
        return arr[np.newaxis, ...]
    return arr


def rand_point_circle(radius, center=np.array([0, 0]), num_samples=1):
    theta = np.random.uniform(size=num_samples) * 2 * np.pi
    return center + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def rand_point_disc(radius, center=np.array([0, 0]), num_samples=1):
    r = radius * np.sqrt(np.random.uniform(size=num_samples))
    theta = np.random.uniform(size=num_samples) * 2 * np.pi
    return center + r * np.stack(np.cos(theta), np.sin(theta), axis=1)


# https://stackoverflow.com/a/45313353/ @Divakar
def view1D(a, b): # a, b are arrays
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    return a.view(void_dt).ravel(),  b.view(void_dt).ravel()

def setdiff_nd(a,b):
    # a,b are the nD input arrays
    A,B = view1D(a,b)    
    return a[~np.isin(A,B)]


class RobotPriorDistribution(AbstractDistribution):
    def __init__(self, start_position_line, num_start_position_modes, start_position_cov) -> None:
        self.means = np.linspace(start=start_position_line[0],
                                 stop=start_position_line[1],
                                 num=num_start_position_modes)
        weights = np.ones(shape=(len(self.means)))
        self.gmm = GMMDistribution(
            comp_params=[(self.means[i], start_position_cov)
                         for i in range(len(weights))],
            weights=weights)

    def dim(self):
        return self.gmm.dim()

    def sample(self, theta=None, num=1):
        """
        Generate samples from model
        """
        return np.squeeze(self.gmm.sample(num=num))

    def query_density(self, x, theta=None):
        """
        Query the density at the x.
        """
        return self.gmm.query_density(fix1d(x))
    
    def query_log2_density(self, x, theta=None):
        return self.gmm.query_log2_density(fix1d(x))


class RobotTransitionModel(AbstractDistribution):
    def __init__(self, cov) -> None:
        self.g = MultivariateNormalDistribution(mean=np.array([0, 0]), cov=cov)

    def dim(self):
        return self.g.dim()

    def sample(self, theta, num=1):
        """
        Generate samples from model
        Args:
            theta: Numpy array of Nx2x2: iterable of (robot_state, action).
            num (int, optional): number of samples per state-action pair. Defaults to 1.

        Returns:
            ndarray: Nxkx2 samples, each kx2 samples conditioned on n'th (robot_state, action).
        """
        means = theta[:, 0] + theta[:, 1]  # Shape is Nx2
        samples = self.g.sample(theta=np.zeros((len(theta), 1)),num=num)  # Shape is Nxkx2
        # We use numpy broadcasting to add k samples for each mean
        if num > 1:
            return means[:, np.newaxis, :] + samples
        else:
            return means + samples

    def query_density(self, x, theta):
        """
        Query the density at robot states.
        Args:
            x: Robot states to query. Shape Nx2.
            theta: Numpy array of Nx2x2: iterable of (robot_state, action).

        Returns:
            ndarray: p(x | theta), of length N.
        """
        transitioned = theta[:, 0, :] + theta[:, 1, :]  # Shape is Nx2
        query_points = x - transitioned
        return self.g.query_density(query_points)
    
    def query_log2_density(self, x, theta):
        transitioned = theta[:, 0, :] + theta[:, 1, :]  # Shape is Nx2
        query_points = x - transitioned
        return self.g.query_log2_density(query_points)


class BeaconObservationModel(AbstractDistribution):
    def __init__(self, beacons, cov_close, cov_far, dist_threshold, k_r_close, k_theta_close, n_sigma_close, k_r_far, k_theta_far, n_sigma_far) -> None:
        # Shape Kx2, each row represents (x,y) of a beacon.
        self.beacons = beacons
        # Distance threshold for beacon measurement.
        self.dist_threshold = dist_threshold
        self.cov_close = cov_close
        self.cov_far = cov_far

        self.k_r_close = k_r_close
        self.k_theta_close = k_theta_close
        self.n_sigma_close = n_sigma_close
        self.k_r_far = k_r_far
        self.k_theta_far = k_theta_far
        self.n_sigma_far = n_sigma_far
        
        self.g_close = None
        self.g_far = None
        self.init_g_close()
        # self.init_g_far()
        self.g_far = MultivariateNormalDistribution(mean=np.array([0, 0]), cov=cov_far)
        
    def init_g_close(self):
        r_weight_scale = np.array([np.exp(-((i * self.n_sigma_close/self.k_r_close) ** 2)/2) for i in range(1, self.k_r_close + 1)])  # Exp(-x^2/2) weight decay
        self.g_close = GMMDistribution.petal_gmm_model(self.cov_close, 
                                                       self.k_r_close, 
                                                       self.k_theta_close,
                                                       n_sigma_to_cover=self.n_sigma_close,
                                                       std_scale_by_radius=np.ones(self.k_r_close), 
                                                       weight_scale_by_radius=r_weight_scale, 
                                                       weight_split_among_theta=False)
        
    def init_g_far(self):
        r_weight_scale = np.array([np.exp(-((i * self.n_sigma_far/self.k_r_far) ** 2)/2) for i in range(1, self.k_r_far + 1)])  # Exp(-x^2/2) weight decay
        self.g_far = GMMDistribution.petal_gmm_model(self.cov_far,
                                                       self.k_r_far, 
                                                       self.k_theta_far,
                                                       n_sigma_to_cover=self.n_sigma_far,
                                                       std_scale_by_radius=np.ones(self.k_r_far), 
                                                       weight_scale_by_radius=r_weight_scale, 
                                                       weight_split_among_theta=False)

    def _close_to_beacons(self, robot_states):
        # robot_states is Nx2
        # self.beacons is Kx2
        diff = self.beacons[np.newaxis, :, :] - robot_states[:, np.newaxis, :]
        distances = np.linalg.norm(diff, axis=2)  # Shape is now NxK
        below_dist_from_beacon = distances <= self.dist_threshold
        return np.any(below_dist_from_beacon, axis=1)

    def dim(self):
        return self.g_close.dim()

    def sample(self, theta, num=1):
        """
        Generate samples from model
        Args:
            theta: Numpy array of Nx2: iterable of (robot_state).
            num (int, optional): number of samples per state. Defaults to 1.

        Returns:
            ndarray: Shape Nxkx2.
        """
        theta = fix1d(theta)

        in_range_from_beacon = self._close_to_beacons(theta).astype(int)
        num_in_range = np.count_nonzero(in_range_from_beacon)
        num_out_range = len(in_range_from_beacon) - num_in_range

        # There might be a better way to write this with numpy broadcasting,
        # instead of splitting to cases.
        if len(in_range_from_beacon) == 1:
            if in_range_from_beacon[0]:
                samples = self.g_close.sample(theta=np.zeros((num_in_range, 1)), num=num)
            else:
                samples = self.g_far.sample(theta=np.zeros((num_out_range, 1)), num=num)
        else:
            samples = np.zeros(shape=(len(theta), num, 2))
            if num_in_range > 0:
                samples[in_range_from_beacon, :, :] = self.g_close.sample(
                    theta=np.zeros((num_in_range, 1)), num=num)
            if num_out_range > 0:
                samples[~in_range_from_beacon, :, :] = self.g_far.sample(
                    theta=np.zeros((num_out_range, 1)), num=num)
        if num > 1:
            return samples + theta[:, np.newaxis, ...]
        else:
            return samples + theta

    def query_density(self, x, theta, log2=False):
        """
        Query the density at robot states.
        Args:
            x: Observation to query. Shape Nx2.
            theta: Numpy array of Nx2: each row is a robot state. Can be 1x2 (assumed constant for all x).

        Returns:
            ndarray: Shape (N,).
        """
        theta = fix1d(theta)
        x = fix1d(x)
        shifted_x = x - theta

        in_range_from_beacon = self._close_to_beacons(theta).astype(bool)
        # There might be a better way to write this with numpy broadcasting,
        # instead of splitting to cases.
        if len(in_range_from_beacon) == 1:
            if in_range_from_beacon[0]:
                heights = self.g_close.query_log2_density(shifted_x) if log2 else self.g_close.query_density(shifted_x)
            else:
                heights = self.g_far.query_log2_density(shifted_x) if log2 else self.g_far.query_density(shifted_x)
        else:
            num_in_range = np.count_nonzero(in_range_from_beacon)
            num_out_range = len(in_range_from_beacon) - num_in_range
            heights = np.zeros(shape=len(shifted_x))
            if num_in_range > 0:
                heights[in_range_from_beacon] = self.g_close.query_log2_density(shifted_x[in_range_from_beacon]) if log2 else self.g_close.query_density(shifted_x[in_range_from_beacon])
            if num_out_range > 0:
                heights[~in_range_from_beacon] = self.g_far.query_log2_density(shifted_x[~in_range_from_beacon]) if log2 else self.g_far.query_density(shifted_x[~in_range_from_beacon])
        return heights
    
    def query_log2_density(self, x, theta):
        return self.query_density(x=x, theta=theta, log2=True)


class BeaconSimplifiedObservationModel(BeaconObservationModel):
    def __init__(self, beacons, cov_close, cov_far, dist_threshold) -> None:
        super().__init__(beacons, cov_close, cov_far, dist_threshold, 1, 1, 1, 1, 1, 1)
        # The simplified distributions are single component gaussians
        self.g_close = MultivariateNormalDistribution(mean=np.array([0, 0]), cov=cov_close)
        self.g_far = MultivariateNormalDistribution(mean=np.array([0, 0]), cov=cov_far)

class BeaconsLightDark(AbstractEnvironment):
    def __init__(self, cfg: BeaconsLightDarkConfig):
        self.cfg = cfg

        self.x = None  # ndarray of length 2: (x, y).
        self.curr_time = 0  # Current time index of the simulation
        self.beacons = None  # Nx2 array of (x, y) of beacons.
        self.goal_entry_point = None  # 1D array of (x, y)
        self.goal_size = None  # 1D array of (width, height)
        self.env_bounding_rect = None  # 2D array where [0, :] is the bottom-left corner, and [1, :] is the top-right corner
        self.env_center = None  # 1D array of (x, y)
        self.env_size = None  # 1D array of (width, height)

        self.time_horizon = self.cfg.time_horizon
        # We offset the time penalty reward such that it's centered around 0.
        self.time_penalty_offset = (
            self.time_horizon + 1) * self.cfg.action_penalty / 2

        self.robot_prior = RobotPriorDistribution(self.cfg.start_position_line,
                                                  self.cfg.num_start_position_modes,
                                                  self.cfg.start_position_cov)
        self.rmax = None  # Float of maximal absolute value of instantaneous reward
        self.vmax = None  # 1D array of maximal value starting at each time step
        
        self._init_rmax()
        self._init_vmax()

        self.actions = None  # In case of discrete actions, 2D array of Kx2 where K is the number of actions
        
        self.init_state()
        self.init_beacons()
        self.init_goal_region()
        self.init_outer_walls()
        self.init_actions()
        
        self.transition_model = RobotTransitionModel(
            self.cfg.transition_model_cov)
        self.obs_model = BeaconObservationModel(self.beacons,
                                                self.cfg.obs_close_cov,
                                                self.cfg.obs_far_cov,
                                                self.cfg.meas_distance_threshold,
                                                self.cfg.obs_close_num_radii,
                                                self.cfg.obs_close_num_components_circumference,
                                                self.cfg.obs_close_num_sigma,
                                                self.cfg.obs_far_num_radii,
                                                self.cfg.obs_far_num_components_circumference,
                                                self.cfg.obs_far_num_sigma)
        self.simp_obs_model = BeaconSimplifiedObservationModel(self.beacons,
                                                               self.cfg.obs_close_cov,
                                                               self.cfg.obs_far_cov,
                                                               self.cfg.meas_distance_threshold)

    def init_state(self):
        self.x = np.squeeze(self.robot_prior.sample())
        self.curr_time = 0

    def init_beacons(self):
        # Numpy mgrid creates a meshgrid of linspace.
        # The complex number is used to specify the number of points, instead
        # of the step size.
        beacons_area = self.cfg.beacons_area

        num_x = self.cfg.num_beacons[0] * 1j
        num_y = self.cfg.num_beacons[1] * 1j
        beacons_grid = np.mgrid[beacons_area[0, 0]:beacons_area[1, 0]:num_x,
                                beacons_area[0, 1]:beacons_area[1, 1]:num_y]

        # Convert meshgrid to list of coordinates - output would be Kx2 (for K beacons).
        self.beacons = np.vstack(list(map(np.ravel, beacons_grid))).T

    def init_goal_region(self):
        goal_center = np.mean(self.cfg.goal_region, axis=0)
        # self.goal_entry_point = np.amin([np.amax([goal_center, self.cfg.outer_walls[0]], axis=0), self.cfg.outer_walls[1]], axis=0)
        self.goal_entry_point = goal_center
        self.goal_size = np.array(rectangle_dims(self.cfg.goal_region))

    def init_outer_walls(self):
        self.walls_center = np.mean(self.cfg.outer_walls, axis=0)
        self.walls_size = np.array(rectangle_dims(self.cfg.outer_walls))
        
        self.env_bounding_rect = np.array([np.amin(np.array([self.cfg.outer_walls[0], self.cfg.goal_region[0]]), axis=0),
                                           np.amax(np.array([self.cfg.outer_walls[1], self.cfg.goal_region[1]]), axis=0)])
        self.env_bounding_rect += np.array([[-self.cfg.action_length, -self.cfg.action_length],
                                            [self.cfg.action_length, self.cfg.action_length]])
        self.env_center = np.mean(self.env_bounding_rect, axis=0)
        self.env_size = np.array(rectangle_dims(self.env_bounding_rect))
        
    def init_actions(self):
        if not self.cfg.discrete_actions:
            return
        angles = np.linspace(start=0, stop=2 * np.pi, num=self.cfg.num_discrete_actions, endpoint=False)
        self.actions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def in_collision(self, states):
        return np.logical_and(np.logical_not(in_axis_aligned_rectangle(states, self.cfg.outer_walls)),
                              np.logical_not(self._in_goal_region(states)))

    def get_prior_model(self):
        return self.robot_prior

    def get_transition_model(self):
        return self.transition_model

    def get_obs_model(self):
        return self.obs_model

    def get_simp_obs_model(self):
        return self.simp_obs_model

    def get_curr_time(self):
        return self.curr_time

    def get_observations(self, num_obs=1):
        """
        Generates observations from the current environment state.
        Args:
            num_obs (int, optional): Number of observations to generate. Defaults to 1.
        """
        return np.squeeze(self.obs_model.sample(self.x[np.newaxis, ...], num=num_obs))

    def _in_goal_region(self, states):
        return in_axis_aligned_rectangle(states, self.cfg.goal_region)

    def is_terminal(self, states, time_index):
        """
        Check for a vector of states at a certain time if this is a termination of the pomdp.
        Termination is when time limit exceeds, entering goal region or colliding with outer wall.

        Args:
            states (_type_): Array of states Nx2.
            time_index: integer.
        """
        return time_index >= self.time_horizon or \
            states is None or \
            states.size == 0 or \
            np.all(self._in_goal_region(states)) or \
            np.all(self.in_collision(states))

    def action_in_direction(self, state, diff):
        if self.cfg.discrete_actions:
            invalid_actions = self.in_collision(self.actions + state.reshape(1, 2))
            valid_actions = self.actions[~invalid_actions]
            return valid_actions[np.argmax(np.dot(valid_actions, np.squeeze(diff)))]
        else:
            return self.cfg.action_length * diff / np.linalg.norm(diff)

    def terminated(self):
        return np.squeeze(self.is_terminal(self.x, self.curr_time))

    def transition(self, b: ParticleBelief, action, sample=True):
        if b.is_empty():
            return b, 0.0, True, True
        
        next_b = copy(b)
        start_weights = next_b.weights
        next_b.increment_time()
        rewards = np.zeros(shape=len(start_weights))
        terminated_particles = np.full(len(start_weights), False)

        state_action_pair = np.stack([next_b.states, np.repeat(
            fix1d(action), len(next_b.states), axis=0)], axis=1)
        if sample:
            next_states = np.squeeze(
                self.transition_model.sample(state_action_pair, num=1))
        else:
            next_states = np.squeeze(state_action_pair[:, 0] + state_action_pair[:, 1])
        next_b.states = fix1d(next_states)
        # Inspired by floor environment in VTS, but with finite time limit.
        # POMDP is terminated when time limit is reached, then hit or miss penalty
        # is calculated.
        # If not terminated, particles entering the goal zone will be rewarded hit
        # reward, and only those particles will terminate.
        # All particles not having hit/miss will be awarded action penalty.
        # Additionally, collision penalty will be added for each particle.
        if self.is_terminal(next_states, next_b.time):
            rewards = np.where(self._in_goal_region(next_states),
                               self.cfg.hit_reward, self.cfg.miss_penalty)
            rewards += np.where(self.in_collision(next_states),
                                self.cfg.collision_penalty, 0)
            # Terminate any remaining particles
            terminated_particles[:] = True
        else:
            particles_in_goal = self._in_goal_region(next_states)
            particles_in_collision = self.in_collision(next_states)
            terminated_particles = np.logical_or(particles_in_goal, particles_in_collision)
            # Add rewards for goal
            rewards += particles_in_goal * self.cfg.hit_reward
            # Add penalty for collisions
            rewards += particles_in_collision * self.cfg.collision_penalty
            # Add transition penalty for non terminated particles
            rewards += ~terminated_particles * self.cfg.action_penalty
        # Take terminated particles out of belief
        next_b.freeze_particles(terminated_particles)
        return next_b, np.dot(rewards, start_weights), np.all(terminated_particles)

    def step(self, action):
        """
        Transition the environment by action.
        """
        weight = np.array([1])
        fake_b = ParticleBelief(states=self.x.reshape(1, 2), weights=weight, time=self.curr_time)
        next_b, reward, is_terminal = self.transition(fake_b, action)
        self.curr_time = next_b.time
        if next_b.states.size == 0:
            self.x = self.x + np.squeeze(action)
        else:
            self.x = np.squeeze(next_b.states)
        return self.get_observations(), reward, is_terminal

    def sample_action(self, b: ParticleBelief, expanded_actions):
        # Gives back a uniformly sampled random action
        if self.cfg.discrete_actions:
            # If all actions have been expanded, return None to signal there are no new actions
            if len(expanded_actions) == self.cfg.num_discrete_actions:
                return None
            # Return a random action from the remaining actions
            remaining_actions = setdiff_nd(self.actions, expanded_actions)
            return remaining_actions[np.random.randint(len(remaining_actions))]
        else:
            return rand_point_circle(self.cfg.action_length)

    def sample_states(self, num_states, quasirandom=True):
        """
        Uniformly sample states in the environment.
        """
        if quasirandom:
            xy_samples = rd_sequence(
                dim=2, start_n=num_states, scale=self.env_size, shift=self.env_center - self.env_size/2)
        else:
            x_samples = np.random.uniform(
                low=self.env_bounding_rect[0, 0], high=self.env_bounding_rect[1, 0], size=num_states)
            y_samples = np.random.uniform(
                low=self.env_bounding_rect[0, 1], high=self.env_bounding_rect[1, 1], size=num_states)
            xy_samples = np.stack([x_samples, y_samples], axis=1)
        return xy_samples

    def _rollout_policy(self, state, time_index):
        """
        Returns a heuristic policy based on a given state.

        Args:
            state (ndarray): ndarray of (time, x, y).
        """
        return self.action_in_direction(state, self.goal_entry_point - state)

    def _rollout(self, start_state, start_belief: ParticleBelief, secondary_reward_func, depth=np.Inf, sample=True):
        """
        Rollout from a heuristic defined on state s, check the average return when applied to 
        states with given weights.
        """
        if start_state is None or start_state.size == 0 or start_belief.is_empty():
            return 0.0, 0.0
        
        next_b = copy(start_belief)
        
        rollout_return = 0
        rollout_secondary_return = 0

        iter_num = 0
        while not self.is_terminal(next_b.states, next_b.time) and not next_b.is_empty() and iter_num < depth:
            rollout_action = self._rollout_policy(next_b.emp_mean(), next_b.time)
            if secondary_reward_func is not None:
                curr_secondary_reward = secondary_reward_func(next_b, rollout_action)
                rollout_secondary_return += curr_secondary_reward
            next_b, curr_reward, is_terminal = self.transition(next_b, action=rollout_action, sample=sample)
            rollout_return += curr_reward
            
            if is_terminal:
                break
            else:
                iter_num += 1
        return rollout_return, rollout_secondary_return

    def rollout(self, start_state, start_belief: ParticleBelief, secondary_reward_func=None, depth=np.Inf):
        return self._rollout(start_state, start_belief, secondary_reward_func, depth, sample=not self.cfg.simple_rollout)
    
    def _max_reward(self, time_index):
        # Computes an upper bound for the future reward at a specific time step.
        return max(abs(self.cfg.hit_reward), abs(self.cfg.miss_penalty) + abs(self.cfg.collision_penalty), abs(self.cfg.action_penalty))
    
    def _init_rmax(self):
        self.rmax = self._max_reward(0)
    
    def _max_value(self, start_time):
        # Computes an upper bound for the portion of maximum attainable value from a future time step w.r.t to the maximum value from the current time step.
        return self.rmax + max(self.time_horizon - start_time - 1, 0) * abs(self.cfg.action_penalty)
    
    def _init_vmax(self):
        self.vmax = np.array([self._max_value(i) for i in range(0, self.time_horizon)])
    
    def max_reward(self, time_index):
        return self.rmax
    
    def max_value(self, start_time):
        return self.vmax[start_time]

def norm_to_range(arr, a=0, b=1):
    if arr is None or arr.size <= 1:
        return np.array([1])
    min_val = np.amin(arr)
    max_val = np.amax(arr)
    if min_val == max_val:
        return a + (b - a) * np.ones(len(arr)) / len(arr)
    return a + (b - a) * (arr - min_val)/(max_val - min_val)


def alphas_to_colors(alphas, base_color):
    r, g, b = to_rgb(c=base_color)
    return [(r, g, b, alpha) for alpha in alphas]


class BeaconsLightDarkPlotting:
    def __init__(self, env) -> None:
        self.env = env

    def plot_obs_model(self, ax: Axes, color='g'):
        ax.scatter(self.env.beacons[:, 0],
                   self.env.beacons[:, 1], c=color, marker='s')
        for i in range(len(self.env.beacons)):
            ax.add_patch(Circle((self.env.beacons[i, 0], self.env.beacons[i, 1]),
                         radius=self.env.cfg.meas_distance_threshold, color=color, alpha=0.35))

    def plot_simp_obs_model(self, ax: Axes, color='orange'):
        pass
        # simp_model = self.env.get_simp_obs_model()
        # bottom_left = simp_model.beacons[0]
        # width, height = rectangle_dims(simp_model.beacons)
        # ax.add_patch(Rectangle(bottom_left, width=width, height=height,
        #              fill=True, alpha=0.25, color=color, linewidth=2))

    def plot_robot_prior(self, ax: Axes, color='black'):
        for component in self.env.get_prior_model().gmm.comp_params:
            confidence_ellipse(component[0], component[1],
                               ax, facecolor=color, alpha=0.3)

    def plot_states(self, ax: Axes, states, color, weights=None):
        if weights is None:
            weights = np.ones(shape=len(states)) / len(states)
        rgbas = alphas_to_colors(norm_to_range(weights, a=0.05, b=1.0), color)
        ax.scatter(states[:, 0], states[:, 1], c=rgbas, s=10, marker='o')
        return rgbas

    def plot_robot_gt(self, ax: Axes, color='black'):
        self.plot_states(ax, self.env.x[np.newaxis, ...], color)

    def plot_outer_walls(self, ax: Axes, color='gray'):
        ax.add_patch(Rectangle(self.env.cfg.outer_walls[0],
                               width=self.env.walls_size[0],
                               height=self.env.walls_size[1],
                               fill=False, color=color, linewidth=5))

    def plot_goal_region(self, ax: Axes, color='blue'):
        ax.add_patch(Rectangle(self.env.cfg.goal_region[0],
                               width=self.env.goal_size[0],
                               height=self.env.goal_size[1],
                               fill=True, alpha=0.8, color=color, linewidth=2))

    def plot_env(self, ax: Axes):
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        self.plot_outer_walls(ax)
        self.plot_goal_region(ax)
        # self.plot_robot_prior(ax)
        self.plot_simp_obs_model(ax)
        self.plot_obs_model(ax)
        self.plot_robot_gt(ax)

    def env_size_ratio(self, base_width):
        return base_width, base_width * self.env.env_size[1] / self.env.env_size[0]

    def plot_observations(self, ax: Axes, obs, color='r'):
        obs = fix1d(obs)
        if len(obs) == 1:
            ax.scatter(obs[:, 0], obs[:, 1], c=color, s=10)
        else:
            heights = self.env.get_obs_model().query_density(obs, self.env.x)
            ax.scatter(obs[:, 0], obs[:, 1], c=alphas_to_colors(
                norm_to_range(heights, a=0.2, b=1), color), s=10)

    def plot_action(self, ax: Axes, states, action, weights=None, color='purple'):
        action = np.squeeze(action)
        self.plot_states(ax=ax, states=states, color=color, weights=weights)
        rgbas = alphas_to_colors(norm_to_range(weights, a=0.3, b=1), color)
        for i in range(len(states)):
            ax.arrow(x=states[i, 0], y=states[i, 1],
                     dx=action[0], dy=action[1],
                     color=rgbas[i], shape='full',
                     length_includes_head=True,
                     width=0.01,
                     head_width=0.1)

    def plot_states_covs(self, ax: Axes, states_and_covs, color='grey'):
        for state, cov in states_and_covs:
            confidence_ellipse(state, cov,
                               ax, facecolor=color, alpha=0.8)

    def plot_action_sequence(self, ax: Axes, state, actions, color='cyan'):
        state = state.reshape(-1, 2)
        # Assuming actions is Nx2 array
        final_positions = np.cumsum(np.vstack([state, actions]), axis=0)
        ax.quiver(final_positions[:-1, 0], final_positions[:-1, 1], actions[:, 0],
                  actions[:, 1], scale_units='xy', angles='xy', scale=1, color=color)
