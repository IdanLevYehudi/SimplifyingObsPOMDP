from dataclasses import dataclass, field
import numpy as np


@dataclass
class BeaconsLightDarkConfig():
    # Time horizon - maximum number of actions to take
    time_horizon: int = 15

    # Length of a movement action
    action_length: float = 1.0
    discrete_actions: bool = True
    num_discrete_actions: int = 4

    # The prior distribution is a gaussian mixture with k equal modes.
    # The means of the gaussians are equispaces along the specified line.
    num_start_position_modes: int = 2
    start_position_line: np.ndarray = np.array([[1, 2],
                                                [9, 2]])

    prior_sigma_x: float = 0.5
    prior_sigma_y: float = 0.25
    start_position_cov: np.ndarray = field(init=False)

    transition_sigma_action_ratio: float = 0.15  # Arbitrary
    # the sigma is ratio * action_length
    transition_sigma: float = field(init=False)
    transition_model_cov: np.ndarray = field(init=False)

    # number in x axis, number in y axis
    # num_beacons: np.ndarray = np.array([6, 2])
    # defined from bottom-left to top-right
    # beacons_area: np.ndarray = np.array([[0, 2],
    #                                      [10, 4]])
    
    num_beacons: np.ndarray = np.array([6, 1])
    beacons_area: np.ndarray = np.array([[0, 4],
                                         [10, 4]])
    
    meas_distance_threshold: float = 1.0

    obs_close_sigma_x: float = 0.3
    obs_close_sigma_y: float = 0.3
    obs_close_cov: np.ndarray = field(init=False)
    obs_close_num_radii: int = 10
    obs_close_num_components_circumference: int = 25
    obs_close_num_sigma: float = 3.0
    
    obs_far_sigma_x: float = 5.0
    obs_far_sigma_y: float = 5.0
    obs_far_cov: np.ndarray = field(init=False)
    obs_far_num_radii: int = 10
    obs_far_num_components_circumference: int = 25
    obs_far_num_sigma: float = 6.0
    

    # outer walls - valid area for states
    # Wide walls:
    # outer_walls: np.ndarray = np.array([[-4, -3],
    #                                     [14, 6]])
    # Tight walls:
    # Goal region is a "door" on the outer wall
    outer_walls: np.ndarray = np.array([[-2, 0],
                                        [12, 6]])

    # defined from bottom-left to top-right
    goal_region: np.ndarray = np.array([[4, -1.5],
                                        [6, 0]])

    ### Rewards ###
    action_penalty: float = -1  # Slight penalty per action to encourage early finish
    hit_reward: float = 100  # If executing termination action in goal region
    miss_penalty: float = -50  # If executing termination action outside goal region
    collision_penalty: float = -50  # If transitioning to a state in collision
    
    # Rollout method - simplified (ML actions) or with sampling
    simple_rollout: bool = True
    # simple_rollout: bool = False
    
    def __post_init__(self):
        self.start_position_cov = np.array([[self.prior_sigma_x ** 2, 0.0],
                                            [0.0, self.prior_sigma_y ** 2]])

        self.transition_sigma = self.action_length * self.transition_sigma_action_ratio
        self.transition_model_cov = np.array([[self.transition_sigma ** 2, 0],
                                              [0, self.transition_sigma ** 2]])

        self.obs_close_cov = np.array([[self.obs_close_sigma_x ** 2, 0],
                                       [0, self.obs_close_sigma_y ** 2]])
        self.obs_far_cov = np.array([[self.obs_far_sigma_x ** 2, 0],
                                     [0, self.obs_far_sigma_y ** 2]])