from dataclasses import dataclass

@dataclass
class PFTDPWConfig:
    num_query: int = 500
    ucb_exploration: float = 50.0
    # k and alpha are chosen such that for all nodes the number of children is between a minimum and a maximum
    # The minimum number determines k: k = n_{min}
    # The maximum number satisfies a = (log(n_{max}) - log(n_{min}))/log(n_q), where n_q is the number of queries
    k_observation: float = 1.1  # Minimum number of 2 observation branches
    alpha_observation: float = 0.19  # Maximum number of 5 observation branches at ~900 queries
    # k_action: float = 3.0  # Minimum number of 3 action branches
    # alpha_action: float = 0.12  # Maximum number of 8 action branches
    k_action: float = 1.1  # Minimum number of 2 action branches
    alpha_action: float = 0.24  # Maximum number of 4 action branches at 66 queries
    num_par: int = 250
    horizon: int = 15
    discount: int = 1.0
    pf_resample: bool = False  # Either True or false