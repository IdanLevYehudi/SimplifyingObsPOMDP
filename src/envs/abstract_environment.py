from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):    
    @abstractmethod
    def get_observations(self, num_obs=1):
        """
        Generates observations from the current environment state.
        Args:
            num_obs (int, optional): Number of observations to generate. Defaults to 1.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Transition the environment by action.
        Args:
            action (_type_): _description_
        """
        pass
    
    @abstractmethod
    def is_terminal(self, states):
        """
        Check for a vector of states if each is in a terminal state.

        Args:
            states (_type_): Array of states (indexed by dim 0).
        """
        pass

    @abstractmethod
    def sample_action(self):
        # Gives back a uniformly sampled random action
        pass
    
    @abstractmethod
    def in_collision(self, s):
        # Returns true if state is in collision
        pass
    
    @abstractmethod
    def sample_states(self):
        """
        Initializes the environment to a uniformly sampled valid state.
        """
        pass
    
    @abstractmethod
    def rollout(self, ss, ws):
        """
        Rollout from a heuristic defined on state s, check how it affects list of states ss with weights ws.

        Args:
            s (_type_): Starting state to calculate heuristic on.
            ss (_type_): Vector of states to rollout.
            ws (_type_): Weights of states.
        """
        pass