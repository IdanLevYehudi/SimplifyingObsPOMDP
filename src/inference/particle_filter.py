from dataclasses import dataclass
from copy import copy

import numpy as np



def fix1d(arr):
    if len(arr.shape) == 1:
        return arr[np.newaxis, ...]
    return arr


def normalize_probabilities(probs):
    if np.sum(probs) == 0:
        return probs
    else:
        return probs / np.sum(probs)


def logaddexp2_arr(arr):
    if arr.size == 0:
        return np.NINF
    with np.errstate(divide='ignore'):
        return np.logaddexp2.reduce(arr)

            
def normalize_log_probs(log_probs, normalized_sum_log2=0):
    # normalized_sum_log2 is the log2 of the sum of the probabilities after normalization
    # Normally we want to normalize probabilities to 1 hence log2(0)=1, but sometimes we want to normalize in parts, so the target normalization value might be smaller.
    # We normalize p_i such that \Sum{p_i} = 2^(normalized_sum_log2)
    
    # Numerically stable routine for normalizing log probs
    # https://stackoverflow.com/questions/18069269/normalizing-a-list-of-very-small-double-numbers-likelihoods
    if np.all(np.isneginf(log_probs)):
        return log_probs
    log_probs -= np.amax(log_probs)
    log_sum_curr_weights = logaddexp2_arr(log_probs)
    return log_probs - log_sum_curr_weights + normalized_sum_log2


def target_normalized_sum(frozen_log_weights):
    # Normalizes log_probs such that \Sum{2^(log_probs_i)} = 1-\Sum{2^(frozen_log_weights_i)}
    with np.errstate(divide='ignore'):
        return np.log2(min(max(1 - np.exp2(logaddexp2_arr(frozen_log_weights)), 0), 1))


class ParticleBelief:
    def __init__(self, states=None, weights=None, log_weights=None, time=None, frozen_log_weights=None) -> None:
        # This is useful in simulations where dead particles should not have their weights zeroed in future time steps (if they terminated early)
        if frozen_log_weights is None:
            self.frozen_log_weights = np.array([])  # Array of the log weights of frozen particles
        else:
            self.frozen_log_weights = frozen_log_weights
        self.non_frozen_log_weight = target_normalized_sum(self.frozen_log_weights)  # Log2 of total weight of non-frozen particles
        self.states = states
        if log_weights is not None:
            self._log_weights = log_weights
        else:
            with np.errstate(divide='ignore'):
                self._log_weights = np.log2(weights)
        self._log_weights = normalize_log_probs(self._log_weights, self.non_frozen_log_weight)
        self._log_weights_unnorm = self._log_weights
        self._max_log_w = np.NINF
        self.time = time
    
    def add_frozen_log_weights(self, new_frozen_log_weights):
        self.frozen_log_weights = np.concatenate([self.frozen_log_weights, new_frozen_log_weights])
        self.non_frozen_log_weight = target_normalized_sum(self.frozen_log_weights)
    
    @property
    def log_weights(self):
        return self._log_weights
        
    @log_weights.setter
    def log_weights(self, new_log_weights):
        self._log_weights = normalize_log_probs(new_log_weights, self.non_frozen_log_weight)
    
    @property
    def weights(self):
        return np.exp2(self.log_weights)
    
    @weights.setter
    def weights(self, new_weights):
        with np.errstate(divide='ignore'):
            self.log_weights = np.log2(new_weights)

    def emp_mean(self):
        if self.is_empty():
            raise RuntimeWarning("Belief is empty")
        return np.average(fix1d(self.states), axis=0, weights=self.weights)

    def emp_cov(self):
        if self.is_empty():
            raise RuntimeWarning("Belief is empty")
        weights_safe = self.weights + 1e-8
        weights_safe = weights_safe / np.sum(weights_safe)
        return np.cov(fix1d(self.states), rowvar=False, aweights=weights_safe)

    def increment_time(self):
        self.time += 1
        
    def mult_weights(self, w):
        with np.errstate(divide='ignore'):
            self.log_weights = self.log_weights + np.log2(w)
            self._log_weights_unnorm = self._log_weights_unnorm + np.log2(w)
            
    def mult_weights_log2(self, w_log2):
        self.log_weights = self.log_weights + w_log2
        self._log_weights_unnorm = self._log_weights_unnorm + w_log2
    
    def is_empty(self):
        return self.states.size == 0 or self.log_weights.size == 0 or np.all(np.isneginf(self.log_weights))
    
    def clean_particles(self):
        if self.states.size > 0 and self._log_weights.size > 0:
            dead_particles = np.isneginf(self._log_weights)
            self.states = fix1d(self.states[~dead_particles])
            self._log_weights = self._log_weights[~dead_particles]
            self._log_weights_unnorm = self._log_weights_unnorm[~dead_particles]
            return dead_particles
        return np.array([], dtype=bool)
        
    def freeze_particles(self, indices=None):
        if indices is None or indices.size == 0 or self.states.size == 0 and self._log_weights.size == 0 or not np.any(indices):
            return
        else:
            self.add_frozen_log_weights(self.log_weights[indices])
            self._max_log_w = max(self._max_log_w, np.amax(self._log_weights_unnorm[indices]))
            self.states = fix1d(self.states[~indices])
            self.log_weights = self.log_weights[~indices]
            self._log_weights_unnorm = self._log_weights_unnorm[~indices]
        
    def clean_frozen(self):
        self.frozen_log_weights = np.array([])  # Array of the log weights of frozen particles
        self.non_frozen_log_weight = target_normalized_sum(self.frozen_log_weights)  # Log2 of total weight of non-frozen particles
        
    def random_states(self, num_particles, p='weights'):
        if p == 'weights':
            weights = normalize_probabilities(self.weights)
            s_idx = np.random.choice(a=len(self.log_weights), size=num_particles, replace=True, p=weights)
        elif p == 'uniform':
            s_idx = np.random.choice(len(self.log_weights), size=num_particles, replace=True)
        return self.states[s_idx]
    
    def add_fake_state(self, state):
        self.states = np.concatenate([state, self.states], axis=0)
        self.log_weights = np.concatenate([np.array([np.NINF]), self.log_weights], axis=0)
        
    def pop_fake_state(self):
        if self.is_empty():
            return None
        ret = self.states[0]
        if self.log_weights[0] == np.NINF:
            self.states = self.states[1:]
            self.log_weights = self.log_weights[1:]
        return ret
    
    def max_unnorm_w(self):
        if self._log_weights_unnorm.size == 0:
            return self._max_log_w
        return np.exp2(max(self._max_log_w, np.amax(self._log_weights_unnorm)))
        
    def __copy__(self):
        return ParticleBelief(states=np.copy(self.states), 
                              log_weights=np.copy(self.log_weights), 
                              time=self.time, 
                              frozen_log_weights=np.copy(self.frozen_log_weights),
                              )
    
    @staticmethod
    def belief_from_prior(prior_model, start_time, num_par):
        states = prior_model.sample(num=num_par)
        b = ParticleBelief(states=states, weights=np.ones(num_par), time=start_time)
        return b


class ParticleFilter:
    """
    A stateless (procedural) basic particle filter.
    """

    def __init__(self, transition_func, observation_model) -> None:
        self.transition_func = transition_func
        self.observation_model = observation_model

    def constant_weights(self, K):
        return np.ones(shape=K) / K

    def transition(self, belief: ParticleBelief, action):
        new_b, weighted_reward, is_terminal = self.transition_func(belief, action)
        return new_b, weighted_reward, is_terminal

    def observation(self, belief: ParticleBelief, observation, resample=True, n_resample=None):
        b = copy(belief)
        lik_log2 = self.observation_model.query_log2_density(x=observation.reshape(1, -1), theta=b.states)
        b.mult_weights_log2(lik_log2)
        if resample and not b.is_empty():
            num_new_states = len(b.log_weights) if n_resample is None else n_resample
            states = b.random_states(num_new_states, p='weights')
            b.states = states
            b.weights = self.constant_weights(num_new_states)
            b._log_weights_unnorm = b._log_weights
        return b
