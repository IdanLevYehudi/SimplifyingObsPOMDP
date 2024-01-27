import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from configs.solvers.pftdpw_params import PFTDPWConfig
from inference.particle_filter import ParticleFilter, ParticleBelief
from mes_simp.mes_simp import MeasurementSimplification

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.colors import to_rgba_array

from tqdm import tqdm
from copy import copy


def sort_a_on_b(a, b):
        return [x for _, x in sorted(zip(a, b))]

    
class PFTNode:
    # Having pointer to parent was unnecessary and slowed vscode down.
    # It may be needed if we want to switch recursion to iterative solving.
    #  parent: typing.Any  # Pointer to parent PFTNode
    node_count = 0

    node_id: int  # Some id for hashing
    children: dict

    def __init__(self, children) -> None:
        self.node_id = PFTNode.node_count
        PFTNode.node_count += 1

        self.children = children

    def __hash__(self) -> int:
        return hash(self.node_id)
    
    @classmethod
    def reset_node_count(cls):
        cls.node_count = 0


class PropogatedNode(PFTNode):
    rs: dict  # Dictionary of observation -> reward
    ms: dict  # Dictionary of observation -> m_i bound
    n_visits: int
    
    def __init__(self, children, rs, ms, n_visits) -> None:
        super().__init__(children)
        self.rs = rs
        self.ms = ms
        self.n_visits = n_visits


class PosteriorNode(PFTNode):
    b: ParticleBelief
    q: dict  # Dictionary of action -> float for cumulative reward
    phi: dict  # Dictionary of action -> float for cumulative bound
    n_visits: int
    children_visits: dict

    def __init__(self, children, b, q, phi, n_visits, children_visits) -> None:
        super().__init__(children)
        self.b = b
        self.q = q
        self.phi = phi
        self.n_visits = n_visits
        self.children_visits = children_visits


class BeliefTree:
    root: PosteriorNode

    def __init__(self, b_init) -> None:
        self.root = PosteriorNode(children={}, b=b_init,
                                  n_visits=0, children_visits={}, q={}, phi={})

    def insert_posterior_node(self, parent: PropogatedNode, obs, b: ParticleBelief, r, m):
        # Insert a belief node into the DPW tree
        post = PosteriorNode(n_visits=0,
                             children={}, children_visits={}, b=b, q={}, phi={})
        parent.children[obs] = post
        parent.rs[obs] = r
        parent.ms[obs] = m
        return post

    def insert_propogated_node(self, parent: PosteriorNode, a):
        # Insert an action node stemming from a belief node into the DPW tree
        prop = PropogatedNode(children={}, rs={}, ms={}, n_visits=0)
        parent.children[a] = prop
        parent.children_visits[a] = 0
        parent.q[a] = 0
        parent.phi[a] = 0
        return prop
    
    def max_unnorm_w(self):
        max_w = 0
        nodes_queue = [self.root]
        while nodes_queue:
            node = nodes_queue.pop()
            b = node.b
            max_w = max(b.max_unnorm_w(), max_w)
            for k1, action_node in node.children.items():
                for k2, post_node in action_node.children.items():
                    nodes_queue.append(post_node)
        return max_w
            

def argmax_dict(d):
    if d:
        return max(d, key=d.get)
    else:
        return 0

def max_policy(pnode: PosteriorNode):
    return argmax_dict(pnode.q)    


class PFTDPW:
    def __init__(self, cfg: PFTDPWConfig, environment, obs_model, do_mes_simp=False, mes_simp: MeasurementSimplification = None):
        self.tree = None
        self.state_obs_ids = list()
        self.action_ids = list()
        # Set up modules
        self.env = environment
        self.obs_model = obs_model
        
        self.do_mes_simp = do_mes_simp
        self.mes_simp = mes_simp
        # Set up parameters
        self.cfg = cfg

        self.pf = ParticleFilter(self.env.transition, self.obs_model)

    def initialize_tree(self, b_init):
        self.tree = BeliefTree(copy(b_init))
        self.state_obs_ids = list()
        self.action_ids = list()
        PFTNode.reset_node_count()

    def insert_id(self, lst: list, obj):
        new_id = len(lst)
        lst.append(obj)
        return new_id

    def insert_state_obs(self, state, obs):
        return self.insert_id(self.state_obs_ids, (state, obs))

    def insert_action(self, action):
        return self.insert_id(self.action_ids, action)

    def is_terminal(self, b):
        # Check if the belief node is terminal (i.e. all particles are terminal)
        return b.is_empty() or self.env.is_terminal(b.states, b.time)

    def next_action(self, b, expanded_actions):
        # Generate next action from belief
        return self.env.sample_action(b, expanded_actions)

    def particle_filter_step(self, b: ParticleBelief, a):
        # Generate b' from T(b,a) and also insert it into the tree
        if self.do_mes_simp:
            m = self.mes_simp.estimate_m_i_belief(b, a)
        else:
            m = 0
        b_prime, reward, is_terminal = self.pf.transition(b, a)
        if not is_terminal and not b_prime.is_empty():
            # Generate an observation from a random state
            rand_state = b_prime.random_states(1, p='weights')
        else:
            # Generate some observation based on some state
            # Doesn't really matter, because there will be no more children (since this is a terminated belief)
            if not b_prime.is_empty():
                rand_state = b_prime.random_states(1, p='uniform')
            else:
                rand_state = b.random_states(1, p='uniform')
        rand_obs = np.squeeze(self.obs_model.sample(rand_state, 1))
        if not is_terminal:
            b_prime = self.pf.observation(b_prime, rand_obs, resample=self.cfg.pf_resample)
        return b_prime, reward, m, rand_state, rand_obs

    def rollout(self, b: ParticleBelief):
        # Rollout simulation starting from belief b
        if self.is_terminal(b):
            return 0, 0
        # rollout_state = b.random_state(p='weights')
        rollout_state = b.emp_mean().reshape(1, -1)
        return self.env.rollout(rollout_state, b, self.mes_simp.estimate_m_i_belief if self.do_mes_simp else None)

    def extract_action_sequence(self, start_node=None):
        # Extract an action sequence from this tree
        # In posterior nodes, choose the action with maximum q value
        # In propogated nodes, choose the belief with the maximum visitation count
        if start_node is None:
            node = self.tree.root
        else:
            node = start_node
        actions = []
        while node.children and node.q:
            a_id = argmax_dict(node.q)
            actions.append(self.action_ids[a_id])
            if a_id in node.children:
                prop = node.children[a_id]
                if prop.children:
                    obs_id = max(prop.children, key=lambda k: prop.children[k].n_visits)
                    node = prop.children[obs_id]
                else:
                    break
            else:
                break
        return actions
    
    def extract_action_at_root(self, policy='expected_value'):
        if policy == 'expected_value':
            return self.action_ids[argmax_dict(self.tree.root.q)]
        elif policy == 'expected_lb':
            q_minus_m_root = {a: self.tree.root.q[a] - self.tree.root.phi[a] for a in self.tree.root.q}
            return self.action_ids[argmax_dict(q_minus_m_root)]
        elif policy == 'expected_ub':
            q_plus_m_root = {a: self.tree.root.q[a] + self.tree.root.phi[a] for a in self.tree.root.q}
            return self.action_ids[argmax_dict(q_plus_m_root)]

    def solve(self, b):
        # call plan when given states and weights
        a_id = self.plan(b)
        return a_id

    def plan(self, b, progress=True):
        """
        Build a belief tree from given belief.

        Args:
            b (_type_): ParticleBelief,
            progress (bool, optional): Whether to print progress bar. Defaults to True.

        Returns:
            int: Best action id from root.
        """
        # Builds a DPW tree and returns the best next action
        # Construct the DPW tree
        self.initialize_tree(b)

        # Plan with the tree by querying the tree for n_query number of times
        iterator = tqdm(range(self.cfg.num_query)) if progress else range(self.cfg.num_query)
        for _ in iterator:
            self.simulate(self.tree.root, self.cfg.horizon)

        # Find the best action from the root node
        return argmax_dict(self.tree.root.q)

    def ucb(self, node: PosteriorNode):
        log_nb = np.log(node.n_visits)
        c = self.cfg.ucb_exploration
        return argmax_dict({a: node.q[a] + c * np.sqrt(
            log_nb / node.children_visits[a]) for a in node.children})

    def prog_widen(self, node, n_visits, k, alpha):
        return len(node.children) <= k * (n_visits ** alpha)

    def action_prog_widen(self, node: PosteriorNode):
        if self.prog_widen(node, node.n_visits,
                           self.cfg.k_action, self.cfg.alpha_action):
            expanded_actions = np.array([self.action_ids[a_id] for a_id in node.children])
            a = self.next_action(node.b, expanded_actions)
            if a is not None:
                a_id = self.insert_action(a)
                self.tree.insert_propogated_node(node, a_id)
                return a_id
            else:
                return self.ucb(node)
        else:
            return self.ucb(node)

    def simulate(self, node: PosteriorNode, d):
        # Simulates dynamics with a DPW tree
        b = node.b

        # Check if d == 0 (full depth) or belief is terminal
        if d == 0:
            return self.rollout(b)
        elif self.is_terminal(b):
            return 0.0, 0.0

        a_id = self.action_prog_widen(node)
        prop_node = node.children[a_id]
        # State PW
        if self.prog_widen(prop_node, node.children_visits[a_id],
                           self.cfg.k_observation, self.cfg.alpha_observation):
            # If no state present or PW condition met, do PW
            bp, r, m, rand_state, rand_obs = self.particle_filter_step(b, self.action_ids[a_id])
            obs_id = self.insert_state_obs(rand_state, rand_obs)
            self.tree.insert_posterior_node(prop_node, obs_id, bp, r, m)
            q_rollout, phi = self.rollout(bp)
            q = r + self.cfg.discount * q_rollout
        else:
            # Otherwise pick a belief node at random
            obs_ids = list(prop_node.children.keys())
            obs_id = obs_ids[int(np.random.choice(range(len(obs_ids)), 1))]
            post_node = prop_node.children[obs_id]
            r = prop_node.rs[obs_id]
            m = prop_node.ms[obs_id]
            q_sim, phi = self.simulate(post_node, d - 1)
            q = r + self.cfg.discount * q_sim
        phi = phi + m
        # Update the counters & quantities
        node.n_visits += 1
        node.children_visits[a_id] += 1
        node.children[a_id].n_visits += 1
        node.q[a_id] += (q - node.q[a_id]) / node.children_visits[a_id]
        node.phi[a_id] += (phi - node.phi[a_id]) / node.children_visits[a_id]
        return q, phi


class BeliefTreePlotting:
    @staticmethod
    def action_labels(action):
        if np.allclose(action, np.array([0, 1])):
            return 'U'
        if np.allclose(action, np.array([0, -1])):
            return 'D'
        if np.allclose(action, np.array([1, 0])):
            return 'R'
        if np.allclose(action, np.array([-1, 0])):
            return 'L'
    
    @staticmethod
    def create_nx_tree(tree: BeliefTree):
        T = nx.Graph()
        min_q = np.inf
        max_q = -np.inf

        nodes_to_expand = [tree.root]
        while nodes_to_expand:
            node = nodes_to_expand.pop()
            T.add_node(node)
            for c in node.children:
                if isinstance(node, PosteriorNode):
                    q = node.q[c]
                elif isinstance(node, PropogatedNode):
                    q = node.rs[c] + argmax_dict(node.children[c].q)
                T.add_edge(node, node.children[c], q=q)
                min_q = min(min_q, q)
                max_q = max(max_q, q)
                nodes_to_expand.append(node.children[c])
        return T, min_q, max_q

    @staticmethod
    def plot_tree(ax: Axes, tree: BeliefTree):
        T, min_q, max_q = BeliefTreePlotting.create_nx_tree(tree)
        pos = nx.nx_agraph.graphviz_layout(T, prog="dot")
        edges = T.edges(data=True)
        q_normalizer = 1 if max_q == min_q else 1 / (max_q - min_q)
        edge_colors = [plt.cm.BuPu((d['q'] - min_q) * q_normalizer) if isinstance(u, PosteriorNode) else plt.cm.Reds((d['q'] - min_q) * q_normalizer) for u, v, d in edges]
        options = {
            "node_color": "#A0CBE2",
            # "edges": edges,
            "edge_color": edge_colors,
            "node_size": 10,
            # "edge_cmap": plt.cm.Reds,
            # "edge_vmin": min_q,
            # "edge_vmax": max_q,
            "with_labels": False,
        }
        nx.draw(T, pos, ax=ax, **options)
    
    @staticmethod
    def plot_actions_bounds_at_root(ax: Axes, pft_tree: PFTDPW):
        root_actions_ids = [a for a in pft_tree.tree.root.q.keys()]
        root_actions = [pft_tree.action_ids[c] for c in root_actions_ids]
        root_actions_labels = [BeliefTreePlotting.action_labels(a) for a in root_actions]
        labels_sorted = sorted(root_actions_labels)
        q_values_root = sort_a_on_b(root_actions_labels, [pft_tree.tree.root.q[c] for c in root_actions_ids])
        max_q_root = np.amax(np.array(q_values_root))
        m_values_root = sort_a_on_b(root_actions_labels, [pft_tree.tree.root.phi[c] for c in root_actions_ids])
        q_minus_m_root = q_values_root - np.array(m_values_root)
        max_q_minus_m_root = np.amax(q_minus_m_root)
        colors = to_rgba_array(['g' if q_values_root[i]==max_q_root else 'b' if q_minus_m_root[i]==max_q_minus_m_root else 'r' for i in range(len(root_actions_ids))])
        ax.bar(x=labels_sorted, height=q_values_root, yerr=m_values_root, color=colors)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        ax.set_xlabel("Actions", fontsize=24)
        ax.set_ylabel("Expected Action Values", fontsize=24)