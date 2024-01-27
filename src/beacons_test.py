import numpy as np

from envs.beacons import BeaconsLightDark, BeaconsLightDarkConfig, BeaconsLightDarkPlotting
from utils.quasirandom import rd_sequence

import matplotlib.pyplot as plt


def test_inc_rd_seq(beacons_env):
    # Testing incremental addition of points to rd sequence
    seq_1 = rd_sequence(dim=2, start_n=2 ** 12, scale=beacons_env.env_size,
                        shift=beacons_env.env_center - beacons_env.env_size/2)
    seq_2 = rd_sequence(dim=2, start_n=2 ** 12, end_n=2**13, scale=beacons_env.env_size,
                        shift=beacons_env.env_center - beacons_env.env_size/2)
    stacked = np.vstack([seq_1, seq_2])
    seq = rd_sequence(dim=2, start_n=2 ** 13, scale=beacons_env.env_size,
                      shift=beacons_env.env_center - beacons_env.env_size/2)
    assert np.all(np.isclose(seq, stacked))


def plot_env_gt(beacons_env, beacons_vis):
    # Testing basic visualization of rd sequence
    ax = beacons_vis.env_fig(12)
    beacons_vis.plot_env(ax)
    seq = rd_sequence(dim=2, start_n=2 ** 12, scale=beacons_env.env_size,
                      shift=beacons_env.env_center - beacons_env.env_size/2)
    next_seq = rd_sequence(dim=2, start_n=2 ** 12, end_n=2**13, scale=beacons_env.env_size,
                           shift=beacons_env.env_center - beacons_env.env_size/2)
    seq = np.vstack([seq, next_seq])
    # beacons_vis.plot_observations(ax, seq)
    # # ax.scatter(seq[:, 0], seq[:, 1])
    # plt.show()

    # # Testing observation model
    # plt.clf()
    # ax = plt.gca()
    # beacons_vis.plot_env(ax)
    # obs = beacons_env.get_observations(num_obs=100)
    # beacons_vis.plot_observations(ax, obs)
    # plt.show()

    # # Testing close observation model
    # plt.clf()
    # beacons_env.x = np.array([0.5, 2.5])
    # ax = plt.gca()
    # beacons_vis.plot_env(ax)
    # obs = beacons_env.get_observations(num_obs=20)
    # beacons_vis.plot_observations(ax, obs)
    # plt.show()

    # Testing transitioning
    beacons_env.init_state()
    a = np.array([0, beacons_env.cfg.action_length])
    plt.clf()
    ax = plt.gca()
    beacons_vis.plot_env(ax)
    beacons_vis.plot_action(ax, beacons_env.x[np.newaxis, ...], a)
    plt.show()
    for _ in range(15):
        plt.clf()
        ax = plt.gca()
        obs, reward, is_terminal = beacons_env.step(a)
        rollout_reward = beacons_env.rollout(
            0, beacons_env.x[np.newaxis, :], np.array([1]), beacons_env.get_curr_time())
        beacons_vis.plot_env(ax)
        beacons_vis.plot_observations(ax, obs)
        
        print("reward: %s, rollout reward: %s, is terminal: %s" %
              (reward, rollout_reward, is_terminal))
        plt.show()


if __name__ == "__main__":
    beacons_env = BeaconsLightDark(cfg=BeaconsLightDarkConfig())
    beacons_vis = BeaconsLightDarkPlotting(beacons_env)
    test_inc_rd_seq(beacons_env)
    plot_env_gt(beacons_env, beacons_vis)
