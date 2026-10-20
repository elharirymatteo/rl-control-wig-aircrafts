import torch

from falcons.envs.latency import LatencyDR


class _Recorder:
    """A two-env stand-in that returns obs = step count and records the action it was handed."""
    num_act, device = 1, "cpu"

    def __init__(self, done_at=None):
        self.t, self.got, self.done_at = 0, [], done_at

    def reset(self):
        self.t = 0
        return torch.zeros(2, 1)

    def step(self, action):
        self.t += 1
        self.got.append(action.clone())
        done = torch.tensor([self.t == self.done_at, False])
        return torch.full((2, 1), float(self.t)), torch.zeros(2), done, {}


def _wrap(env, k_act, k_obs, K=5):
    w = LatencyDR(env, max_steps=K)
    obs = w.reset()
    w._k_act[:] = k_act
    w._k_obs[:] = k_obs
    return w, obs


def test_actuator_gets_the_command_from_k_steps_ago_and_neutral_before():
    env = _Recorder()
    w, _ = _wrap(env, k_act=2, k_obs=0)
    for t in range(1, 6):
        w.step(torch.full((2, 1), float(t)))
    assert [float(a[0]) for a in env.got] == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_policy_sees_the_obs_from_k_steps_ago_and_the_first_before():
    env = _Recorder()
    w, obs0 = _wrap(env, k_act=0, k_obs=3)
    seen = [float(w.step(torch.zeros(2, 1))[0][0]) for _ in range(6)]
    assert float(obs0[0]) == 0.0 and seen == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_a_reset_episode_restarts_its_history_and_keeps_its_neighbour():
    env = _Recorder(done_at=3)
    w, _ = _wrap(env, k_act=2, k_obs=0)
    for t in range(1, 4):                    # env 0 terminates on step 3
        w.step(torch.full((2, 1), float(t)))
    w._k_act[:] = 2                          # its delay was redrawn; pin it for the check
    for t in range(4, 7):
        w.step(torch.full((2, 1), float(t)))
    env0 = [float(a[0]) for a in env.got]
    env1 = [float(a[1]) for a in env.got]
    assert env1 == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]      # untouched
    assert env0[3:] == [0.0, 0.0, 4.0]                 # restarted: neutral again, then its own history
