"""Latency randomization for training: every episode draws an action delay and an observation delay,
each uniform in 0..max_steps control steps, and the policy acts through them.

The semantics are those of the robustness sweep's ActionDelay and SensorDelay
(falcons/benchmark/robustness.py), so a policy is trained on the disturbance it is later scored
under: until an episode has issued k actions the actuator receives a neutral (zero) command, and
until it has k observations the policy sees the first one. The two delays are drawn independently,
so an episode may carry either, both or neither.

The wrapped env auto-resets terminated episodes inside `step`; their histories restart there and
their delays are redrawn. `info["terminal_obs"]` is passed through undelayed: it only feeds the
critic's bootstrap at a truncation, where the true final state is the better estimate.

    env = LatencyDR(AttitudeEnv(...), max_steps=10)      # 0-100 ms at 100 Hz
"""
import torch


class LatencyDR:
    def __init__(self, env, max_steps, seed=0):
        self.env, self.K = env, int(max_steps)
        self._gen = torch.Generator(device=env.device)
        self._gen.manual_seed(int(seed))

    def __getattr__(self, name):            # everything else (num_obs, set_curriculum, ...) is the env's
        return getattr(self.env, name)

    def _draw(self, n):
        return torch.randint(0, self.K + 1, (n,), generator=self._gen, device=self.env.device)

    def _delayed_obs(self):
        return self._obuf[self._k_obs, self._idx]

    def reset(self):
        obs = self.env.reset().clone()      # reset returns a view of the warp buffer
        n, dev = obs.shape[0], obs.device
        self._idx = torch.arange(n, device=dev)
        self._k_act, self._k_obs = self._draw(n), self._draw(n)
        self._abuf = torch.zeros(self.K + 1, n, self.env.num_act, device=dev)   # [j] = issued j steps ago
        self._obuf = obs.unsqueeze(0).repeat(self.K + 1, 1, 1)                 # [j] = observed j steps ago
        return self._delayed_obs()

    def step(self, action):
        self._abuf = torch.roll(self._abuf, 1, 0)
        self._abuf[0] = action
        obs, reward, done, info = self.env.step(self._abuf[self._k_act, self._idx])
        self._obuf = torch.roll(self._obuf, 1, 0)
        self._obuf[0] = obs
        if done.any():                      # these episodes were reset inside env.step
            self._abuf[:, done] = 0.0
            self._obuf[:, done] = obs[done]
            m = int(done.sum())
            self._k_act[done], self._k_obs[done] = self._draw(m), self._draw(m)
        return self._delayed_obs(), reward, done, info
