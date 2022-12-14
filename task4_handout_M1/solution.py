import numpy as np
import matplotlib.pyplot as plt

import time

import scipy.signal
from gym.spaces import Box, Discrete

import torch
from torch.optim import Adam
import torch.nn as nn
from  torch.distributions.categorical import Categorical


def discount_cumsum(x, discount):
    """Compute  cumulative sums of vectors."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    """Produces MLP with given layers and activation function."""
    layer = []
    for i in range(len(sizes)-1):
        lay = nn.Linear(sizes[i],sizes[i+1])
        layer += [lay,activation()]

    layer.pop()
    layer.append(output_activation())
    return nn.Sequential(*layer)


class Actor(nn.Module):
    """A class for the policy network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """Takes the observation and outputs a distribution over actions."""
        assert torch.is_tensor(obs)
        pi_log_prob = self.logits_net(obs)
        return Categorical(logits=pi_log_prob)

    def _log_prob_from_distribution(self, pi, act):
        """
        Take a distribution and action, then gives the log-probability of the action
        under that distribution.
        """
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute the
        log-likelihood of given actions under those distributions.
        """
        assert torch.is_tensor(obs)
        pi = self._distribution(obs)
        if act == None:
            log_like = None
        else:
            log_like = self._log_prob_from_distribution(pi,act)

        return pi, log_like


class Critic(nn.Module):
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        """
        Return the value estimate for a given observation.
        """
        assert torch.is_tensor(obs)
        return torch.squeeze(self.v_net(obs), -1)


class VPGBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        # Pointer to the latest data point in the buffer.
        self.ptr = 0 
        # Pointer to the start of the trajectory.
        self.path_start_idx = 0
        # Maximum size of the buffer. 
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the observed outcome in    
        """

        # buffer has to have room so you can store
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        # Update pointer after data is stored.
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Calculate for a trajectory:
            1) TD residuals
            2) discounted rewards-to-go
        Store these into self.ret_buf, and self.tdres_buf respectively.

        The function is called after a trajectory ends.
        """

        # Get the indexes where TD residuals and discounted 
        # rewards-to-go are stored.
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.ret_buf[self.ptr:self.path_start_idx] = (
            np.cumsum(self.rew_buf[self.ptr:self.path_start_idx][::-1])[::-1]
        )

        # TD residuals calculation.
        # td_res = r_t + V^π(s_t+1) − V^π(s_t)
        delta = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.tdres_buf[path_slice] = discount_cumsum(delta,self.gamma*self.lam)

        # Rewards-to-go calculation. 
        self.ret_buf[path_slice] = discount_cumsum(rews[:-1],self.gamma)

        # Update the path_start_idx
        self.path_start_idx = self.ptr
        pass

    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        self.tdres_buf = self.tdres_buf
        tdres_mean = np.mean(self.tdres_buf)
        tdres_std = np.std(self.tdres_buf)
        self.tdres_buf = (self.tdres_buf - tdres_mean) / tdres_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    tdres=self.tdres_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Agent:
    def __init__(self, env, activation=nn.Tanh):
        self.env = env
        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        hidden_sizes = [self.hid]*self.l
        obs_dim = 8
        self.actor = Actor(obs_dim, 4, hidden_sizes, activation)
        self.critic = Critic(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take an state and return action, value function, and log-likelihood
        of chosen action.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            pi, _ = self.actor(state)
            act = pi.sample()
            _, logp = self.actor(state,act)
            v = self.critic(state)

        return act.item(), v.item(), logp.item()

    def act(self, state):
        return self.step(state)[0]

    def get_action(self, obs):
        """
        Sample an action from your policy/actor.

        IMPORTANT: This function called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """
        # TODO: Implement this function.
        # Currently, this just returns a random action.
        obs = torch.as_tensor(obs, dtype=torch.float32)
        pi, _ = self.actor(obs)
        act = pi.sample()
        
        return act


def train(env, seed=0):
    """
    Main training loop.

    IMPORTANT: This function is called by the checker to train your agent.
    You SHOULD NOT change the arguments this function takes and what it outputs!
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # The observations are 8 dimensional vectors, and the actions are numbers,
    # i.e. 0-dimensional vectors (hence act_dim is an empty list).
    obs_dim = [8]
    act_dim = []

    # initialize agent
    agent = Agent(env)

    # Training parameters
    # You may wish to change the following settings for the buffer and training
    # Number of training steps per epoch
    steps_per_epoch = 2000
    # Number of epochs to train for
    epochs = 75
    # The longest an episode can go on before cutting it off
    max_ep_len = 300
    # Discount factor for weighting future rewards
    gamma = 0.99
    lam = 0.97

    # Learning rates for actor and critic function
    actor_lr = 3e-3
    critic_lr = 1e-3

    # Set up buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Def Loss function for critic
    loss_fn_critic = nn.MSELoss()

    # Initialize the ADAM optimizer using the parameters 
    # of the actor and then critic networks
    actor_optimizer = Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optimizer = Adam(agent.critic.parameters(), lr=critic_lr)

    # Initialize the environment
    state, ep_ret, ep_len = agent.env.reset(), 0, 0

    # Main training loop: collect experience in env and update / log each epoch
    for epoch in range(epochs):
        ep_returns = []
        # Saving the values and log probs with the gradient calalculation
        value_with_grad = []
        logp_with_grad = []
        for t in range(steps_per_epoch):
            a, v, logp = agent.step(torch.as_tensor(state, dtype=torch.float32))

            _, logp_grad = agent.actor(torch.as_tensor(state, dtype=torch.float32), 
                                       torch.as_tensor(a, dtype=torch.float32))
            logp_with_grad.append(logp_grad)

            v_grad = agent.critic(torch.as_tensor(state, dtype=torch.float32))
            value_with_grad.append(v_grad)

            next_state, r, terminal = agent.env.transition(a)
            ep_ret += r
            ep_len += 1

            # Log transition
            buf.store(state, a, r, v, logp)

            # Update state (critical!)
            state = next_state

            timeout = ep_len == max_ep_len
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or timeout or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = agent.step(torch.as_tensor(state, dtype=torch.float32))
                else:
                    v = 0
                if timeout or terminal:
                    ep_returns.append(ep_ret)  # only store return when episode ended
                buf.end_traj(v)
                state, ep_ret, ep_len = agent.env.reset(), 0, 0

        mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
        print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

        # This is the end of an epoch, so here is where you likely want to update
        # the actor and / or critic function.

        """
        data = {
            obs, act, ret, tdres (mean 0 var 1), logp
        }
        """
        data = buf.get()
        ad = data['tdres']
        #logp = data['logp']
        rewards_to_go = data['ret']
        value_with_grad_torch = torch.stack(value_with_grad)
        logp_with_grad_torch = torch.stack(logp_with_grad)
        
        # Do 1 policy gradient update
        # Hint: you need to compute a 'loss' such that its derivative with respect 
        # to the actor parameters is the policy gradient.
        actor_optimizer.zero_grad() 
        loss = torch.sum(-ad*logp_with_grad_torch)
        loss.backward()
        actor_optimizer.step()

        # We suggest to do 100 iterations of value function updates
        # loss_fn_critic = (A - r - gamma G old)^2
        #for _ in range(100):
        critic_optimizer.zero_grad()
        loss = loss_fn_critic(value_with_grad_torch, rewards_to_go)
        loss.backward()
        critic_optimizer.step()

    return agent


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    This function is only meant for testing purposes. Any changes made here do not
    affect the submission. 
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()
    env.seed(0)
    
    agent = train(env)

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action)
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")


if __name__ == "__main__":
    main()
