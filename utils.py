import ray
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle


def run_agent(agent, env):
    done = False
    observation = env.reset()

    observations = []
    infos = []
    actions = []
    rewards = []

    while not done:
        observations.append(observation)
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        infos.append(info)

    return observations, rewards, actions, infos


class RestoredAgent:
    def __init__(self, config, env, trainer, checkpoint):
        ray.init(ignore_reinit_error=True)
        config['num_workers'] = 1
        agent = trainer(config, env=env)
        agent.restore(checkpoint)
        self.policy = agent.workers.local_worker().get_policy()

    def act(self, observation):
        return self.policy.compute_actions([observation])[0][0]


def episode_df(observations, rewards, actions, infos):
    return (
        pd.DataFrame(infos)
        .assign(observation=observations)
        .explode("observation")
        .assign(reward=rewards)
        .assign(action=actions)
        .set_index('timestamp')
    )


def run_sample_episode(env):
    np.random.seed(1)
    infos = []
    actions = []
    observations = []
    rewards = []

    observation = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observations.append(observation)
        observation, reward, done, info = env.step(action)
        infos.append(info)
        actions.append(action)
        rewards.append(reward)
    return observations, rewards, actions, infos


def detrend(time_series):
    x = np.array(list(range(len(time_series)))).reshape(-1, 1)
    m = LinearRegression().fit(x, time_series)
    y_hat = m.predict(x)
    detrended = time_series - y_hat
    return detrended


def get_checkpoint_config(path, checkpoint):
    with open(path + "/params.pkl", "rb") as f:
        config = pickle.load(f)

    checkpoint_path = path + f"/checkpoint_{checkpoint}/checkpoint-{checkpoint}"

    return config, checkpoint_path