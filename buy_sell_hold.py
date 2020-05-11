import gym
import numpy as np
import quandl
from gym.spaces import Discrete, Box
import random
from utils import episode_df, run_sample_episode
import my_secrets


class BuySellHold(gym.Env):
    def __init__(self, config):
        self.action_space = Discrete(3) # Buy Sell Hold
        # only returns the open price for now
        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(1,), dtype=np.float32)

        self.price_series = (
            config['price_series']
            .values
        )
        self.timestamps = (
            config['price_series']
            .index
            .values
        )

        self.max_episode_length = len(self.price_series) - 1
        if "episode_length" in config:
            self.episode_length = config["episode_length"]
            self.episode_length = min(self.episode_length, self.max_episode_length)
        else:
            self.episode_length = None

    def unrealized_pnl(self, sell_price):
        return self.capital + self.possession * sell_price

    def reset(self):
        self.episode_time = 0

        if self.episode_length is None:
            price = self.price_series[self.episode_time]
        else:
            max_start_index = self.max_episode_length - self.episode_length
            self.episode_start_index = random.randint(0, max_start_index)
            price = self.price_series[self.episode_start_index + self.episode_time]


        self.capital = 1
        self.possession = 0
        self.prev_unrealized_pnl = self.unrealized_pnl(sell_price=price)

        observation = np.array([price], dtype=np.float32)
        return observation

    def step(self, action):
        self.episode_time += 1

        if self.episode_length is None:
            price = self.price_series[self.episode_time]
        else:
            price = self.price_series[self.episode_start_index + self.episode_time]

        if action == 0:  # BUY
            if self.capital > 0:
                traded_amount = self.capital / price
                self.possession += traded_amount
                self.capital = 0
        elif action == 1:  # Sell
            if self.possession > 0:
                traded_amount = self.possession * price
                self.capital += traded_amount
                self.possession = 0

        unrealized_pnl = self.unrealized_pnl(sell_price=price)
        reward = (unrealized_pnl - self.prev_unrealized_pnl) / self.prev_unrealized_pnl
        self.prev_unrealized_pnl = unrealized_pnl


        if self.episode_length is None:
            done = self.episode_time == self.max_episode_length
        else:
            done = self.episode_time == self.episode_length

        info = {
            "capital": self.capital,
            "posession": self.possession,
            "unrealized_pnl": unrealized_pnl,
            "price": price,
            "timestamp": self.timestamps[self.episode_time]
        }
        observation = np.array([price], dtype=np.float32)

        return observation, reward, done, info


def run():
    open_prices = (
        quandl.get('WIKI/MSFT',
                   start_date="2017-12-01",
                   end_date="2018-01-01",
                   api_key=my_secrets.quandl_api_key)
        ['Open']
    )

    env = BuySellHold(config={'price_series': open_prices})
    observations, rewards, actions, infos = run_sample_episode(env)

    (
        episode_df(observations, rewards, actions, infos)
        .pipe(lambda df: print(df.to_string()))
    )

    env = BuySellHold(config={'price_series': open_prices, 'episode_length': 5})
    observations, rewards, actions, infos = run_sample_episode(env)

    (
        episode_df(observations, rewards, actions, infos)
        .pipe(lambda df: print(df.to_string()))
    )


if __name__ == "__main__":
    run()