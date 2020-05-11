import quandl
from gym import ObservationWrapper

from buy_sell_hold import BuySellHold
from utils import episode_df, run_sample_episode
import my_secrets


class PctChange(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev = None

    def observation(self, current):
        if self.prev is None:
            self.prev = current

        observation = (current - self.prev) / self.prev
        self.prev = current
        return observation


class EWMA(ObservationWrapper):
    def __init__(self, env, alpha):
        super().__init__(env)
        self.prev = None
        self.alpha = alpha

    def observation(self, current):
        if self.prev is None:
            self.prev = current

        observation = self.prev + self.alpha*(current - self.prev)
        self.prev = observation
        return observation


def run_pct():
    open_prices = (
        quandl.get('WIKI/MSFT',
                   start_date="2017-12-01",
                   end_date="2018-01-01",
                   api_key=my_secrets.quandl_api_key)
        ['Open']
    )

    env = BuySellHold(config={'price_series': open_prices})
    env = PctChange(env)

    observations, rewards, actions, infos = run_sample_episode(env)

    (
        episode_df(observations, rewards, actions, infos)
        .pipe(lambda df: print(df.to_string()))
    )


def run_ewma():
    open_prices = (
        quandl.get('WIKI/MSFT',
                   start_date="2017-12-01",
                   end_date="2018-01-01",
                   api_key=my_secrets.quandl_api_key)
        ['Open']
    )

    env = BuySellHold(config={'price_series': open_prices})
    env = EWMA(env, alpha=0.01)

    observations, rewards, actions, infos = run_sample_episode(env)

    (
        episode_df(observations, rewards, actions, infos)
        .pipe(lambda df: print(df.to_string()))
    )


if __name__ == "__main__":
    run_pct()
    run_ewma()
