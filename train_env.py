import quandl
import ray
from ray.tune import tune, register_env, grid_search
from buy_sell_hold import BuySellHold
from wrappers import PctChange, EWMA
import my_secrets
from utils import detrend


def buy_sell_hold_pct_ewma(config):
    alpha = config.pop('alpha')
    env = BuySellHold(config)
    env = PctChange(env)
    env = EWMA(env, alpha=alpha)
    return env


env_name = "buy_sell_hold_pct_ewma"
register_env(env_name, lambda config: buy_sell_hold_pct_ewma(config))

if __name__ == "__main__":
    ray.init()
    open_prices_detrended = (
        quandl.get('WIKI/MSFT',
                   start_date="2014-01-01",
                   end_date="2017-01-01",
                   api_key=my_secrets.quandl_api_key)
        .assign(Open=lambda df: detrend(df['Open']))
        .assign(Open=lambda df: df['Open'] - df['Open'].min() + 1)
        ['Open']
    )

    tune.run_experiments({
        "PPO_Detrended": {
            "run": "PPO",
            "stop": {
                "time_total_s": 60 * 10,
            },
            "checkpoint_at_end": True,
            "checkpoint_freq": 20,

            "config": {
                "env": env_name,
                "num_workers": 2,  # parallelism

                "lr": grid_search([5e-4, 5e-5]),  # try different lrs
                "train_batch_size": grid_search([4_000]),
                "clip_param": grid_search([0.1, 0.3]),
                "gamma": grid_search([0.99]),
                "model": {
                    "fcnet_hiddens": grid_search([[256, 256], [64, 64]])
                },

            "env_config": {
                "price_series": open_prices_detrended,
                'alpha': grid_search([0.25, 0.5]),
                "episode_length": 60
                }
            }
        }

    })
