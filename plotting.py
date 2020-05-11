import matplotlib.pyplot as plt


def plot_episode(df):
    last_unrealized_pnl = df["unrealized_pnl"].values[-1:][0]
    start = df.index.min()
    end = df.index.max()
    title = f"Episode start{start} end:{end} unr_pnl:{last_unrealized_pnl:.4f}"
    ax = (
        df
        ['price']
        .plot(figsize=(15, 5), title=title)
    )

    (
        df
        .loc[lambda df: df['action'] == 0, 'price']
        .reset_index()
        .plot.scatter(x='timestamp', y='price',
                      ax=ax, marker="o", color='green')
    )

    (
        df
        .loc[lambda df: df['action'] == 1, 'price']
        .reset_index()
        .plot.scatter(x='timestamp', y='price',
                      ax=ax, marker="o", color='red')
    )

    (
        df['unrealized_pnl']
        .plot(ax=ax, secondary_y=True)
    )
    plt.show()