<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Binance_BTCUSDT_d.csv", index_col=False)


close_position = data["close"]


if __name__ == "__main__":
    plt.plot(close_position)
    plt.show()

=======
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Binance_BTCUSDT_d.csv", index_col=False)


close_position = data["close"]


if __name__ == "__main__":
    plt.plot(close_position)
    plt.show()

>>>>>>> 68a3b1e894bc20852fef2f9f78e8d7e5ae7e00a4
