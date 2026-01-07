import os
import json


import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as log_file:
        content_list = []
        for line in tqdm(log_file.readlines(), desc="Loading data...", leave=True):
            content = json.loads(line)
            data = {"time": content["time"], **content["message"]}
            content_list.append(data)
    content_df = pd.DataFrame(content_list)
    return content_df


def main(path: str) -> None:
    content = load_data(path)
    content = content.loc[content["step"] > 0]
    largest_tiles = []
    for epoch in tqdm(
        range(content["epoch"].max() + 1),
        desc="Collecting largest tiles...",
        leave=True,
    ):
        largest_tiles.append(
            max(content["board-state"].loc[content["epoch"] == epoch].iloc[-1])
        )

    run_num = len(
        [
            f
            for f in os.listdir(os.path.join(".", "plots"))
            if os.path.isfile(os.path.join(".", "plots", f))
        ]
    )

    print("Plotting everything...")
    plt.plot(largest_tiles)
    plt.savefig(os.path.join(".", "plots", f"run_{run_num}.pdf"), format="pdf")
    print("...Done!")


if __name__ == "__main__":
    main("../agents/AgentDeepQLearning/training.logs")
