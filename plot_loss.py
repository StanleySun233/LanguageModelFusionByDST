import json

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Loss:
    def __init__(self, name):
        with open(f"./save/loss/{name}.json", 'r') as f:
            data = json.load(f)

        plt.plot(range(1, len(data["accuracy"]) + 1), data["accuracy"], label="accuracy")
        plt.plot(range(1, len(data["valid_accuracy"]) + 1), data["valid_accuracy"], label="valid_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Rate")
        plt.legend()
        plt.title(f"Accuracy of {name}")
        plt.savefig(f'./save/plot_loss/accuracy_{name}.png')
        plt.show()

        plt.plot(range(1, len(data["loss"]) + 1), data["loss"], label="loss")
        plt.plot(range(1, len(data["valid_loss"]) + 1), data["valid_loss"], label="valid_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Rate")
        plt.legend()
        plt.title(f"Loss of {name}")
        plt.savefig(f'./save/plot_loss/loss_{name}.png')
        plt.show()


if __name__ == "__main__":
    ls = Loss("BP")
