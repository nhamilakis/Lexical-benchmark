import matplotlib.pyplot as plt


class PlotSettings:
    COLORS = {
        "human_CDI": "#d62728",
        "CHILDES": "orange",
        "0.3": "#DCE3E8",
        "0.6": "#8A9AA0",
        "0.8": "#3F5C6A",
        "1.0": "#163B4A",
        "1.5": "#021518",
        "LSTM": "#ff9896",
        "Transformer": "#c49c94",
    }

    LINESTYLES = {"CHILDES": "-", "child": "-", "0.3": "-", "0.6": "-.", "0.8": "--", "1.0": ":", "1.5": "-"}

    FIG_SIZE = (10, 10)

    @classmethod
    def configure_plot(cls, title=None, model=None):
        plt.figure(figsize=cls.FIG_SIZE)

        if title or model:
            plt.title(title or model, fontsize=18, fontweight="bold")

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle=":", linewidth=1, color="#bbbbbb")

    @classmethod
    def configure_legend(cls):
        plt.legend(title="Settings", title_fontsize=28, fontsize=28, loc="upper left", bbox_to_anchor=(1.1, 1))

    @classmethod
    def save_figure(cls, filepath):
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
