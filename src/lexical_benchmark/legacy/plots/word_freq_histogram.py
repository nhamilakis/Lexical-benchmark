import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_frequency_histogram(
    df, title: str = "Word Frequency Distribution", bin_strategy="equal_count", num_bins=10, custom_bins=None
):
    """Create a histogram of word frequencies with multiple binning strategies.

    Arguments:
    ---------
    df (pandas.DataFrame): DataFrame with 'freq' column
    bin_strategy (str): Binning method
        - 'equal_count': Equal number of words per bin
        - 'equal_width': Equal width frequency ranges
        - 'custom': Use custom bin edges
    num_bins (int): Number of bins to create (for 'equal_count' and 'equal_width')
    custom_bins (list/array): Custom bin edges (for 'custom' strategy)

    Returns:
    -------
    matplotlib.figure.Figure: Plotted histogram figure

    """
    # Create the figure and axis
    plt.figure(figsize=(14, 7))

    # Ensure frequency column is numeric
    frequencies = pd.to_numeric(df["freq"], errors="coerce")
    # Remove any NaN values
    frequencies = frequencies.dropna()
    # Convert to regular numpy array
    freq_array = frequencies.values

    # Determine bin edges based on strategy
    if bin_strategy == "equal_count":
        percentile_edges = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(freq_array, percentile_edges)

        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)

    elif bin_strategy == "equal_width":
        # Create bins of equal width across frequency range
        bin_edges = np.linspace(frequencies.min(), frequencies.max(), num_bins + 1)

    elif bin_strategy == "custom":
        # Use user-provided custom bins
        if custom_bins is None:
            raise ValueError("Must provide custom_bins when using 'custom' strategy")
        bin_edges = np.array(custom_bins)

    else:
        raise ValueError("Invalid bin_strategy. Choose 'equal_count', 'equal_width', or 'custom'")

    # Plot the histogram
    hist, actual_edges, _ = plt.hist(frequencies, bins=bin_edges, edgecolor="black", alpha=0.7)

    # Customize the plot
    plt.title(title, fontsize=15)
    plt.xlabel("Frequency Range", fontsize=12)
    plt.ylabel("Number of Words", fontsize=12)

    # Add grid for readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Create x-tick labels showing the bin ranges
    tick_labels = [f"{actual_edges[i]:.2f}-{actual_edges[i + 1]:.2f}" for i in range(len(actual_edges) - 1)]
    plt.xticks((actual_edges[:-1] + actual_edges[1:]) / 2, tick_labels, rotation=45, ha="right")

    # Add bin count annotations
    for i, count in enumerate(hist):
        plt.text((actual_edges[i] + actual_edges[i + 1]) / 2, count, str(int(count)), ha="center", va="bottom")

    plt.tight_layout()

    return plt
