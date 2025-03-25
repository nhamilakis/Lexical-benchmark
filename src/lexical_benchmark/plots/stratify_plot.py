from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

ChunkIDType = tuple[int, int]  # (x, y)
BlocksType = dict[int, dict[int, list[ChunkIDType]]]


def organize_blocks_in_grid(blocks: BlocksType) -> dict[tuple[int, int], tuple[float, float]]:
    """Organize blocks in a grid layout grouped by size to prevent overlapping.

    Returns a dictionary mapping block coordinates to their positions.

    Raises:
        ValueError: If blocks is empty

    """
    if not blocks:
        raise ValueError("Empty blocks structure")

    # Calculate size of each block (number of chunks)
    block_sizes = {}
    for x in blocks:
        for y in blocks[x]:
            block_sizes[(x, y)] = len(blocks[x][y])

    # Group blocks by size
    size_groups = {}
    for coords, size in block_sizes.items():
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(coords)

    # Sort sizes for consistent layout
    sorted_sizes = sorted(size_groups.keys())

    # Place blocks in rows by size group
    positions = {}
    current_y = 0
    spacing_x = 6.5
    spacing_y = 7.5

    for size in sorted_sizes:
        size_group = sorted(size_groups[size])  # Sort blocks within the same size group

        # Calculate positions for this row
        for i, (x, y) in enumerate(size_group):
            positions[(x, y)] = (i * spacing_x, -current_y)

        # Move to next row with proper spacing
        current_y += spacing_y

    return positions


def calculate_block_sizes(blocks: BlocksType) -> dict[tuple[int, int], float]:
    """Calculate appropriate block sizes based on the number of chunks they contain.

    Returns a dictionary mapping block coordinates to their sizes.

    Raises:
        ValueError: If blocks is empty

    """
    if not blocks:
        raise ValueError("Empty blocks structure")

    block_sizes = {}

    for x in blocks:
        for y in blocks[x]:
            num_chunks = len(blocks[x][y])
            # Base size with minimum for empty blocks
            base_size = 3.5  # Minimum block size

            # Scale based on chunk count - larger blocks for more chunks
            # Using log scale to prevent exponential growth of block size
            size = base_size + 0.8 * np.log(num_chunks + 1) if num_chunks > 0 else base_size

            block_sizes[(x, y)] = size

    return block_sizes


def generate_color_map(blocks: BlocksType) -> dict[tuple[int, int], str]:
    """Generate a color map for blocks to ensure visual distinction.

    Creates a unique color for each block based on its coordinates.

    Raises:
        ValueError: If there are more blocks than available distinct colors

    """
    # Count total number of blocks for color distribution
    total_blocks = sum(len(blocks[x]) for x in blocks)

    # Get a color map with distinct colors
    cmap = plt.cm.get_cmap("tab20", max(20, total_blocks))

    # Create a mapping of block coordinates to colors
    colors = {}
    idx = 0

    for x in blocks:
        for y in blocks[x]:
            if idx >= total_blocks:
                raise ValueError("Too many blocks for distinct colors")
            colors[(x, y)] = mcolors.to_hex(cmap(idx))
            idx += 1

    return colors


def plot_block(
    ax: plt.Axes,
    pos_x: float,
    pos_y: float,
    chunks: list[ChunkIDType],
    block_color: str,
    block_size: float = 3.0,
    block_id: tuple[int, int] | None = None,
    *,
    is_dev: bool = False,
) -> None:
    """Plot a single block with its chunks.

    Draws a block as a rectangle with a title at the top and chunks clearly inside.

    Raises:
        ValueError: If block_size is not positive

    """
    if block_size <= 0:
        raise ValueError("Block size must be positive")

    # Draw outer block rectangle
    block_rect = patches.Rectangle(
        (pos_x, pos_y), block_size, block_size, linewidth=2, edgecolor="black", facecolor=block_color, alpha=0.2
    )
    ax.add_patch(block_rect)

    # Create title area at the top (10% of block height)
    title_height = block_size * 0.2
    title_rect = patches.Rectangle(
        (pos_x, pos_y + block_size - title_height),
        block_size,
        title_height,
        linewidth=1,
        edgecolor="black",
        facecolor=block_color,
        alpha=0.7,
    )
    ax.add_patch(title_rect)

    # Add block identifier or DEV label in the title area
    if is_dev:
        ax.text(
            pos_x + block_size / 2,
            pos_y + block_size - title_height / 2,
            "DEV BLOCK",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
    elif block_id:
        ax.text(
            pos_x + block_size / 2,
            pos_y + block_size - title_height / 2,
            f"Block ({block_id[0]},{block_id[1]})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Calculate the content area
    content_y = pos_y
    content_height = block_size - title_height

    # Place chunks inside block content area
    plot_chunks_in_block(ax, pos_x, content_y, chunks, block_size, content_height)


def plot_chunks_in_block(
    ax: plt.Axes, block_x: float, block_y: float, chunks: list[ChunkIDType], block_width: float, block_height: float
) -> None:
    """Plot chunk IDs inside a block with individual chunk boundaries.

    Shows chunks as distinct items within the block with improved visibility.

    Raises:
        ValueError: If there are too many chunks to fit in the block

    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        # Draw a "No chunks" message
        ax.text(
            block_x + block_width / 2,
            block_y + block_height / 2,
            "No chunks",
            ha="center",
            va="center",
            fontsize=10,
            fontstyle="italic",
            color="gray",
        )
        return

    # Determine grid dimensions based on number of chunks
    # Use a wider grid for better readability
    cols = min(4, max(1, int(np.ceil(np.sqrt(num_chunks)))))
    rows = int(np.ceil(num_chunks / cols))

    # Calculate dimensions
    margin = 0.15  # Increased margin within block
    content_width = block_width - 2 * margin
    content_height = block_height - 2 * margin

    chunk_width = content_width / cols
    chunk_height = content_height / rows

    # Minimum sizes
    chunk_width = max(chunk_width, 0.6)  # Increased minimum width
    chunk_height = max(chunk_height, 0.6)  # Increased minimum height

    # Calculate font size based on chunk size
    font_size = min(10, max(7, chunk_width * 3))  # Improved font sizing

    # Place each chunk in a cell with border
    for i, chunk in enumerate(chunks):
        row = i // cols
        col = i % cols

        # Calculate position for this chunk
        x_pos = block_x + margin + col * chunk_width
        y_pos = block_y + margin + row * chunk_height

        # Draw chunk box with stronger border
        chunk_rect = patches.Rectangle(
            (x_pos, y_pos),
            chunk_width,
            chunk_height,
            linewidth=1.5,
            edgecolor="darkgray",
            facecolor="white",
            alpha=0.7,  # Increased opacity
        )
        ax.add_patch(chunk_rect)

        # Add chunk ID text with improved visibility
        ax.text(
            x_pos + chunk_width / 2,
            y_pos + chunk_height / 2,
            f"({chunk[0]},{chunk[1]})",
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",  # Added bold font
            color="black",  # Ensured black text
        )


def visualize_blocks(  # noqa: C901, PLR0912, PLR0915
    blocks: BlocksType, dev_chunks: list[ChunkIDType] | None = None, filename: str | None = None
) -> None:
    """Visualize the entire blocks structure with blocks grouped by size.

    Creates a figure showing all blocks and their chunks with clear boundaries.
    Blocks are grouped by size (number of chunks) and arranged in rows.
    Optionally includes a special DEV block at the bottom.

    Raises:
        ValueError: If the blocks structure is empty
        OSError: If saving to file fails

    """
    if not blocks:
        raise ValueError("Empty blocks structure")

    # Calculate block sizes (number of chunks)
    size_map = {}
    for x in blocks:
        for y in blocks[x]:
            blocks[x][y].sort()
            size_map[(x, y)] = len(blocks[x][y])

    # Organize blocks in a grid layout grouped by size
    block_positions = organize_blocks_in_grid(blocks)

    # Calculate visual block sizes
    block_sizes = calculate_block_sizes(blocks)

    # Generate color map for blocks
    color_map = generate_color_map(blocks)

    # Group blocks by size for labeling
    size_groups = {}
    for coords, size in size_map.items():
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(coords)

    # Determine figure size based on block count
    block_count = sum(len(blocks[x]) for x in blocks)
    base_size = 12  # Increased base size
    fig_width = base_size + block_count * 0.9
    fig_height = base_size + block_count * 0.9

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot each block with its chunks
    for x in blocks:
        for y in blocks[x]:
            pos = block_positions[(x, y)]
            size = block_sizes[(x, y)]
            plot_block(ax, pos[0], pos[1], blocks[x][y], color_map[(x, y)], size, block_id=(x, y))

    # Add size group labels
    for size, coords_list in size_groups.items():
        # Find leftmost block in this size group
        leftmost = min(coords_list, key=lambda c: block_positions[c][0])
        leftmost_pos = block_positions[leftmost]

        # Add a label for this size group
        label_x = leftmost_pos[0] - 1.0
        label_y = leftmost_pos[1] + block_sizes[leftmost] / 2
        ax.text(
            label_x,
            label_y,
            f"Size {size}",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray", "boxstyle": "round,pad=0.5"},
        )

        # Draw a line connecting blocks of the same size
        positions = [block_positions[c] for c in coords_list]
        visual_sizes = [block_sizes[c] for c in coords_list]

        # Find bounds for this group
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] + size for pos, size in zip(positions, visual_sizes, strict=False))
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] + size for pos, size in zip(positions, visual_sizes, strict=False))

        # Draw a light background rectangle for this group
        group_rect = patches.Rectangle(
            (min_x - 0.5, min_y - 0.5),
            max_x - min_x + 1.0,
            max_y - min_y + 1.0,
            linewidth=1.5,
            linestyle="--",
            edgecolor="gray",
            facecolor="lightgray",
            alpha=0.2,
            zorder=-1,  # Ensure it's behind the blocks
        )
        ax.add_patch(group_rect)

    # The rest of the function remains the same
    # Plot special DEV block if provided
    if dev_chunks is not None:
        # Find the lowest point in the layout
        if block_positions:
            positions = list(block_positions.values())
            sizes = list(block_sizes.values())

            # The lowest block's bottom position
            min_y = min(pos[1] for pos in positions)

            # Place DEV block below all other blocks with padding
            dev_size = calculate_block_sizes({-1: {-1: dev_chunks}}).get((-1, -1), 3.0)
            dev_pos = (0, min_y - dev_size - 2.0)

            # Use a distinct color for DEV block
            dev_color = "#FF5733"  # A distinct orange-red

            # Plot DEV block
            plot_block(ax, dev_pos[0], dev_pos[1], dev_chunks, dev_color, dev_size, is_dev=True)
        else:
            # If no other blocks, just place DEV at origin
            dev_size = 3.0
            plot_block(ax, 0, -dev_size - 2.0, dev_chunks, "#FF5733", dev_size, is_dev=True)

    # Set axis properties
    ax.set_aspect("equal")
    ax.set_title("Visualization of Blocks and Chunks (Grouped by Size)")

    # Calculate plot limits with margins
    margin = 3.0  # Increased margin
    if block_positions:
        positions = list(block_positions.values())
        sizes = [block_sizes.get((x, y), 3.0) for x, y in block_positions]

        x_min = min(pos[0] for pos in positions) - margin
        x_max = max(pos[0] + size for pos, size in zip(positions, sizes, strict=False)) + margin
        y_min = min(pos[1] for pos in positions) - margin
        y_max = max(pos[1] + size for pos, size in zip(positions, sizes, strict=False)) + margin

        # Adjust for DEV block if present
        if dev_chunks is not None:
            dev_size = calculate_block_sizes({-1: {-1: dev_chunks}}).get((-1, -1), 3.0)
            y_min = min(y_min, min(pos[1] for pos in positions) - dev_size - 4.0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.grid(visible=False)
    plt.tight_layout()

    # Save figure if filename is provided
    if filename:
        try:
            file_path = Path(filename)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {file_path.absolute()}")
        except OSError as e:
            raise OSError(f"Failed to save figure: {e}") from e

    plt.show()
