# my_alluvial.py (Updated)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba
from gradient_patch import *
import pandas as pd # Import pandas



def get_flow_polygon(start_coord, end_coord, width, h_gap=1.0):
    """Creates polygon coordinates for a flow using curved splines"""
    x0, y0 = start_coord
    x1, y1 = end_coord
    cp1_x = x0 + h_gap * 0.2
    cp2_x = x1 - h_gap * 0.2
    t = np.linspace(0, 1, 20)
    top_curve = np.array([(1-t)**3 * x0 + 3*t*(1-t)**2 * cp1_x + 3*t**2*(1-t) * cp2_x + t**3 * x1,
                         (1-t)**3 * (y0 + width/2) + 3*t*(1-t)**2 * (y0 + width/2) +
                         3*t**2*(1-t) * (y1 + width/2) + t**3 * (y1 + width/2)]).T
    bottom_curve = np.array([(1-t)**3 * x0 + 3*t*(1-t)**2 * cp1_x + 3*t**2*(1-t) * cp2_x + t**3 * x1,
                            (1-t)**3 * (y0 - width/2) + 3*t*(1-t)**2 * (y0 - width/2) +
                            3*t**2*(1-t) * (y1 - width/2) + t**3 * (y1 - width/2)]).T
    poly = np.vstack([top_curve, bottom_curve[::-1]])
    return poly

def preprocess_matrix(matrix):
    """Preprocesses matrix, removing zero rows/cols, returns filtered matrix and original indices."""
    # Ensure input is DataFrame for consistent label handling
    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix) # Basic conversion

    original_row_labels = list(matrix.index)
    original_col_labels = list(matrix.columns)

    matrix_np = matrix.to_numpy()

    if matrix_np.size == 0:
        # Return empty DataFrame and empty lists if input is empty
        return pd.DataFrame(), [], [], [], []

    if np.any(matrix_np < 0):
        raise ValueError("Matrix cannot contain negative values")

    row_mask = ~np.all(matrix_np == 0, axis=1)
    col_mask = ~np.all(matrix_np == 0, axis=0)

    # Original indices of the rows/cols that were kept
    row_indices_kept = np.where(row_mask)[0]
    col_indices_kept = np.where(col_mask)[0]

    # Labels of the rows/cols that were kept
    filtered_row_labels = [original_row_labels[i] for i in row_indices_kept]
    filtered_col_labels = [original_col_labels[i] for i in col_indices_kept]

    # Check if matrix would be empty after filtering
    if not np.any(row_mask) or not np.any(col_mask):
         # Return empty DataFrame but original labels for context if needed
        return pd.DataFrame(), [], [], original_row_labels, original_col_labels


    # Filter matrix using boolean masks on DataFrame
    filtered_matrix_df = matrix.loc[row_mask, col_mask]

    return filtered_matrix_df, row_indices_kept, col_indices_kept, filtered_row_labels, filtered_col_labels

# <<< MODIFIED create_flow_diagram >>>
def create_flow_diagram(matrix_in, # Should be DataFrame
                        color_map, # Pass the main color map
                        alpha=0.7,
                        interp_frac=0.0, # Defaulting back to source color
                        v_space=1.0,
                        ax=None,
                        default_color='#808080'):
    """
    Creates a flow diagram from a matrix (DataFrame). Uses color_map for coloring.

    Args:
        matrix_in (pd.DataFrame): An n x m DataFrame of float values with labels.
        color_map (dict): Dictionary mapping community labels to color strings.
        alpha (float): Opacity for flows.
        interp_frac (float): Interpolation fraction for flow colors (0=source, 1=dest).
        v_space (float): Vertical spacing factor.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, uses current axes.
        default_color (str): Fallback color if a label is not in color_map.

    Returns:
        tuple: (matplotlib.axes.Axes, dict)
               The axes containing the flow diagram and a dictionary containing layout info (optional).
               Returns (ax, {}) for now.
    """
    if ax is None:
        ax = plt.gca()

    # Preprocess: remove zero rows/cols, get filtered data and labels
    try:
        matrix_df, row_ind, col_ind, filtered_row_labels, filtered_col_labels = preprocess_matrix(matrix_in)
        if matrix_df.empty:
            # Handle case where matrix becomes empty after filtering
            print("Warning: Matrix is empty or all zero after preprocessing. Skipping plot.")
            ax.axis('off')
            return ax, {} # Return empty info
    except ValueError as e:
         print(f"Error during preprocessing: {e}. Skipping plot.")
         ax.axis('off')
         return ax, {}

    matrix = matrix_df.to_numpy() # Work with numpy array for calculations
    n, m = matrix.shape

    # --- Color Setup using color_map and filtered labels ---
    colors_in_list = [color_map.get(label, default_color) for label in filtered_row_labels]
    colors_out_list = [color_map.get(label, default_color) for label in filtered_col_labels]

    colors_in = np.array([to_rgba(c) for c in colors_in_list])
    colors_out = np.array([to_rgba(c) for c in colors_out_list])
    if colors_in.size > 0: colors_in[:, 3] = alpha
    if colors_out.size > 0: colors_out[:, 3] = alpha

    # Create colors_out_mat as DataFrame with proper labels
    # colors_out_mat = pd.DataFrame(
    #     np.zeros((n, m)),
    #     index=filtered_row_labels,
    #     columns=filtered_col_labels
    # )
    # for i, row_label in enumerate(filtered_row_labels):
    #     for j, col_label in enumerate(filtered_col_labels):
    #         colors_out_mat.loc[row_label, col_label] = interp_frac * colors_out[j] + (1 - interp_frac) * colors_in[i]
    # --- End Color Setup ---


    # Configure layout
    h_gap = 2.0
    # Adjust margin calculation to prevent division by zero if n or m is 1
    v_margin_left = v_space * 0.2 / (n) if n > 1 else (0.1 * v_space if n == 1 else 0) # Scaled margin based on count
    v_margin_right = v_space * 0.2 / (m) if m > 1 else (0.1 * v_space if m == 1 else 0)
    left_x = 0

    # Normalize matrix and calculate positions
    total = matrix.sum()
    if total <= 1e-9: # Avoid division by zero if total flow is negligible
        print("Warning: Total flow in matrix is near zero. Skipping plot.")
        ax.axis('off')
        return ax, {}

    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    # Calculate overall heights required (including margins)
    total_height_left = (row_sums / total).sum() + max(0, n - 1) * v_margin_left
    total_height_right = (col_sums / total).sum() + max(0, m - 1) * v_margin_right
    max_total_height = max(total_height_left, total_height_right, 1e-6) # Ensure non-zero max height

    # Calculate source and destination block centers (y-coordinates)
    src_positions = np.zeros(n)
    dst_positions = np.zeros(m)
    current_y_left = 0
    for i in range(n):
        height = row_sums[i] / total
        src_positions[i] = current_y_left + height / 2
        current_y_left += height + v_margin_left

    current_y_right = 0
    for j in range(m):
        height = col_sums[j] / total
        dst_positions[j] = current_y_right + height / 2
        current_y_right += height + v_margin_right

    # --- Store layout info (optional, can be expanded) ---
    layout_info = {
        'src_positions': {label: pos for label, pos in zip(filtered_row_labels, src_positions)},
        'dst_positions': {label: pos for label, pos in zip(filtered_col_labels, dst_positions)},
        'src_heights': {label: row_sums[i] / total for i, label in enumerate(filtered_row_labels)},
        'dst_heights': {label: col_sums[j] / total for j, label in enumerate(filtered_col_labels)},
        'total_height_left': current_y_left - (v_margin_left if n > 0 else 0), # Actual top edge
        'total_height_right': current_y_right - (v_margin_right if m > 0 else 0), # Actual top edge
        # 'colors_out_matrix': colors_out_mat
    }
    # print(layout_info['colors_out_matrix'])
    # print('testtt')
    # --- End Layout Info ---


    # Create flows
    src_offsets = np.zeros(n) # Tracks vertical offset within the source block
    dst_offsets = np.zeros(m) # Tracks vertical offset within the destination block

    for i, row_label in enumerate(filtered_row_labels):
        for j, col_label in enumerate(filtered_col_labels):
            if matrix[i, j] > 1e-9: # Check for non-negligible flow
                flow_width = matrix[i, j] / total
                # Calculate y-coords relative to the bottom of the block
                src_block_bottom = src_positions[i] - (row_sums[i] / (2 * total))
                dst_block_bottom = dst_positions[j] - (col_sums[j] / (2 * total))

                start_y = src_block_bottom + src_offsets[i] + flow_width / 2
                end_y = dst_block_bottom + dst_offsets[j] + flow_width / 2

                poly_coords = get_flow_polygon((left_x, start_y), (left_x + h_gap, end_y), flow_width, h_gap=h_gap)

                # Use colors from DataFrame using labels
                add_gradient_patch(
                    polygon=poly_coords,
                    start=(left_x, start_y),
                    end=(left_x + h_gap, end_y),
                    color1=colors_in[i],
                    color2=colors_in[i],
                    # color2=colors_out_mat[row_label, col_label],
                    ax=ax,
                    edgecolor='grey'
                )

                src_offsets[i] += flow_width
                dst_offsets[j] += flow_width

    # Set view limits based on calculated total heights
    ax.set_xlim(0.0, left_x + h_gap + 0) # Add small padding
    # Use the actual max y reached, considering offsets
    max_y_lim = max(layout_info['total_height_left'], layout_info['total_height_right'], 1e-6)
    ax.set_ylim(0, max_y_lim)
    ax.axis('off')

    return ax, layout_info # Return axes and layout info dictionary

# Example usage (remains the same)
if __name__ == "__main__":
    n = np.random.randint(3, 7)
    m = n + np.random.randint(-2, 3)
    matrix_np = np.random.rand(n, m) * 10
    threshold = 1.0
    for i in range(n):
        for j in range(m):
            matrix_np[i, j] *= 0.8**abs(i - j)
            if matrix_np[i, j] < threshold:
                matrix_np[i, j] = 0.0

    # Create dummy labels and colormap
    row_labels = [f'Src_{k}' for k in range(n)]
    col_labels = [f'Dst_{k}' for k in range(m)]
    matrix_df = pd.DataFrame(matrix_np, index=row_labels, columns=col_labels)
    all_labels = sorted(list(set(row_labels) | set(col_labels)))
    cmap = {label: plt.cm.tab20(i / len(all_labels)) for i, label in enumerate(all_labels)}


    fig, ax = plt.subplots(figsize=(6, 8))
    create_flow_diagram(matrix_df, color_map=cmap, interp_frac=0.5, v_space=1.0, ax=ax)
    plt.title("Example Flow Diagram")
    plt.tight_layout()
    plt.show()