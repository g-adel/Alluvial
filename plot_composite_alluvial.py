# plot_composite_alluvial.py (Refactored)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
# Assuming my_alluvial contains create_flow_diagram
import importlib
import my_alluvial 
importlib.reload(my_alluvial)

from my_alluvial import create_flow_diagram
import seaborn as sns
import pandas as pd
import numpy as np

# --- Helper Functions ---

def _sort_communities(communities_set):
    """Sorts a set of communities, handling numeric and non-numeric types."""
    if not communities_set: return []
    try:
        numeric_items = []
        non_numeric_items = []
        for item in communities_set:
            try:
                numeric_items.append(float(item)) # Store as float for sorting
            except (ValueError, TypeError):
                non_numeric_items.append(str(item)) # Store as string

        # Sort numerics as numbers, non-numerics as strings
        # Convert numeric back to original type if needed (or keep as float/int)
        # For simplicity, we'll return sorted numerics (as floats) and sorted strings
        # If original types (e.g., int vs float) matter, more complex handling is needed
        sorted_numeric = sorted(numeric_items)
        # Attempt to convert back to int if they were originally int
        sorted_numeric_typed = [int(x) if x.is_integer() else x for x in sorted_numeric]

        sorted_non_numeric = sorted(non_numeric_items)
        # Combine, ensuring consistent types if mixed (e.g., all strings)
        # Or keep them separate if types should be preserved (current approach)
        return sorted_numeric_typed + sorted_non_numeric
    except Exception:
        # Fallback to simple string sorting
        return sorted(list(communities_set), key=str)

def _get_all_communities_from_matrices(direct_mats, inter_mats):
    """Gets sorted list of all unique communities from matrix indices/columns."""
    all_communities_set = set()

    for mat in direct_mats:
        if isinstance(mat, pd.DataFrame):
            all_communities_set.update(mat.index.unique())
            all_communities_set.update(mat.columns.unique())

    for step_list in inter_mats:
        for _, matrix in step_list:
             if isinstance(matrix, pd.DataFrame):
                all_communities_set.update(matrix.index.unique())
                all_communities_set.update(matrix.columns.unique())

    return _sort_communities(all_communities_set)


def _setup_gridspec(n_direct, n_inter, n_identity, direct_width, inter_width, identity_width, gap_width, figsize):
    """Sets up the main GridSpec and width ratios for the composite plot."""
    plot_count = n_identity + n_direct + n_inter
    if plot_count <= 0: return None, None, {}
    gap_count = max(0, plot_count - 1)
    ncols_actual = plot_count + gap_count
    if ncols_actual <= 0: return None, None, {}

    width_ratios = []
    plot_indices = {'identity': [], 'direct': [], 'inter': []}
    current_gs_col = 0

    def add_element(width, plot_type=None, index_list=None):
        nonlocal current_gs_col, width_ratios
        width_ratios.append(width)
        if plot_type is not None and index_list is not None:
             index_list.append(current_gs_col)
        current_gs_col += 1
        # Add gap only if not the last element
        if current_gs_col < ncols_actual:
            width_ratios.append(gap_width)
            current_gs_col += 1

    # Add start identity
    if n_identity > 0:
        add_element(identity_width, 'identity', plot_indices['identity'])

    # Add direct and intermediate plots
    for i in range(n_direct):
        add_element(direct_width, 'direct', plot_indices['direct'])
        if i < n_inter:
            add_element(inter_width, 'inter', plot_indices['inter'])

    # Add end identity if needed and space allows
    if n_identity > 1:
        # Check if the end identity slot is already implicitly the last element
        if len(width_ratios) < ncols_actual :
             add_element(identity_width, 'identity', plot_indices['identity'])
        elif len(width_ratios) == ncols_actual:
            # If the last element is a gap, replace it with the identity plot
             if len(plot_indices['direct']) + len(plot_indices['inter']) < plot_count -1: # Check if really expecting a gap
                 width_ratios[-1] = identity_width
                 current_gs_col -= 1 # Reclaim the column index
                 plot_indices['identity'].append(current_gs_col)
            # If last element is already a plot, assume it's the intended end identity? (Shouldn't happen with loop logic)

    # Final check and adjustment
    if len(width_ratios) != ncols_actual:
       print(f"Warning: GridSpec setup mismatch. Expected {ncols_actual} columns, calculated {len(width_ratios)}. Adjusting gridspec.")
       ncols_actual = len(width_ratios)
       if ncols_actual == 0: return None, None, {}

    fig = plt.figure(figsize=figsize)
    # Ensure width ratios are positive
    width_ratios_safe = [max(wr, 0.01) for wr in width_ratios]
    gs_main = gridspec.GridSpec(1, ncols_actual, width_ratios=width_ratios_safe, wspace=0, hspace=0)

    return fig, gs_main, plot_indices


def _get_color_map(colors, all_communities):
    """Creates a dictionary mapping community IDs to color strings."""
    if not all_communities: return {}
    if colors is None:
        # Generate colors if none provided
        palette = sns.color_palette("tab20", len(all_communities))
        return {comm: palette[i] for i, comm in enumerate(all_communities)}
    elif isinstance(colors, list):
        # Cycle through list of colors
        return {comm: colors[i % len(colors)] for i, comm in enumerate(all_communities)}
    elif isinstance(colors, dict):
        # Use provided dictionary, with fallback
        default_color = '#808080' # Grey for missing communities
        cmap = {}
        for comm in all_communities:
             # Try matching both original and string versions of keys
             cmap[comm] = colors.get(comm, colors.get(str(comm), default_color))
        return cmap
    else:
        raise TypeError("colors must be a list, dict, or None")


def _plot_intermediate_column(fig, subplot_spec, matrices_info, color_map, height_ratios, title, interp_frac, show_titles, show_labels, spacing=0.05):
    """Plots a vertical stack of flow diagrams for intermediate transitions."""
    if not matrices_info:
        ax = fig.add_subplot(subplot_spec); ax.axis('off'); return [ax]

    # Ensure height ratios are positive and sum to something reasonable if needed
    height_ratios_safe = [max(hr, 1e-9) for hr in height_ratios]
    if sum(height_ratios_safe) < 1e-9 : height_ratios_safe = [1]*len(matrices_info) # Equal heights if all zero

    # Reverse order for plotting bottom-up if GridSpec requires it (depends on gs_nested creation)
    # Let's assume GridSpec handles top-down order correctly.
    matrices_info = matrices_info[::-1]
    height_ratios_safe = height_ratios_safe[::-1]

    gs_nested = gridspec.GridSpecFromSubplotSpec(
        len(matrices_info), 1,
        subplot_spec=subplot_spec,
        height_ratios=height_ratios_safe,
        hspace=0.3 # Vertical gap between subplots in the stack
    )

    axes_in_column = []
    for j, (label, matrix_df) in enumerate(matrices_info): # Iterate through communities
        ax = fig.add_subplot(gs_nested[j, 0])
        axes_in_column.append(ax)

        # Plot the individual flow for this intermediate community path
        _, _ = create_flow_diagram(
            matrix_df,
            color_map=color_map, # Use the global color map
            interp_frac=interp_frac,
            v_space=0, # Minimal vertical space within each small plot
            ax=ax
        )

        if show_labels:
            label_text = f"Via {label}"
            # Position label slightly above the small plot
            ax.text(0.5, 1.0 + spacing, label_text, ha='center', va='bottom',
                    transform=ax.transAxes, fontsize=8, clip_on=False, zorder=10)

    # Add overall column title
    if show_titles:
        try:
            col_pos = subplot_spec.get_position(fig)
            title_y_pos = col_pos.y1 + 0.01 # Position above the column
            title_y_pos = min(title_y_pos, 0.98) # Clamp to avoid going off-figure
            fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title,
                     ha='center', va='bottom', fontsize=10)
        except Exception as e:
            print(f"Error positioning title '{title}': {e}")

    return axes_in_column


def plot_composite_alluvial(direct_trans_matrices: list,
                            inter_trans_matrices: list,
                            colors=None,
                            figsize=(16, 8),
                            direct_plot_width_ratio=2.5,
                            inter_plot_width_ratio=1,
                            identity_plot_width_ratio=0.5,
                            gap_width_ratio=0.1,
                            interp_frac=0.0,
                            direct_vspace=1.5,
                            identity_vspace=1.5,
                            show_titles=False,
                            show_labels=False,
                            show_suptitle=False,
                            min_flow_for_ratio = 0.005
                           ):
    """
    Creates a composite alluvial diagram figure using GridSpec from transition matrices.

    Parameters:
    - direct_trans_matrices: List of DataFrames (T_i -> T_{i+1}).
    - inter_trans_matrices: List of lists of tuples [(intermediate_community, DataFrame)].
    - colors: Color mapping (dict, list, or None).
    - figsize: Figure size.
    - *_plot_width_ratio: Relative widths of plot columns.
    - gap_width_ratio: Relative width of gaps between columns.
    - interp_frac: Color interpolation fraction (0=source, 1=destination).
    - *_vspace: Vertical spacing factor within plot columns.
    - show_titles: Display titles above columns.
    - show_labels: Display "Via ..." labels for intermediate plots.
    - show_suptitle: Display overall figure title.
    - min_flow_for_ratio: Minimum relative flow threshold for showing an intermediate path.
    """

    # --- Input Validation and Setup ---
    n_direct = len(direct_trans_matrices)
    n_inter = len(inter_trans_matrices)
    n_identity = 2 if n_direct > 0 else 0 # Need start/end identity only if transitions exist

    if n_direct > 0 and n_direct <= 1 and n_inter != 0:
        print(f"Warning: Intermediate matrices provided but <= 1 direct matrices. Ignoring intermediates.")
        n_inter = 0
    elif n_direct > 1 and n_inter != n_direct - 1:
        print(f"Warning: Number of intermediate matrices ({n_inter}) != expected ({n_direct - 1}). Plotting may be misaligned.")
        # Adjust n_inter to match available data if desired, or proceed cautiously.
        n_inter = min(n_inter, n_direct -1) # Limit intermediates to available slots

    all_communities = _get_all_communities_from_matrices(direct_trans_matrices, inter_trans_matrices)
    if not all_communities:
        print("Warning: No communities found in matrices. Cannot plot.")
        return plt.figure(figsize=figsize), [] # Return empty figure

    color_map = _get_color_map(colors, all_communities)
    comm_order_map = {comm: idx for idx, comm in enumerate(all_communities)}

    fig, gs_main, plot_indices = _setup_gridspec(
        n_direct, n_inter, n_identity,
        direct_plot_width_ratio, inter_plot_width_ratio, identity_plot_width_ratio,
        gap_width_ratio, figsize
    )

    if fig is None:
        print("GridSpec setup failed.")
        return plt.figure(figsize=figsize), []
    all_axes = []

    # --- Generate Identity Matrices ---
    identity_matrix_start = pd.DataFrame(0.0, index=all_communities, columns=all_communities)
    identity_matrix_end = pd.DataFrame(0.0, index=all_communities, columns=all_communities)

    if n_direct > 0:
        # Start Identity: Sum of outgoing flows from the first direct matrix
        start_mat = direct_trans_matrices[0]
        if isinstance(start_mat, pd.DataFrame):
            start_flows = start_mat.sum(axis=1) # Sum rows (outgoing)
            for comm, count in start_flows.items():
                 if comm in identity_matrix_start.index:
                    identity_matrix_start.loc[comm, comm] = float(count)

        # End Identity: Sum of incoming flows to the last direct matrix
        end_mat = direct_trans_matrices[-1]
        if isinstance(end_mat, pd.DataFrame):
            end_flows = end_mat.sum(axis=0) # Sum columns (incoming)
            for comm, count in end_flows.items():
                if comm in identity_matrix_end.index:
                     identity_matrix_end.loc[comm, comm] = float(count)

    # --- Plot Identity Start Column ---
    if plot_indices.get('identity'):
        id_start_col_idx = plot_indices['identity'][0]
        id_start_subplot_spec = gs_main[0, id_start_col_idx]
        title_text = "Start Distribution"

        ax_id_start = fig.add_subplot(id_start_subplot_spec)
        all_axes.append(ax_id_start)
        try:
            _, _ = create_flow_diagram(
                identity_matrix_start,
                color_map=color_map,
                interp_frac=0.0, # Source color = dest color for identity
                v_space=identity_vspace,
                ax=ax_id_start
            )
            if show_titles:
                col_pos = id_start_subplot_spec.get_position(fig)
                title_y_pos = min(col_pos.y1 + 0.01, 0.98)
                fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)
        except Exception as e:
            print(f"Error plotting Start Identity: {e}")
            ax_id_start.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); ax_id_start.axis('off')

    # --- Plot Direct and Intermediate Columns ---
    for i in range(n_direct):
        target_comms_series = pd.Series(dtype=float) # Track flow arriving at T_{i+1}

        # --- Plot Direct Matrix T_i -> T_{i+1} ---
        if i < len(plot_indices.get('direct', [])):
            direct_col_idx = plot_indices['direct'][i]
            direct_matrix = direct_trans_matrices[i]
            title_text = f"Transition {i+1}"

            # Ensure it's a DataFrame before proceeding
            if not isinstance(direct_matrix, pd.DataFrame):
                 print(f"Warning: Direct matrix {i} is not a DataFrame. Skipping plot.")
                 # Add a blank axis placeholder
                 ax = fig.add_subplot(gs_main[0, direct_col_idx]); ax.axis('off'); all_axes.append(ax)
                 continue # Skip to next direct matrix or intermediate

            # Ensure matrix uses all known communities for consistency
            # This is crucial if generate_alluvial_matrices_from_edgelists didn't guarantee it
            direct_matrix = direct_matrix.reindex(index=all_communities, columns=all_communities, fill_value=0.0)

            direct_ax = fig.add_subplot(gs_main[0, direct_col_idx])
            all_axes.append(direct_ax)

            try:
                _, _ = create_flow_diagram(
                    direct_matrix,
                    color_map=color_map,
                    interp_frac=interp_frac,
                    v_space=direct_vspace,
                    ax=direct_ax
                )

                # Calculate flow distribution arriving at the *end* of this direct plot (T_{i+1})
                target_comms_series = direct_matrix.sum(axis=0) # Sum columns (incoming flow)

                if show_titles:
                     col_pos = gs_main[0, direct_col_idx].get_position(fig)
                     title_y_pos = min(col_pos.y1 + 0.01, 0.98)
                     fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)

            except Exception as e:
                print(f"Error plotting direct matrix {i} ({title_text}): {e}")
                direct_ax.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); direct_ax.axis('off')
                target_comms_series = pd.Series(dtype=float) # Ensure empty on error

        else:
            print(f"Warning: Skipping direct plot {i}, index out of bounds in plot layout.")


        # --- Plot Intermediate Column T_i -> T_{i+2} via T_{i+1} ---
        if i < n_inter: # Check if an intermediate plot should exist for this step
            if i < len(plot_indices.get('inter', [])): # Check if layout includes space
                inter_col_idx = plot_indices['inter'][i]
                inter_subplot_spec = gs_main[0, inter_col_idx]
                intermediate_set = inter_trans_matrices[i] # List of (comm, matrix_df)
                title_text = f"Transition {i+1} \u2192 {i+3}" # Unicode right arrow

                height_ratios = []
                valid_intermediate_set = []
                # Sort intermediate paths based on the global community order
                intermediate_set_sorted = sorted(intermediate_set, key=lambda x: comm_order_map.get(x[0], float('inf')))
                total_incoming_flow = target_comms_series.sum() # Total flow arriving at T_{i+1}

                for comm, matrix_df_in in intermediate_set_sorted:
                    # Ensure matrix_df_in is a DataFrame
                    if not isinstance(matrix_df_in, pd.DataFrame):
                        print(f"Warning: Intermediate matrix for community {comm} in step {i} is not a DataFrame. Skipping.")
                        continue
                    matrix_df = matrix_df_in

                    # Ensure matrix uses all known communities
                    matrix_df = matrix_df.reindex(index=all_communities, columns=all_communities, fill_value=0.0)

                    # Flow into the intermediate community (comm) at T_{i+1}
                    flow_into_comm = target_comms_series.get(comm, 0)
                    relative_flow_into = flow_into_comm / total_incoming_flow if total_incoming_flow > 1e-9 else 0

                    # Check if this path has significant flow *into* it and *through* it
                    if relative_flow_into >= min_flow_for_ratio and not matrix_df.empty and matrix_df.sum().sum() > 1e-9:
                        height_ratios.append(flow_into_comm) # Height based on flow *into* the intermediate node
                        valid_intermediate_set.append((comm, matrix_df))

                if valid_intermediate_set:
                    inter_axes = _plot_intermediate_column(fig, inter_subplot_spec, valid_intermediate_set, color_map, height_ratios, title_text, interp_frac, show_titles, show_labels, spacing=0.05)
                    all_axes.extend(inter_axes)
                else: # If no valid intermediate flows for this step, add dummy axis
                    ax = fig.add_subplot(inter_subplot_spec); ax.axis('off'); all_axes.append(ax)
                    if show_titles:
                         col_pos = inter_subplot_spec.get_position(fig)
                         title_y_pos = min(col_pos.y1 + 0.01, 0.98)
                         fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)
                    print(f"No valid intermediate flows >= {min_flow_for_ratio*100:.1f}% for {title_text}, column left blank.")
            else:
                print(f"Warning: Skipping intermediate plot {i}, index out of bounds in plot layout.")

    # --- Plot Identity End Column ---
    if len(plot_indices.get('identity', [])) > 1:
        id_end_col_idx = plot_indices['identity'][-1]
        # Ensure this is actually the end identity index (different from start if n_identity=2)
        if len(plot_indices['identity']) == 1 or id_end_col_idx != plot_indices['identity'][0]:
            id_end_subplot_spec = gs_main[0, id_end_col_idx]
            title_text = "End Distribution"

            ax_id_end = fig.add_subplot(id_end_subplot_spec)
            all_axes.append(ax_id_end)
            try:
                 _, _ = create_flow_diagram(
                     identity_matrix_end,
                     color_map=color_map,
                     interp_frac=0.0,
                     v_space=identity_vspace,
                     ax=ax_id_end
                 )
                 if show_titles:
                      col_pos = id_end_subplot_spec.get_position(fig)
                      title_y_pos = min(col_pos.y1 + 0.01, 0.98)
                      fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)

            except Exception as e:
                print(f"Error plotting End Identity: {e}")
                ax_id_end.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); ax_id_end.axis('off')

    elif n_identity > 0 and len(plot_indices.get('identity',[])) <= 1 :
         # This case should only happen if n_identity=1 was forced, or layout failed
         print("Skipping Identity End Plot (only one identity plot specified or layout issue).")


    # --- Final Adjustments ---
    if show_suptitle and show_titles:
        fig.suptitle("Composite Alluvial Flow Diagram", fontsize=16, y=0.99)

    # Adjust layout to prevent overlap and use space efficiently
    top_margin = 0.92 if show_titles or show_suptitle else 0.98
    bottom_margin = 0.02
    left_margin = 0.02
    right_margin = 0.98
    try:
        # Using subplots_adjust provides fine control
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, wspace=0, hspace=0)
        # Alternatively, tight_layout can sometimes work but offers less control:
        # plt.tight_layout(rect=[left_margin, bottom_margin, right_margin, top_margin])
    except Exception as e_adj:
        print(f"Warning: Could not apply final layout adjustments: {e_adj}")

    return fig, all_axes

# --- Example Usage Placeholder ---
# You would call this function like this:
# fig, axes = plot_composite_alluvial(
#     direct_mats,  # Output from generate_alluvial_matrices_from_edgelists
#     inter_mats,   # Output from generate_alluvial_matrices_from_edgelists
#     colors=my_color_dict, # Optional
#     show_titles=True,
#     show_labels=True
# )
# plt.show()