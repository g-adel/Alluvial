# plot_composite_alluvial.py (Updated)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import my_alluvial # Import the updated module
from my_alluvial import create_flow_diagram
import seaborn as sns
import pandas as pd
import numpy as np

# _get_all_communities, _setup_gridspec, _get_color_map remain the same as your last version

def _get_all_communities(df, time_cols):
    """Gets sorted list of all unique communities across specified time columns."""
    all_communities = set()
    for col in time_cols:
        all_communities.update(df[col].dropna().unique())
    if not all_communities: return []
    try:
        numeric_items = []
        non_numeric_items = []
        for item in all_communities:
            try: float(item); numeric_items.append(item)
            except (ValueError, TypeError): non_numeric_items.append(item)
        sorted_numeric = sorted(numeric_items, key=float)
        sorted_non_numeric = sorted(non_numeric_items, key=str)
        return sorted_numeric + sorted_non_numeric
    except Exception: return sorted(list(all_communities), key=str)

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
        if plot_type and index_list is not None: index_list.append(current_gs_col)
        current_gs_col += 1
        if current_gs_col < ncols_actual: width_ratios.append(gap_width); current_gs_col += 1
    if n_identity > 0: add_element(identity_width, 'identity', plot_indices['identity'])
    for i in range(n_direct):
        add_element(direct_width, 'direct', plot_indices['direct'])
        if i < n_inter: add_element(inter_width, 'inter', plot_indices['inter'])
    if n_identity > 1:
        if not plot_indices['identity'] or len(plot_indices['identity'])==1: # If only start added or none
             if len(width_ratios) == ncols_actual and width_ratios[-1] == gap_width: # Ends with gap
                 width_ratios[-1] = identity_width; current_gs_col -=1
                 plot_indices['identity'].append(current_gs_col)
             elif len(width_ratios) < ncols_actual: # Ends with plot
                 width_ratios.append(identity_width); plot_indices['identity'].append(current_gs_col); current_gs_col += 1
    if len(width_ratios) != ncols_actual:
       print(f"Warning: GridSpec setup mismatch. Expected {ncols_actual}, got {len(width_ratios)}. Adjusting."); ncols_actual = len(width_ratios)
       if ncols_actual == 0: return None, None, {}
    fig = plt.figure(figsize=figsize)
    width_ratios_safe = [max(wr, 0.01) for wr in width_ratios]
    gs_main = gridspec.GridSpec(1, ncols_actual, width_ratios=width_ratios_safe, wspace=0, hspace=0)
    return fig, gs_main, plot_indices

def _get_color_map(colors, all_communities):
    """Creates a dictionary mapping community IDs to color strings."""
    if not all_communities: return {}
    if colors is None:
        palette = sns.color_palette("tab20", len(all_communities))
        return {comm: palette[i] for i, comm in enumerate(all_communities)}
    elif isinstance(colors, list):
        return {comm: colors[i % len(colors)] for i, comm in enumerate(all_communities)}
    elif isinstance(colors, dict):
        default_color = '#808080'
        # Try matching both original and string versions of keys
        cmap = {}
        for comm in all_communities:
             cmap[comm] = colors.get(comm, colors.get(str(comm), default_color))
        return cmap
    else: raise TypeError("colors must be a list or dict or None")


# <<< REMOVED _plot_identity_column >>>

# <<< HELPER for Identity Matrix >>>
def _create_identity_matrix(df, time_col, all_communities):
    """Creates a diagonal DataFrame for identity plots based on counts."""
    counts = df[time_col].value_counts()
    # Create an empty DataFrame with all communities as index and columns
    identity_matrix = pd.DataFrame(0.0, index=all_communities, columns=all_communities)
    # Fill diagonal with counts (or proportions if desired)
    for comm in all_communities:
        identity_matrix.loc[comm, comm] = counts.get(comm, 0)
    return identity_matrix


# <<< MODIFIED _plot_intermediate_column >>>
def _plot_intermediate_column(fig, subplot_spec, matrices_info, color_map, height_ratios, title, interp_frac, show_titles, show_labels, spacing=0.4):
    """Plots a vertical stack of flow diagrams for intermediate transitions."""
    if not matrices_info:
        ax = fig.add_subplot(subplot_spec); ax.axis('off'); return [ax]

    height_ratios_safe = [max(hr, 1e-9) for hr in height_ratios]
    matrices_info = matrices_info[::-1] # Reverse for bottom-up plotting if desired by gridspec
    height_ratios_safe = height_ratios_safe[::-1]

    try:
        gs_nested = gridspec.GridSpecFromSubplotSpec(
            len(matrices_info), 1,
            subplot_spec=subplot_spec,
            height_ratios=height_ratios_safe, # Use reversed heights
            hspace=0.3 # Vertical gap between subplots in the stack
        )
    except ValueError as e:
        print(f"Error creating nested GridSpec for intermediate column '{title}': {e}")
        ax = fig.add_subplot(subplot_spec); ax.axis('off')
        ax.text(0.5, 0.5, "Layout Error", color='red', ha='center', va='center'); return [ax]

    axes_in_column = []
    for j, (label, matrix,layout_info) in enumerate(matrices_info): # Iterate reversed
        ax = fig.add_subplot(gs_nested[j, 0])
        axes_in_column.append(ax)

        # Ensure matrix is DataFrame
        matrix_df = pd.DataFrame(matrix)

        # try:
        # <<< PASS color_map to create_flow_diagram >>>
        # print(layout_info['colors_out_matrix'][label])
        _, _ = create_flow_diagram(
            matrix_df,
            # color_map=layout_info['colors_out_matrix'][label], # Pass the main color map
            color_map=color_map,
            # No need to pass colors_in/out lists anymore
            interp_frac=interp_frac, # Use the value passed to intermediate (was hardcoded to 1)
            v_space=0, # No internal v_space within each small plot
            ax=ax
        )
        # except Exception as e: # Catch potential errors from create_flow_diagram too
        #     print(f"Error plotting intermediate matrix for label '{label}': {e}")
        #     # Optionally print matrix_df.info() or shape for debugging
        #     ax.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center')
        #     ax.axis('off')

        if show_labels:
            label_text = f"Via {label}"
            # Adjust label position if needed based on hspace
            ax.text(0.5, 1.0 + spacing*0.1, label_text, ha='center', va='bottom', # Position slightly above
                    transform=ax.transAxes, fontsize=8, clip_on=False, zorder=10)

    # Add overall column title (remains the same)
    if show_titles:
        try:
            col_pos = subplot_spec.get_position(fig)
            title_y_pos = col_pos.y1 + 0.01
            if title_y_pos > 0.98: title_y_pos = 0.98
            fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title,
                     ha='center', va='bottom', fontsize=10)
        except Exception as e: print(f"Error positioning title '{title}': {e}")

    # Return axes in the original order (optional, might not matter)
    return axes_in_column[::-1] # Reverse back if needed


# <<< MODIFIED Main plotting function >>>
def plot_composite_alluvial(df: pd.DataFrame,
                            time_cols: list,
                            direct_trans_matrices: list,
                            inter_trans_matrices: list,
                            colors=None,
                            figsize=(16, 8),
                            direct_plot_width_ratio=2.5,
                            inter_plot_width_ratio=1,
                            identity_plot_width_ratio=0.5,
                            gap_width_ratio=0.1,
                            interp_frac=0.0, # Default to source color
                            direct_vspace=1.5,
                            identity_vspace=1.5, # Added vspace for identity plots
                            show_titles=False,
                            show_labels=False,
                            show_suptitle=False,
                            min_flow_for_ratio = 0.005
                           ):
    """ Creates a composite alluvial diagram figure using GridSpec. """

    # --- Input Validation and Setup (mostly same) ---
    if not time_cols: raise ValueError("time_cols cannot be empty.")
    n_direct = len(direct_trans_matrices)
    n_inter = len(inter_trans_matrices)
    n_identity = 2 # Always aim for 2 identity plots
    if n_direct == 0 and n_inter == 0: n_identity = 0 # No plots if no transitions

    if n_direct > 0 and len(time_cols) != n_direct + 1:
         raise ValueError(f"Length of time_cols ({len(time_cols)}) must be n_direct ({n_direct}) + 1")
    if n_direct > 1 and n_inter != n_direct - 1:
        print(f"Warning: Number of intermediate matrices ({n_inter}) != expected ({n_direct - 1}).")
    elif n_direct <= 1 and n_inter != 0:
         print(f"Warning: Intermediate matrices provided but <= 1 direct matrices. Ignoring intermediates."); n_inter = 0

    all_communities = _get_all_communities(df, time_cols)
    if not all_communities: print("Warning: No communities found in df."); # Proceed cautiously

    color_map = _get_color_map(colors, all_communities)
    comm_order_map = {comm: idx for idx, comm in enumerate(all_communities)}

    fig, gs_main, plot_indices = _setup_gridspec(
        n_direct, n_inter, n_identity,
        direct_plot_width_ratio, inter_plot_width_ratio, identity_plot_width_ratio,
        gap_width_ratio, figsize
    )

    if fig is None: print("GridSpec setup failed."); return plt.figure(figsize=figsize), []
    all_axes = []

    # --- Plot Identity Start Column (using create_flow_diagram) ---
    if plot_indices.get('identity'):
        id_start_col_idx = plot_indices['identity'][0]
        id_start_subplot_spec = gs_main[0, id_start_col_idx]
        start_time_col = time_cols[0]
        title_text = f"{start_time_col} Dist."

        # Create diagonal matrix
        identity_matrix_start = _create_identity_matrix(df, start_time_col, all_communities)

        ax_id_start = fig.add_subplot(id_start_subplot_spec)
        all_axes.append(ax_id_start)
        # try:
        _, _ = create_flow_diagram(
            identity_matrix_start,
            color_map=color_map,
            interp_frac=0.0, # Source color = dest color for identity
            v_space=identity_vspace, # Use dedicated vspace
            ax=ax_id_start
        )
        if show_titles:
                # Position title for identity column
                col_pos = id_start_subplot_spec.get_position(fig)
                title_y_pos = col_pos.y1 + 0.01; title_y_pos = min(title_y_pos, 0.98)
                fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)

        # except Exception as e:
        #     print(f"Error plotting Identity Start column: {e}")
        #     ax_id_start.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); ax_id_start.axis('off')

    else: print("Skipping Identity Start Plot (not in layout).")

    # --- Plot Direct and Intermediate Columns (Main Loop) ---
    for i in range(n_direct):
        target_comms_series = pd.Series(dtype=float) # Reset flow tracker
        direct_ax = None

        # --- Plot Direct Matrix T_i -> T_{i+1} ---
        if i < len(plot_indices.get('direct', [])):
            direct_col_idx = plot_indices['direct'][i]
            direct_matrix_in = direct_trans_matrices[i]
            if not isinstance(direct_matrix_in, pd.DataFrame):
                try: direct_matrix = pd.DataFrame(direct_matrix_in); print(f"Warning: Direct matrix {i} converted.")
                except Exception as e: print(f"Error converting direct matrix {i}: {e}. Skipping."); continue
            else: direct_matrix = direct_matrix_in

            # Ensure matrix has labels matching all_communities if possible (BEST DONE UPSTREAM)
            # Example: direct_matrix = direct_matrix.reindex(index=all_communities, columns=all_communities, fill_value=0.0)

            direct_ax = fig.add_subplot(gs_main[0, direct_col_idx])
            all_axes.append(direct_ax)
            title_text = f"{time_cols[i]} → {time_cols[i+1]}"

            try:
                # <<< PASS color_map to create_flow_diagram >>>
                _, layout_info = create_flow_diagram(
                    direct_matrix,
                    color_map=color_map, # Pass the main color map
                    interp_frac=interp_frac,
                    v_space=direct_vspace,
                    ax=direct_ax
                )

                # Get flow distribution arriving at T_i+1 from the matrix *before* preprocessing
                target_comms_series = direct_matrix.sum(axis=0)

                if show_titles:
                     col_pos = gs_main[0, direct_col_idx].get_position(fig)
                     title_y_pos = col_pos.y1 + 0.01; title_y_pos = min(title_y_pos, 0.98)
                     fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)

            except Exception as e:
                print(f"Error plotting direct matrix {i} ({title_text}): {e}")
                direct_ax.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); direct_ax.axis('off')
                target_comms_series = pd.Series(dtype=float) # Ensure empty on error

        else: print(f"Warning: Skipping direct plot {i}, index out of bounds.")


        # --- Plot Intermediate Column T_i -> T_{i+2} via T_{i+1} ---
        if i < n_inter:
            if i < len(plot_indices.get('inter', [])):
                inter_col_idx = plot_indices['inter'][i]
                inter_subplot_spec = gs_main[0, inter_col_idx]
                intermediate_set = inter_trans_matrices[i]
                title_text = f"{time_cols[i]} → {time_cols[i+2]}"

                height_ratios = []
                valid_intermediate_set = []
                intermediate_set_sorted = sorted(intermediate_set, key=lambda x: comm_order_map.get(x[0], float('inf')))
                total_incoming_flow = target_comms_series.sum()

                for comm, matrix_df_in in intermediate_set_sorted:
                    matrix_df = pd.DataFrame(matrix_df_in) # Ensure DataFrame
                    flow_into_comm = target_comms_series.get(comm, 0)
                    relative_flow_into = flow_into_comm / total_incoming_flow if total_incoming_flow > 1e-9 else 0

                    if relative_flow_into >= min_flow_for_ratio and not matrix_df.empty and matrix_df.sum().sum() > 1e-9:
                        # Ensure matrix columns/index are usable (align if needed)
                        # Example: matrix_df = matrix_df.reindex(index=..., columns=..., fill_value=0)
                        height_ratios.append(flow_into_comm)
                        valid_intermediate_set.append((comm, matrix_df,layout_info))

                if valid_intermediate_set:
                    if sum(height_ratios) < 1e-9: height_ratios = [1] * len(valid_intermediate_set)
                    # <<< Pass spacing=0.0 to _plot_intermediate_column if needed >>>
                    inter_axes = _plot_intermediate_column(fig, inter_subplot_spec, valid_intermediate_set, color_map, height_ratios, title_text, interp_frac, show_titles, show_labels, spacing=0.05) # Small spacing between stacked
                    all_axes.extend(inter_axes)

                    # --- Experimental: Adjust Y limits to match Direct plot ---
                    # This is tricky because heights are relative within GridSpec
                    # if direct_ax and inter_axes:
                    #     try:
                    #         direct_ylim = direct_ax.get_ylim()
                    #         # Find overall min/max y across all inter_axes in data coords? Complex.
                    #         # Simpler: set all intermediate plots to match direct plot's y range?
                    #         # Might distort individual intermediate plots.
                    #         # print(f"Direct YLim: {direct_ylim}")
                    #         # for ax_inter in inter_axes:
                    #         #     ax_inter.set_ylim(direct_ylim)
                    #     except Exception as e_ylim:
                    #         print(f"Warning: Could not align Y limits: {e_ylim}")
                    # --- End Experimental ---

                else: # If no valid flows, add dummy axis
                    ax = fig.add_subplot(inter_subplot_spec); ax.axis('off'); all_axes.append(ax)
                    if show_titles:
                         col_pos = inter_subplot_spec.get_position(fig)
                         title_y_pos = col_pos.y1 + 0.01; title_y_pos = min(title_y_pos, 0.98)
                         fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)
                    print(f"No valid intermediate flows for {title_text}, column left blank.")
            else: print(f"Warning: Skipping intermediate plot {i}, index out of bounds.")


    # --- Plot Identity End Column (using create_flow_diagram) ---
    if len(plot_indices.get('identity', [])) > 1:
        id_end_col_idx = plot_indices['identity'][-1]
        # Check if it's different from start index if only 2 identity plots total
        if len(plot_indices['identity'])==1 or id_end_col_idx != plot_indices['identity'][0]:
            id_end_subplot_spec = gs_main[0, id_end_col_idx]
            end_time_col = time_cols[-1]
            title_text = f"{end_time_col} Dist."

            identity_matrix_end = _create_identity_matrix(df, end_time_col, all_communities)

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
                      title_y_pos = col_pos.y1 + 0.01; title_y_pos = min(title_y_pos, 0.98)
                      fig.text(col_pos.x0 + col_pos.width / 2, title_y_pos, title_text, ha='center', va='bottom', fontsize=10)

            except Exception as e:
                print(f"Error plotting Identity End column: {e}")
                ax_id_end.text(0.5, 0.5, "Plot Error", color='red', ha='center', va='center'); ax_id_end.axis('off')

    elif n_identity > 0 and len(plot_indices.get('identity',[])) <= 1 : # Only start plotted or none plotted
         print("Skipping Identity End Plot (not required or not in layout).")


    # --- Final Adjustments (using your latest values) ---
    if show_suptitle and show_titles:
        fig.suptitle("Composite Alluvial Flow Diagram", fontsize=16, y=0.99)

    top_margin = 0.92 if show_titles else 0.98 # Adjust based on title presence
    bottom_margin = 0.02
    left_margin = 0.02
    right_margin = 0.98
    try:
        # Use tight_layout first? Or just subplots_adjust?
        # plt.tight_layout(rect=[left_margin, bottom_margin, right_margin, top_margin]) # Alternative
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin, left=left_margin, right=right_margin, wspace=0, hspace=0)
    except Exception as e_adj: # Catch potential errors during layout adjustment
        print(f"Warning: Could not apply final layout adjustments: {e_adj}")

    return fig, all_axes