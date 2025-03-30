import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba # Assuming this is used by create_flow_diagram context

# --- Data Generation Functions ---

def _get_all_communities(df, time_cols):
    """Gets sorted list of all unique communities across specified time columns."""
    all_communities = set()
    for col in time_cols:
        # Handle potential NaN values if necessary
        all_communities.update(df[col].dropna().unique())
    return sorted(list(all_communities))

def _create_transition_df(df, start_col, end_col, all_communities):
    """
    Creates a transition matrix (DataFrame) between two time points.
    Rows are source communities, columns are target communities.
    Ensures all communities are present in index/columns.
    """
    # Use community values directly as labels
    matrix = pd.DataFrame(0.0, index=all_communities, columns=all_communities)

    # Calculate transition counts, handle potential NaNs by dropping them during grouping
    counts = df.dropna(subset=[start_col, end_col])\
               .groupby([start_col, end_col])\
               .size()\
               .unstack(fill_value=0)

    # Add counts to the template matrix, aligning by index/columns
    matrix = matrix.add(counts, fill_value=0)
    return matrix

def _create_identity_df(df, time_col, all_communities):
    """
    Creates a diagonal matrix (DataFrame) representing community distribution
    at a single time point. diag(comm) = count(comm).
    """
    matrix = pd.DataFrame(0.0, index=all_communities, columns=all_communities)
    counts = df[time_col].value_counts()
    for comm, count in counts.items():
        if comm in matrix.index: # Ensure community is in the full list
            matrix.loc[comm, comm] = float(count)
    return matrix

def generate_alluvial_matrices(df, time_cols):
    """
    Generates direct, intermediate, and identity transition matrices (DataFrames)
    for alluvial plots over multiple time points.

    Parameters:
    df (pd.DataFrame): DataFrame with community classifications per time point.
    time_cols (list): List of column names representing timepoints (>= 2).

    Returns:
    tuple: (direct_trans_matrices, inter_trans_matrices, identity_matrices)
        - direct: List of DataFrames for T_i -> T_{i+1}.
        - inter: List of lists. Each inner list corresponds to transitions
                 T_i -> T_{i+2} via T_{i+1}, containing tuples of
                 (intermediate_community, intermediate_matrix_df).
        - identity: List containing two tuples:
                    ('start', identity_df_T0) and ('end', identity_df_Tn).
    """
    if len(time_cols) < 2:
        raise ValueError("Requires at least 2 timepoints")

    all_communities = _get_all_communities(df, time_cols)
    n_timepoints = len(time_cols)
    direct_trans_matrices = []
    inter_trans_matrices = []
    identity_matrices = []

    # --- Direct Transitions (T_i -> T_{i+1}) ---
    for i in range(n_timepoints - 1):
        matrix = _create_transition_df(df, time_cols[i], time_cols[i+1], all_communities)
        direct_trans_matrices.append(matrix)

    # --- Intermediate Transitions (T_i -> T_{i+2} via T_{i+1}) ---
    for i in range(n_timepoints - 2):
        t_start = time_cols[i]
        t_intermediate = time_cols[i+1]
        t_end = time_cols[i+2]

        intermediate_community_matrices = []
        # Ensure intermediate communities are sorted for consistent plotting order
        intermediate_communities = sorted(df[t_intermediate].dropna().unique())

        for inter_comm in intermediate_communities:
            intermediate_nodes_df = df[df[t_intermediate] == inter_comm]
            if intermediate_nodes_df.empty:
                continue

            matrix = _create_transition_df(intermediate_nodes_df, t_start, t_end, all_communities)

            # Only add if there's actual flow in the intermediate step
            if matrix.sum().sum() > 0:
                 intermediate_community_matrices.append((int(inter_comm), matrix))

        # Append the list for this T_i -> T_{i+2} step
        inter_trans_matrices.append(intermediate_community_matrices)

    # --- Identity Matrices (T0 and Tn) ---
    if n_timepoints >= 1:
        start_identity_df = _create_identity_df(df, time_cols[0], all_communities)
        identity_matrices.append(('start', start_identity_df))

        end_identity_df = _create_identity_df(df, time_cols[-1], all_communities)
        identity_matrices.append(('end', end_identity_df))

    return direct_trans_matrices, inter_trans_matrices, identity_matrices
