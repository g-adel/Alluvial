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

    return direct_trans_matrices, inter_trans_matrices

def _get_communities_for_timepoint(df, year_col, comm_col, year):
    """Gets sorted list of unique communities present at a specific timepoint."""
    communities = df.loc[df[year_col] == year, comm_col].dropna().unique()
    return sorted(list(communities))

def _get_communities_for_transition(df, start_year_col, start_comm_col, end_year_col, end_comm_col, start_year, end_year):
    """Gets sorted lists of unique source and target communities for a specific transition."""
    transition_df = df[(df[start_year_col] == start_year) & (df[end_year_col] == end_year)]
    source_communities = sorted(list(transition_df[start_comm_col].dropna().unique()))
    target_communities = sorted(list(transition_df[end_comm_col].dropna().unique()))
    return source_communities, target_communities

def generate_alluvial_matrices_from_edgelists(direct_edgelist_df, intermediate_edgelist_df=None):
    """
    Generates direct, intermediate, and identity transition matrices (DataFrames)
    for alluvial plots from weighted edgelists. Handles changing community sets.

    Parameters:
    direct_edgelist_df (pd.DataFrame): DataFrame with direct transitions (T_i -> T_{i+1}).
        Required columns: 'origin', 'origin_year', 'destination', 'destination_year', 'count'.
    intermediate_edgelist_df (pd.DataFrame): DataFrame with intermediate transitions
        (T_i -> T_{i+1} -> T_{i+2}). Required columns: 'origin', 'origin_year',
        'throughway', 'throughway_year', 'destination', 'destination_year', 'count'.

    Returns:
    tuple: (direct_trans_matrices, inter_trans_matrices, identity_matrices)
        - direct: List of DataFrames for T_i -> T_{i+1}. Index = origin comm, Columns = destination comm.
        - inter: List of lists. Each inner list corresponds to transitions
                 T_i -> T_{i+2} via T_{i+1}, containing tuples of
                 (intermediate_community, intermediate_matrix_df).
                 Matrix Index = origin comm, Columns = destination comm.
        - identity: List containing two tuples:
                    ('start', identity_df_T0) and ('end', identity_df_Tn).
                    Matrix Index = comm, Columns = comm, Value = total flow on diag.
    """
    # --- Identify Time Points ---
    origin_years = direct_edgelist_df['origin_year'].unique()
    dest_years = direct_edgelist_df['destination_year'].unique()
    all_years = set(origin_years) | set(dest_years)
    if intermediate_edgelist_df is not None and not intermediate_edgelist_df.empty:
        all_years |= set(intermediate_edgelist_df['origin_year'].unique())
        all_years |= set(intermediate_edgelist_df['throughway_year'].unique())
        all_years |= set(intermediate_edgelist_df['destination_year'].unique())

    time_cols = sorted(list(all_years))
    n_timepoints = len(time_cols)

    if n_timepoints < 2:
        raise ValueError("Requires at least 2 timepoints represented in the edgelists")

    direct_trans_matrices = []
    inter_trans_matrices = []
    identity_matrices = []

    # --- Direct Transitions (T_i -> T_{i+1}) ---
    for i in range(n_timepoints - 1):
        t_start = time_cols[i]
        t_end = time_cols[i+1]

        # Filter relevant edges for this specific transition
        trans_df = direct_edgelist_df[
            (direct_edgelist_df['origin_year'] == t_start) &
            (direct_edgelist_df['destination_year'] == t_end)
        ].copy()
        if trans_df.empty:
            # Handle cases where no direct transition exists between these years
            # Decide how to represent this: empty dataframe, or dataframe of zeros?
            # Let's create a DataFrame with appropriate index/columns if possible, else empty.
            source_comms, target_comms = _get_communities_for_transition(
                direct_edgelist_df, 'origin_year', 'origin', 'destination_year', 'destination', t_start, t_end
            )
            # Need communities present *at* these timepoints even if no flow between them
            source_comms_at_t = _get_communities_for_timepoint(direct_edgelist_df, 'origin_year', 'origin', t_start)
            target_comms_at_t = _get_communities_for_timepoint(direct_edgelist_df, 'destination_year', 'destination', t_end)

            all_relevant_source = sorted(list(set(source_comms) | set(source_comms_at_t)))
            all_relevant_target = sorted(list(set(target_comms) | set(target_comms_at_t)))

            matrix = pd.DataFrame(0.0, index=all_relevant_source, columns=all_relevant_target)

        else:
             # Determine communities active in this specific transition
            source_communities = sorted(list(trans_df['origin'].dropna().unique()))
            target_communities = sorted(list(trans_df['destination'].dropna().unique()))

            # Pivot to create the matrix
            matrix = pd.pivot_table(trans_df,
                                    values='count',
                                    index='origin',
                                    columns='destination',
                                    fill_value=0.0,
                                    aggfunc="sum") # Use sum in case of duplicate entries

            # Ensure all relevant communities are present, even if they have zero flow
            # Get communities present *at* these times, even if not in *this specific* transition flow
            source_comms_at_t = _get_communities_for_timepoint(direct_edgelist_df, 'origin_year', 'origin', t_start)
            target_comms_at_t = _get_communities_for_timepoint(direct_edgelist_df, 'destination_year', 'destination', t_end)

            all_relevant_source = sorted(list(set(source_communities) | set(source_comms_at_t)))
            all_relevant_target = sorted(list(set(target_communities) | set(target_comms_at_t)))


            matrix = matrix.reindex(index=all_relevant_source, columns=all_relevant_target, fill_value=0.0)

        direct_trans_matrices.append(matrix)

    # --- Intermediate Transitions (T_i -> T_{i+2} via T_{i+1}) ---
    if intermediate_edgelist_df is not None and not intermediate_edgelist_df.empty:
        for i in range(n_timepoints - 2):
            t_start = time_cols[i]
            t_intermediate = time_cols[i+1]
            t_end = time_cols[i+2]

            # Filter edges relevant for this 3-step transition period
            step_df = intermediate_edgelist_df[
                (intermediate_edgelist_df['origin_year'] == t_start) &
                (intermediate_edgelist_df['throughway_year'] == t_intermediate) &
                (intermediate_edgelist_df['destination_year'] == t_end)
            ].copy()

            intermediate_community_matrices = []
            if step_df.empty:
                inter_trans_matrices.append(intermediate_community_matrices) # Append empty list
                continue

            # Find unique intermediate communities for this specific step
            intermediate_communities = sorted(step_df['throughway'].dropna().unique())

            # Determine the universe of source/target communities for this T_i -> T_{i+2} step
            source_communities_step = sorted(list(step_df['origin'].dropna().unique()))
            target_communities_step = sorted(list(step_df['destination'].dropna().unique()))

            # Also consider communities present at start/end times even if not in intermediate flow
            source_comms_at_t_start = _get_communities_for_timepoint(direct_edgelist_df, 'origin_year', 'origin', t_start)
            # Use destination at T_end for target communities
            target_comms_at_t_end = _get_communities_for_timepoint(direct_edgelist_df, 'destination_year', 'destination', t_end)


            all_relevant_source = sorted(list(set(source_communities_step) | set(source_comms_at_t_start)))
            all_relevant_target = sorted(list(set(target_communities_step) | set(target_comms_at_t_end)))


            for inter_comm in intermediate_communities:
                # Filter for paths going through this specific intermediate community
                inter_comm_df = step_df[step_df['throughway'] == inter_comm]

                if inter_comm_df.empty or inter_comm_df['count'].sum() == 0:
                    continue # Skip if no flow through this intermediate community

                # Pivot for this specific intermediate path
                matrix = pd.pivot_table(inter_comm_df,
                                        values='count',
                                        index='origin',
                                        columns='destination',
                                        fill_value=0.0,
                                        aggfunc='sum')

                # Ensure matrix uses the full set of source/target communities for this T_i -> T_{i+2} period
                matrix = matrix.reindex(index=all_relevant_source, columns=all_relevant_target, fill_value=0.0)

                # Only add if there's actual flow represented
                if matrix.sum().sum() > 0:
                     # Ensure intermediate community label is consistent type (e.g., int) if needed
                    try:
                        inter_comm_label = int(inter_comm)
                    except ValueError:
                        inter_comm_label = str(inter_comm) # Handle non-integer communities
                    intermediate_community_matrices.append((inter_comm_label, matrix))

            # Append the list for this T_i -> T_{i+2} step
            inter_trans_matrices.append(intermediate_community_matrices)
    else:
         # If no intermediate df provided, create empty lists for each potential step
        inter_trans_matrices = [[] for _ in range(n_timepoints - 2)]


    # --- Identity Matrices (T0 and Tn) ---
    # Calculate based on total flow *originating* at T0 and *arriving* at Tn
    if n_timepoints >= 1:
        t_start = time_cols[0]
        t_end = time_cols[-1]

        # Start Identity (T0) - Sum of outgoing flows from direct transitions
        start_flows = direct_edgelist_df[direct_edgelist_df['origin_year'] == t_start]\
                                       .groupby('origin')['count'].sum()
        start_communities = sorted(list(start_flows.index))
        # Also include communities that *receive* flow in the first direct step, even if they don't send any
        # This defines the set of communities present at the start axis
        first_dest_comms = _get_communities_for_timepoint(direct_edgelist_df, 'destination_year', 'destination', time_cols[1] if n_timepoints > 1 else t_start )
        all_start_comms = sorted(list(set(start_communities) | set(first_dest_comms)))


        start_identity_df = pd.DataFrame(0.0, index=all_start_comms, columns=all_start_comms)
        for comm, count in start_flows.items():
             if comm in start_identity_df.index: # Check if community exists in the matrix index
                start_identity_df.loc[comm, comm] = float(count)
        identity_matrices.append(('start', start_identity_df))

        # End Identity (Tn) - Sum of incoming flows from direct transitions
        end_flows = direct_edgelist_df[direct_edgelist_df['destination_year'] == t_end]\
                                     .groupby('destination')['count'].sum()
        end_communities = sorted(list(end_flows.index))
         # Also include communities that *send* flow in the last direct step, even if they don't receive any
        last_origin_comms = _get_communities_for_timepoint(direct_edgelist_df, 'origin_year', 'origin', time_cols[-2] if n_timepoints > 1 else t_end)
        all_end_comms = sorted(list(set(end_communities) | set(last_origin_comms)))

        end_identity_df = pd.DataFrame(0.0, index=all_end_comms, columns=all_end_comms)
        for comm, count in end_flows.items():
            if comm in end_identity_df.index: # Check if community exists in the matrix index
                end_identity_df.loc[comm, comm] = float(count)
        identity_matrices.append(('end', end_identity_df))

    # If only one time point, identity matrices might be empty if calculated from transitions
    # The logic above handles >=1 timepoint, but meaningful identity requires >=2 for flow calculation

    return direct_trans_matrices, inter_trans_matrices, identity_matrices


if __name__ == "__main__":
    # # Create Dummy Data for demonstration:
    # Example direct edgelist
    data_direct = {
        'origin': [2, 4, 0, 3, 1, 4, 1, 0, 3, 2, 1, 2, 4, 0, 3, 1, 2, 3, 0, 4, 1, 2, 4, 3, 0, # 2009->2010
                1, 1, 1, 2, 2, 0, 0, 3, 3, 4, 4], # 2010->2011
        'origin_year': [2009]*25 + [2010]*11,
        'destination': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, # 2009->2010
                    1, 2, 0, 2, 3, 0, 1, 3, 0, 4, 0],# 2010->2011
        'destination_year': [2010]*25 + [2011]*11,
        'count': [6905, 1289, 68357, 6466, 1464241, 7816, 6504, 65429, 22837, 3398982, 7728, 23769, 16623, 229912, 7029548, 1338, 7709, 14030, 131608, 3368552, 23524, 47183, 53375, 139468, 10337706, # 2009->2010
                1000, 200, 300, 4000, 50, 6000, 70, 8000, 90, 10000, 110] # 2010->2011
    }
    direct_df = pd.DataFrame(data_direct)

    # Example intermediate edgelist (2009 -> 2010 -> 2011)
    data_intermediate = {
        'origin': [1, 1, 1, 1, 1, 2, 2, 0, 0, 3, 3, 4, 4],
        'origin_year': [2009]*13,
        'throughway': [1, 1, 1, 0, 0, 2, 2, 0, 1, 3, 3, 4, 4],
        'throughway_year': [2010]*13,
        'destination': [1, 2, 0, 0, 1, 2, 3, 0, 1, 3, 0, 4, 0],
        'destination_year': [2011]*13,
        'count': [1000, 20, 30, 40, 5, 4000, 50, 6000, 7, 8000, 9, 10000, 11]
    }
    intermediate_df = pd.DataFrame(data_intermediate)


    # Generate the matrices
    direct_mats, inter_mats, id_mats = generate_alluvial_matrices_from_edgelists(direct_df, intermediate_df)

    # --- Inspect the output ---
    print("--- Direct Transition Matrices ---")
    for i, mat in enumerate(direct_mats):
        print(f"Transition {i} -> {i+1}:")
        print(mat)
        print("-" * 20)

    print("\n--- Intermediate Transition Matrices ---")
    for i, step_list in enumerate(inter_mats):
        print(f"Step {i} -> {i+2}:")
        if not step_list:
            print("  (No intermediate transitions)")
        for inter_comm, mat in step_list:
            print(f"  Via Intermediate Community: {inter_comm}")
            print(mat)
            print("  ---")
        print("-" * 20)

    print("\n--- Identity Matrices ---")
    for label, mat in id_mats:
        print(f"Identity Matrix: {label}")
        print(mat)
        print("-" * 20)