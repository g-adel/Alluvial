import numpy as np
def create_alluvial_data(df, time_cols):
    """
    Convert DataFrame with community classifications to alluvial input format
    for multiple timepoints
    
    Parameters:
    df: DataFrame with community classifications
    time_cols: list of column names representing timepoints
    
    Returns:
    list: Array of dictionaries ready for alluvial.plot
    """
    results = []
    
    # Create transition matrices for consecutive timepoints
    for i in range(len(time_cols) - 1):
        current_time = time_cols[i]
        next_time = time_cols[i + 1]
        
        result = {}
        source_communities = sorted(df[current_time].unique())
        target_communities = sorted(df[next_time].unique())
        
        # Initialize the nested dictionary
        for source in source_communities:
            source_label = f"{source}_{current_time}"
            result[source_label] = {}
            for target in target_communities:
                target_label = f"{target}_{next_time}"
                result[source_label][target_label] = 0.0
        
        # Count transitions between communities
        for source in source_communities:
            source_label = f"{source}_{current_time}"
            source_group = df[df[current_time] == source]
            
            for target in target_communities:
                target_label = f"{target}_{next_time}"
                count = len(source_group[source_group[next_time] == target])
                if count > 0:
                    result[source_label][target_label] = float(count)
        
        # Remove empty source communities
        result = {k: v for k, v in result.items() 
                 if any(val > 0 for val in v.values())}
        result = dict_to_matrix(result, len(source_communities))
        results.append(result)
    
    return results


def create_alluvial_data_with_intermediates(df, time_cols):
    """
    Convert DataFrame with community classifications to alluvial input format
    including indirect transitions through intermediate communities
    
    Parameters:
    df: DataFrame with community classifications
    time_cols: list of 3 column names representing timepoints
    
    Returns:
    tuple: (direct_transitions, intermediate_transitions)
        - direct_transitions: list of direct transition matrices
        - intermediate_transitions: list of transition matrices through each T2 community
    """
    if len(time_cols) != 3:
        raise ValueError("This function requires exactly 3 timepoints")
    
    # Get direct transitions (T1->T2 and T2->T3)
    direct_transitions = create_alluvial_data(df, time_cols)
    
    # Get indirect transitions through each T2 community
    intermediate_transitions = []
    t2_communities = sorted(df[time_cols[1]].unique())
    
    for t2_community in t2_communities:
        # Filter nodes that passed through this T2 community
        intermediate_nodes = df[df[time_cols[1]] == t2_community]
        
        result = {}
        source_communities = sorted(intermediate_nodes[time_cols[0]].unique())
        target_communities = sorted(intermediate_nodes[time_cols[2]].unique())
        
        # Initialize the nested dictionary
        for source in source_communities:
            source_label = f"{source}_{time_cols[0]}"
            result[source_label] = {}
            for target in target_communities:
                target_label = f"{target}_{time_cols[2]}"
                result[source_label][target_label] = 0.0
        
        # Count transitions
        for source in source_communities:
            source_label = f"{source}_{time_cols[0]}"
            source_group = intermediate_nodes[intermediate_nodes[time_cols[0]] == source]
            
            for target in target_communities:
                target_label = f"{target}_{time_cols[2]}"
                count = len(source_group[source_group[time_cols[2]] == target])
                if count > 0:
                    result[source_label][target_label] = float(count)
        
        # Remove empty source communities
        result = {k: v for k, v in result.items() 
                 if any(val > 0 for val in v.values())}
        result = dict_to_matrix(result, len(df[time_cols[0]].unique()))
        intermediate_transitions.append((t2_community, result))
    
    return direct_transitions, intermediate_transitions

def dict_to_matrix(dict, matrix_size):
    matrix = np.zeros((matrix_size, matrix_size))
    for source, targets in dict.items():
        source_idx = int(source.split('_')[0])
        for target, value in targets.items():
            target_idx = int(target.split('_')[0])
            matrix[source_idx, target_idx] = value
    return matrix