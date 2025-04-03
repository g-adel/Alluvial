import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from my_alluvial import *
from plot_composite_alluvial import *
from generate_alluvial_matrices import *



def plot_df(df):
    """
    Produces a composite alluvial plot out of an inputted csv filename 
    """

    time_cols = list(df.columns)[1:]
    direct_trans_matrices, inter_trans_matrices = generate_alluvial_matrices(df,list(df.columns)[1:])

    fig, all_axes = plot_composite_alluvial(df,time_cols,direct_trans_matrices,
                            inter_trans_matrices, interp_frac=1)
    plt.show()
    

if __name__ == "__main__":
    
    # Default CSV file
    csv_file = 'test_data.csv'
    
    # Check if a CSV file path is provided as command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        df = pd.read_csv(csv_file)
        plot_df(df)
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_file}'")
        sys.exit(1)