import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from my_alluvial import *
from plot_composite_alluvial import *
from generate_alluvial_matrices import *



def main():
    df = pd.read_csv('test_data.csv')
    time_cols = list(df.columns)[1:]
    direct_trans_matrices, inter_trans_matrices, identity_matrices = generate_alluvial_matrices(df,list(df.columns)[1:])

    fig, all_axes = plot_composite_alluvial(df,time_cols,direct_trans_matrices,
                            inter_trans_matrices, interp_frac=1)
    # fig.suptitle('Composite Alluvial Diagram', fontsize=16)
    # fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()