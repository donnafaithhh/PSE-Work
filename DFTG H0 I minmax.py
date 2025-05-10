import os
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# for the persistence diagrams and the betti numbers
import scipy as sp
from matplotlib import cm
import ripser
import persim

# for the persistence landscape
from persim import PersLandscapeApprox, plot_diagrams
from persim.landscapes import plot_landscape_simple

# for running the code faster
import multiprocessing as mp

# to remove the warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

directory = r"matrices"
plot_directory = r"minmax_connected_components"

for name in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, name)
    
    if os.path.isfile(file_path) and name.endswith('.pickl'):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                dist_matrix = data.get('distance matrix')
                if dist_matrix is None:
                    print(f"Distance matrix not found in {name}")
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"Error loading {name}: {e}")
            continue
    
    # making a persistence diagram from a rips filtration 
    pers_diag = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=2)
    diagrams = pers_diag['dgms']
    thresh_list = np.linspace(0, 1, 50)
    b0_data = []

    for t in thresh_list:
        pers_diag2 = ripser.ripser(dist_matrix, distance_matrix=True, thresh=t, maxdim=2)
        diagrams2 = pers_diag2['dgms']
        b0 = sum(d[1] == np.inf for d in diagrams2[0]) if len(diagrams2) > 0 else 0
        b0_data.append((t, b0))
        
    b0_df = pd.DataFrame(b0_data, columns=['threshold', 'b0'])

    # # trying to normalize the values via percent
    # b0_df['percent'] = b0_df['b0'] / max(b0_data)[1]
    
    # trying to normalize the values via min max
    b0_df['min max'] = (b0_df['b0'] - min(b0_data)[1])/(max(b0_data)[1] - min(b0_data)[1])
    
    # plotting diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    persim.plot_diagrams(diagrams, plot_only=[0], ax=ax1)
    ax1.set_title('H0 Persistence Diagram')
    
    ax2.plot(b0_df['threshold'], b0_df['min max'])
    ax2.set_xlabel('Death Threshold')
    ax2.set_ylabel('B0 (in percent)')
    ax2.set_title("H0 at Different Thresholds")
    ax2.grid()
    fig.suptitle(f'Connected Components for {name}')
    plt.tight_layout()
    
    # saving the diagram
    diagram_path = os.path.join(plot_directory, f"{name} H0 minmax Connected Components.png")
    print(diagram_path)
    plt.savefig(diagram_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
