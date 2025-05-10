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
plot_directory = r"connected_components"
max_thresh_df = pd.DataFrame(columns=['file name', 'max threshold'])

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

    t = 0
    step = 0.001
    
    while True:
        pers_diag2 = ripser.ripser(dist_matrix, distance_matrix=True, thresh=t, maxdim=2)
        diagrams2 = pers_diag2['dgms']
        b0 = sum(d[1] == np.inf for d in diagrams2[0]) if len(diagrams2) > 0 else 0
        
        if b0 == 1:
            new_row = pd.DataFrame([{
                "file name": name,
                'max threshold': t
            }])
            print(new_row)
            max_thresh_df = pd.concat([max_thresh_df, new_row], ignore_index=True)
            break
        else:
            t += step

max_thresh_df.to_pickle("H0_components.pkl") 