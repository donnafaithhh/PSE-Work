import os
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# for choosing random dates
import random
from datetime import datetime, timedelta

# for getting the wasserstein distance
import itertools
import persim
import ripser
from persim import wasserstein
from scipy.stats import wasserstein_distance

# to remove the warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

directory = r"matrices"
random_dates = ['2013-11-29','2014-10-28','2015-11-18','2016-09-06','2017-03-08','2018-02-13','2019-03-01','2020-03-06','2020-08-18','2020-09-04','2020-10-27','2020-11-17','2021-11-05','2022-01-19','2022-04-26','2022-06-28','2023-05-01']

rep_b0_lists = []
rep_file_names = []

for name in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, name)
    
    if any(date in name for date in random_dates):
        if os.path.isfile(file_path) and name.endswith('.pickl'):
            rep_file_names.append(name)
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
        b0_df['min max'] = (b0_df['b0'] - min(b0_data)[1])/(max(b0_data)[1] - min(b0_data)[1])
        
        rep_b0_lists.append(b0_df['min max'].to_list())

b0_lists = []
file_names = []
year = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

for name in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, name)
    
    if any(y in name for y in year):
        if os.path.isfile(file_path) and name.endswith('.pickl'):
            file_names.append(name)
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
        b0_df['min max'] = (b0_df['b0'] - min(b0_data)[1])/(max(b0_data)[1] - min(b0_data)[1])
        
        b0_lists.append(b0_df['min max'].to_list())
        print(name)

distance_matrix = pd.DataFrame(
    0.0, 
    index=file_names, 
    columns=rep_file_names
)

for i, file_i in enumerate(file_names):
    for j, file_j in enumerate(rep_file_names):
        dist = wasserstein_distance(b0_lists[i], rep_b0_lists[j])
        distance_matrix.loc[file_i, file_j] = dist

if '.ipynb_checkpoints' in distance_matrix.index:
    distance_matrix = distance_matrix.drop('.ipynb_checkpoints', axis=0)
    distance_matrix = distance_matrix.drop('.ipynb_checkpoints', axis=1)

distance_matrix.to_pickle("Wasserstein Precrash.pkl") 
