import os
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# for the persistence diagrams
import gudhi as gd
import gudhi.representations

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# this is for H2 
directory = r"matrices"
plot_directory = r"trial_pers_hom"

all_stats_df = pd.DataFrame(columns=['file name', 'holes', 'birth time', 'death time', 'persistence'])
h2_list_holes_all = []
h2_birth_all = []
h2_death_all = []
h2_persistence_all = []

for name in sorted(os.listdir(directory)):
    file_path = os.path.join(directory, name)
    
    if os.path.isfile(file_path) and name.endswith('.pickl'):
        try:
            with open(file_path, 'rb') as f:
                print(f"\n\nDistance matrix of {name}")
                data = pickle.load(f)
                dist_matrix = data.get('distance matrix')
                if dist_matrix is None:
                    print(f"Distance matrix not found in {name}")
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"Error loading {name}: {e}")
            continue

    # Create Rips complex
    max_dist = np.max(dist_matrix)
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_dist)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence()
    
    # Extract 1D persistence intervals (holes)
    persistence_diagram = simplex_tree.persistence_intervals_in_dimension(2)
    
    # Define a persistence threshold
    threshold = 0.1
    persistent_points = [(birth, death) for birth, death in persistence_diagram if (death - birth) > threshold]
    
    if persistent_points:
        counter = 0
        for i, (birth, death) in enumerate(persistent_points):
            print(f"Persistent hole {i+1}: Birth = {birth:.4f}, Death = {death:.4f}, Persistence = {death - birth:.4f}")
            counter += 1
        print(f"{counter} persistent hole/(s) found!")
        h2_list_holes_all.append(counter)
        h2_birth_all.append(birth)
        h2_death_all.append(death)
        h2_persistence_all.append(death - birth)
        all_stats_df.loc[len(all_stats_df)] = [name, counter, birth, death, death - birth]
    else:
        print("No significant persistent holes detected.")
        h2_list_holes_all.append(0)
        h2_birth_all.append(None)
        h2_death_all.append(None)
        h2_persistence_all.append(None)
        all_stats_df.loc[len(all_stats_df)] = [name, 0, None, None, None]

all_stats_df.to_pickle("H2_voids.pkl") 