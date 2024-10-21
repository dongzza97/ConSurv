import os
import yaml
import json
import numpy as np
import pandas as pd

def save_checkpoint(model_checkpoints_folder, state):
    with open(os.path.join(model_checkpoints_folder, 'top_performance.json'), 'w') as outfile:
        json.dump(state, outfile)

def save_test_1(data_name, performance_data_1): 

    df = pd.DataFrame(performance_data_1)
    # csv_filename = '{}_ConSurv_performance.csv'.format(data_name, data_name)
    # csv_filename = '{}_NLL_performance.csv'.format(data_name, data_name)
    csv_filename = '{}_NLL_NCE_performance.csv'.format(data_name, data_name)

    if not os.path.isfile(csv_filename):
        df.to_csv(csv_filename, index=False)
    else:
        df.to_csv(csv_filename, mode='a', header=False, index=False)

def save_test_2(data_name, performance_data_2, performance_data_3, seed):

    # Time-dependent C-index
    results1_df = pd.DataFrame(np.array(performance_data_2).reshape(1,3), columns=[f'time_{t}' for t in range(3)])
    results1_df['SEED'] = seed
    # results1_csv = '{}_ConSurv_TD_Cindex.csv'.format(data_name)
    # results1_csv = '{}_NLL_TD_Cindex.csv'.format(data_name)
    results1_csv = '{}_NLL_NCE_TD_Cindex.csv'.format(data_name)

    
    if not os.path.isfile(results1_csv):
        results1_df.to_csv(results1_csv, index=False)
    else:
        results1_df.to_csv(results1_csv, mode='a', header=False, index=False)
    
    # Time-dependent Brier Score
    results2_df = pd.DataFrame(np.array(performance_data_3).reshape(1,3), columns=[f'time_{t}' for t in range(3)])
    results2_df['seed'] = seed
    # results2_csv = '{}_ConSurv_TD_Brier.csv'.format(data_name)
    # results2_csv = '{}_NLL_TD_Brier.csv'.format(data_name)
    results2_csv = '{}_NLL_NCE_TD_Brier.csv'.format(data_name)
    
    if not os.path.isfile(results2_csv):
        results2_df.to_csv(results2_csv, index=False)
    else:
        results2_df.to_csv(results2_csv, mode='a', header=False, index=False)


    