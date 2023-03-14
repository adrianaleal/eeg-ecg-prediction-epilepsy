'''
********************************** TEST MAIN **********************************
Test models for all patients (main)

Author: Beatriz Martinho
Date: December 2021
'''

# %% Import functions
import os

from test_patient import test
from import_data import get_patients_names
from logger import Logger

# %% Main    

if __name__ == '__main__':

    signal_type_vec = ['ECG']  # , 'EEG'

    # approaches = ['Hybrid', 'Standard']
    approaches = ['Hybrid', 'Control', 'SOTA']

    # Define firing power threshold
    # th_string = 'opt'  # threshold grid-search
    th_string = '0.7'
    # th_string = '0.5'

    train_perf_metric = 'GM'
    threshold_redundance = 0.9

    # path to save figures and excel files
    folder2saveFigures = os.path.abspath(os.path.join('Results', 'figures_paper_threshold_redundance' + str(
        int(threshold_redundance * 100)) + '_th_' + th_string + '_' + train_perf_metric.lower()))
    if not os.path.exists(folder2saveFigures):
        os.makedirs(folder2saveFigures)

    # test('109502', signal_type, 'Standard', train_perf_metric, features_path, folder2saveFigures, th_string,
    # study_logger)

    for signal_type in signal_type_vec:
        features_path = os.path.abspath(os.path.join('Data', signal_type + 'FeaturesPython'))
        patient_names_sorted = get_patients_names(features_path)

        log_file_name = 'log_file_' + signal_type.lower() + '_seizure_prediction_testing'
        study_logger = Logger().get(log_file_name, display_console=True)

        # [8:] --> 32702
        # [28:] --> 98102
        # [6:] --> 26102
        for patient in patient_names_sorted[8:]:
            for approach in approaches:
                test(patient, signal_type, approach, train_perf_metric, features_path, folder2saveFigures, th_string,
                     study_logger)
                print('')
        print('End testing for signal ' + signal_type)
