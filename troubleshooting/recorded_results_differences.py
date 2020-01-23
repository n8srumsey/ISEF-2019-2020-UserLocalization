"""
recorded_results_differences.py
~~~~~~~~~~~~~~~~~~
Lists differences in stored histories and records of optimization.

"""
import json
import os
import pickle


def list_trials_not_trial_history():
    trials_in_results = []
    trials_in_model_visualizations = []
    trials_in_weights = []
    trials_in_trials_history = []

    results_folder_path = "../results"
    results = sorted(os.listdir(results_folder_path))
    for file_name in results:
        file_path = os.path.join(results_folder_path, file_name)
        with open(file_path) as f:
            j = json.load(f)
        trial_uuid = j["model_uuid"]
        trials_in_results.append(trial_uuid)

    model_visualizations_folder_path = "../model-visualizations"
    model_visualizations = sorted(os.listdir(model_visualizations_folder_path))
    for file_name in model_visualizations:
        file_uuid = file_name[:file_name.find(".")]
        if file_uuid != "model_best":
            trials_in_model_visualizations.append(file_uuid)

    weights_folder_path = "../weights"
    weights = sorted(os.listdir(weights_folder_path))
    for file_name in weights:
        file_uuid = file_name[:file_name.find(".")]
        trials_in_weights.append(file_uuid)

    trials_history = pickle.load((open("../trials_history.pkl", "rb")))
    for trial in trials_history:
        trial_uuid = trial["result"]["model_uuid"]
        trials_in_trials_history.append(trial_uuid)

    print("Results:")
    print(sorted(trials_in_results))
    print("Model_Visualizations:")
    print(sorted(trials_in_model_visualizations))
    print("Weights:")
    print(sorted(trials_in_weights))
    print("Trials_history:")
    print(sorted(trials_in_trials_history))

    diff_results_model_visual = list(set(trials_in_results).symmetric_difference(trials_in_model_visualizations))
    diff_results_weights = list(set(trials_in_results).symmetric_difference(trials_in_weights))
    diff_results_trial_history = list(set(trials_in_results).symmetric_difference(trials_in_trials_history))
    diff_model_visual_weights = list(set(trials_in_model_visualizations).symmetric_difference(trials_in_weights))
    diff_model_visual_trial_history = list(set(trials_in_model_visualizations).symmetric_difference(trials_in_trials_history))
    diff_weights_trial_history = list(set(trials_in_weights).symmetric_difference(trials_in_trials_history))

    print("\nDifferences:")
    print("Results-model_visualizations:")
    print(sorted(diff_results_model_visual))
    print("Results-weights")
    print(sorted(diff_results_weights))
    print("Results-Trial_history:")
    print(sorted(diff_results_trial_history))
    print("Model_visual-weights")
    print(sorted(diff_model_visual_weights))
    print("model_visual-trial_history")
    print(sorted(diff_model_visual_trial_history))
    print("weights-trial_history")
    print(sorted(diff_weights_trial_history))


list_trials_not_trial_history()
