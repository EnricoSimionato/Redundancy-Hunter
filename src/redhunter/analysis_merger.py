import numpy as np
import os
import pickle as pkl
from matplotlib import pyplot as plt

from exporch import Config
from exporch.experiment import benchmark_id_metric_name_mapping
from exporch.utils.plot_utils import plot_heatmap
from redhunter.analysis.analysis_utils import extract_number, layer_name_matrix_name_mapping
from redhunter.analysis.layer_replacement_analysis import LayerReplacementAnalysis, SingleNullLayersReplacementAnalysis


def main():
    #model_id = "Llama-3.1-8B"
    model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_replacement = "5"
    version_null = "5"
    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement}", "storage.pkl"), "rb") as f:
        data_couples = pkl.load(f)

    config = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement}/config.yaml"))
    fig_size = config.get("figure_size")
    x_rotation = 0

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version_null}", "storage.pkl"), "rb") as f:
        data_ablation = pkl.load(f)

    results_dict_couples = data_couples[1]
    results_data_ablation = data_ablation[1]
    original_model_performance_dict = {}
    for benchmark_id, results in results_dict_couples.items():
        original_model_performance_dict[benchmark_id] = results.pop(("original", "original"))
        results_data_ablation[benchmark_id].pop(("original", "original"))
    source_paths_couples, destination_paths_couples, performance_arrays_couples = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples), original_model_performance_dict)
    source_paths_ablation, destination_paths_ablation, performance_arrays_ablation = SingleNullLayersReplacementAnalysis.format_result_dictionary_to_plot(
        *SingleNullLayersReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_data_ablation), original_model_performance_dict)

    # Checking if the path to save the plots exists, if not it creates it
    if not os.path.exists(os.path.join(model_path, f"difference_between_replacement_and_ablation_analysis/version_{version_replacement}")):
        os.makedirs(os.path.join(model_path, f"difference_between_replacement_and_ablation_analysis/version_{version_replacement}"))

    for benchmark_id in results_dict_couples.keys():
        max_abs = np.abs([el for row in performance_arrays_couples[benchmark_id] - performance_arrays_ablation[benchmark_id] for el in row if not np.isnan(el)]).max()
        x_labels = None
        y_labels = None
        try:
            for tensor_keys in destination_paths_couples[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = "\n".join(plot_labels)

                if x_labels is None:
                    x_labels = [plot_label,]
                else:
                    x_labels.append(plot_label)

            for tensor_keys in source_paths_couples[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = " ".join(plot_labels)

                if y_labels is None:
                    y_labels = [plot_label, ]
                else:
                    y_labels.append(plot_label)
        except KeyError as e:
            print(e)
            x_labels = destination_paths_couples[benchmark_id]
            y_labels = source_paths_couples[benchmark_id]
            x_rotation = 90

        plot_heatmap(
            [[performance_arrays_couples[benchmark_id] - performance_arrays_ablation[benchmark_id]]],
            os.path.join(model_path,
                         f"difference_between_replacement_and_ablation_analysis/version_{version_replacement}",
                         f"heatmap_{benchmark_id}_couples_minus_ablation.pdf"),
            #title=f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
            #axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
            x_title="Labels of the overwritten layers",
            y_title="Labels of the duplicated layers",
            x_labels=[x_labels,],
            y_labels=[y_labels,],
            cmap_str="seismic",
            x_rotation=x_rotation,
            fig_size=fig_size,
            edge_color="white",
            fontsize=23,
            tick_label_size=20,
            x_title_size=26,
            y_title_size=26,
            precision=3,
            vmin=[-max_abs, ],
            vmax=[max_abs, ],
            #colorbar_top_label="High\nsimilarity",
            #colorbar_bottom_label="Low\nsimilarity",
            colorbar_title=r"$\Delta\text{accuracy} = \text{accuracy}_{\,\text{replacement}} - \text{accuracy}_{\,\text{ablation}}$"
        )
"""
def main():
    model_id = "Llama-3.1-8B"
    #model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_replacement_1 = "0"
    version_replacement_2 = "4"
    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_1}", "storage.pkl"), "rb") as f:
        data_couples_1 = pkl.load(f)

    config = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_1}/config.yaml"))
    fig_size = config.get("figure_size")
    x_rotation = 0

    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_2}", "storage.pkl"), "rb") as f:
        data_couples_2 = pkl.load(f)

    results_dict_couples_1 = data_couples_1[1]
    results_dict_couples_2 = data_couples_2[1]
    original_model_performance_dict = {}
    for benchmark_id, results in results_dict_couples_1.items():
        original_model_performance_dict[benchmark_id] = results.pop(("original", "original"))
        results_dict_couples_2[benchmark_id].pop(("original", "original"))
    source_paths_couples_1, destination_paths_couples_1, performance_arrays_couples_1 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_1), original_model_performance_dict)
    source_paths_couples_2, destination_paths_couples_2, performance_arrays_couples_2 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_2), original_model_performance_dict)

    # Checking if the path to save the plots exists, if not it creates it
    if not os.path.exists(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}")):
        os.makedirs(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}"))

    for benchmark_id in results_dict_couples_1.keys():

        max_abs = np.abs([el for row in performance_arrays_couples_1[benchmark_id] - performance_arrays_couples_2[benchmark_id] for el in row if not np.isnan(el)]).max()
        # Creating overwriting and overwritten layers labels
        x_labels = None
        y_labels = None
        try:
            for tensor_keys in destination_paths_couples_1[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = "\n".join(plot_labels)

                if x_labels is None:
                    x_labels = [plot_label,]
                else:
                    x_labels.append(plot_label)

            for tensor_keys in source_paths_couples_1[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = ", ".join(plot_labels)

                if y_labels is None:
                    y_labels = [plot_label, ]
                else:
                    y_labels.append(plot_label)
        except KeyError as e:
            print(e)
            x_labels = destination_paths_couples_1[benchmark_id]
            y_labels = source_paths_couples_1[benchmark_id]
            x_rotation = 90

        plot_heatmap(
            [[performance_arrays_couples_1[benchmark_id] - performance_arrays_couples_2[benchmark_id]]],
            os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}",
                         f"heatmap_{benchmark_id}_version_{version_replacement_1}_minus_version_{version_replacement_2}.pdf"),
            #title=f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
            #axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
            x_title="Labels of the overwritten layers",
            y_title="Labels of the duplicated layers",
            x_labels=[x_labels,],
            y_labels=[y_labels,],
            cmap_str="seismic",
            x_rotation=x_rotation,
            fig_size=fig_size,
            edge_color="white",
            fontsize=23,
            tick_label_size=20,
            x_title_size=26,
            y_title_size=26,
            precision=3,
            vmin=[-max_abs, ],
            vmax=[max_abs, ]
        )
"""
"""
def main():
    # Difference between differences
    model_id = "Llama-3.1-8B"
    #model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_replacement_1 = "0"
    version_null_1 = "0"
    version_replacement_2 = "4"
    version_null_2 = "4"

    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_1}", "storage.pkl"), "rb") as f:
        data_couples_1 = pkl.load(f)

    config_1 = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_1}/config.yaml"))
    fig_size = config_1.get("figure_size")
    x_rotation = 0

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version_null_1}", "storage.pkl"), "rb") as f:
        data_ablation_1 = pkl.load(f)

    config_2 = Config(os.path.join(model_path,
                                   f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_2}/config.yaml"))

    results_dict_couples_1 = data_couples_1[1]
    results_data_ablation_1 = data_ablation_1[1]
    original_model_performance_dict_1 = {}
    for benchmark_id, results in results_dict_couples_1.items():
        original_model_performance_dict_1[benchmark_id] = results.pop(("original", "original"))
        results_data_ablation_1[benchmark_id].pop(("original", "original"))
    source_paths_couples_1, destination_paths_couples_1, performance_arrays_couples_1 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_1), original_model_performance_dict_1)
    source_paths_ablation_1, destination_paths_ablation_1, performance_arrays_ablation_1 = SingleNullLayersReplacementAnalysis.format_result_dictionary_to_plot(
        *SingleNullLayersReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_data_ablation_1), original_model_performance_dict_1)

    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_2}", "storage.pkl"), "rb") as f:
        data_couples_2 = pkl.load(f)

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version_null_2}", "storage.pkl"), "rb") as f:
        data_ablation_2 = pkl.load(f)

    results_dict_couples_2 = data_couples_2[1]
    results_data_ablation_2 = data_ablation_2[1]
    original_model_performance_dict_2 = {}
    for benchmark_id, results in results_dict_couples_2.items():
        original_model_performance_dict_2[benchmark_id] = results.pop(("original", "original"))
        results_data_ablation_2[benchmark_id].pop(("original", "original"))
    source_paths_couples_2, destination_paths_couples_2, performance_arrays_couples_2 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_2), original_model_performance_dict_2)
    source_paths_ablation_2, destination_paths_ablation_2, performance_arrays_ablation_2 = SingleNullLayersReplacementAnalysis.format_result_dictionary_to_plot(
        *SingleNullLayersReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_data_ablation_2), original_model_performance_dict_2)



    # Checking if the path to save the plots exists, if not it creates it
    if not os.path.exists(os.path.join(model_path, f"difference_between_difference_and_difference_analysis/version_{version_replacement_1}_{version_null_1}_minus_{version_replacement_2}_{version_null_2}")):
        os.makedirs(os.path.join(model_path, f"difference_between_difference_and_difference_analysis/version_{version_replacement_1}_{version_null_1}_minus_{version_replacement_2}_{version_null_2}"))

    for benchmark_id in source_paths_couples_1.keys():
        max_abs = np.abs([el for row in performance_arrays_couples_1[benchmark_id] - performance_arrays_ablation_1[benchmark_id] - (performance_arrays_couples_2[benchmark_id] - performance_arrays_ablation_2[benchmark_id]) for el in row if not np.isnan(el)]).max()
        x_labels = None
        y_labels = None
        try:
            for tensor_keys in destination_paths_couples_1[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = "\n".join(plot_labels)

                if x_labels is None:
                    x_labels = [plot_label,]
                else:
                    x_labels.append(plot_label)

            for tensor_keys in source_paths_couples_1[benchmark_id]:
                key_elements = tensor_keys.split("\n")
                plot_labels = []
                for key_element in key_elements:
                    key_element_list_format = [el.replace("'", "").replace('"',"") for el in key_element[1:-1].split(", ")]
                    layer_idx = extract_number(key_element)
                    key_element_label = layer_name_matrix_name_mapping[key_element_list_format[-1]][:-2] + rf"\, , {layer_idx}" + "}$"
                    plot_labels.append(key_element_label)
                plot_label = ", ".join(plot_labels)

                if y_labels is None:
                    y_labels = [plot_label, ]
                else:
                    y_labels.append(plot_label)
        except KeyError as e:
            print(e)
            x_labels = destination_paths_couples_1[benchmark_id]
            y_labels = source_paths_couples_1[benchmark_id]
            x_rotation = 90

        remove_diagonal = True
        mask_1 = np.ones(performance_arrays_couples_1[benchmark_id].shape)
        mask_2 = np.ones(performance_arrays_couples_2[benchmark_id].shape)
        if remove_diagonal:
            mask_1 = mask_1 - np.diag(np.diag(mask_1))
            mask_2 = mask_2 - np.diag(np.diag(mask_2))

        print(f"benchmark_id: {benchmark_id}")
        print(config_1.get("targets"))
        print(np.nanmean((performance_arrays_couples_1[benchmark_id] - performance_arrays_ablation_1[benchmark_id])*mask_1))
        print(config_2.get("targets"))
        print(np.nanmean((performance_arrays_couples_2[benchmark_id] - performance_arrays_ablation_2[benchmark_id])*mask_2))

        plot_heatmap(
            [[performance_arrays_couples_1[benchmark_id] - performance_arrays_ablation_1[benchmark_id] - (performance_arrays_couples_2[benchmark_id] - performance_arrays_ablation_2[benchmark_id])]],
            os.path.join(model_path, f"difference_between_difference_and_difference_analysis/version_{version_replacement_1}_{version_null_1}_minus_{version_replacement_2}_{version_null_2}",
                         f"heatmap_{benchmark_id}_couples_minus_ablation.pdf"),
            #title=f"Results for the model {config.get('model_id').split('/')[-1]} on the task {benchmark_id}",
            #axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
            x_title="Labels of the overwritten layers",
            y_title="Labels of the duplicated layers",
            x_labels=[x_labels,],
            y_labels=[y_labels,],
            cmap_str="seismic",
            x_rotation=x_rotation,
            fig_size=fig_size,
            edge_color="white",
            fontsize=23,
            tick_label_size=20,
            x_title_size=26,
            y_title_size=26,
            precision=3,
            vmin=[-max_abs, ],
            vmax=[max_abs, ]
        )
"""
"""
def main():
    # Average performance of 2 replacement experiments
    #model_id = "Llama-3.1-8B"
    model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_replacement_1 = "0"
    version_replacement_2 = "4"
    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_1}", "storage.pkl"), "rb") as f:
        data_couples_1 = pkl.load(f)

    config_1 = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_1}/config.yaml"))

    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_2}", "storage.pkl"), "rb") as f:
        data_couples_2 = pkl.load(f)

    config_2 = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_2}/config.yaml"))

    results_dict_couples_1 = data_couples_1[1]
    results_dict_couples_2 = data_couples_2[1]
    original_model_performance_dict = {}
    for benchmark_id, results in results_dict_couples_1.items():
        original_model_performance_dict[benchmark_id] = results.pop(("original", "original"))
        results_dict_couples_2[benchmark_id].pop(("original", "original"))
    source_paths_couples_1, destination_paths_couples_1, performance_arrays_couples_1 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_1), original_model_performance_dict)
    source_paths_couples_2, destination_paths_couples_2, performance_arrays_couples_2 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_2), original_model_performance_dict)

    # Checking if the path to save the plots exists, if not it creates it
    if not os.path.exists(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}")):
        os.makedirs(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}"))

    for benchmark_id in results_dict_couples_1.keys():
        print(f"benchmark_id: {benchmark_id}")
        print(config_1.get("targets"))
        print(np.nanmean(performance_arrays_couples_1[benchmark_id]))
        print(config_2.get("targets"))
        print(np.nanmean(performance_arrays_couples_2[benchmark_id]))
"""
"""
def main():
    # Average performance of 2 replacement experiments
    #model_id = "Llama-3.1-8B"
    model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_replacement_1 = "0"
    version_replacement_2 = "4"
    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_1}", "storage.pkl"), "rb") as f:
        data_couples_1 = pkl.load(f)

    config_1 = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_1}/config.yaml"))

    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version_replacement_2}", "storage.pkl"), "rb") as f:
        data_couples_2 = pkl.load(f)

    config_2 = Config(os.path.join(model_path, f"all_layer_couples_displacement_based_replacement_redundancy_analysis/version_{version_replacement_2}/config.yaml"))

    results_dict_couples_1 = data_couples_1[1]
    results_dict_couples_2 = data_couples_2[1]
    original_model_performance_dict = {}
    for benchmark_id, results in results_dict_couples_1.items():
        original_model_performance_dict[benchmark_id] = results.pop(("original", "original"))
        results_dict_couples_2[benchmark_id].pop(("original", "original"))
    source_paths_couples_1, destination_paths_couples_1, performance_arrays_couples_1 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_1), original_model_performance_dict)
    source_paths_couples_2, destination_paths_couples_2, performance_arrays_couples_2 = LayerReplacementAnalysis.format_result_dictionary_to_plot(
        *LayerReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_dict_couples_2), original_model_performance_dict)

    # Checking if the path to save the plots exists, if not it creates it
    if not os.path.exists(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}")):
        os.makedirs(os.path.join(model_path, f"difference_between_replacements/version_{version_replacement_1}_version_{version_replacement_2}"))

    for benchmark_id in results_dict_couples_1.keys():
        print(f"benchmark_id: {benchmark_id}")
        print(config_1.get("targets"))
        print(np.nanmean(performance_arrays_couples_1[benchmark_id]))
        print(config_2.get("targets"))
        print(np.nanmean(performance_arrays_couples_2[benchmark_id]))
"""
"""
def main():
    # Average performance of 2 ablation experiments
    model_id = "Llama-3.1-8B"
    #model_id = "gemma-2-2b"
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version_ablation_1 = "0"
    version_ablation_2 = "4"

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version_ablation_1}", "storage.pkl"), "rb") as f:
        data_ablation_1 = pkl.load(f)

    config_1 = Config(os.path.join(model_path, f"single_null_layers_replacement_redundancy_analysis/version_{version_ablation_1}/config.yaml"))

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version_ablation_2}", "storage.pkl"), "rb") as f:
        data_ablation_2 = pkl.load(f)

    config_2 = Config(os.path.join(model_path, f"single_null_layers_replacement_redundancy_analysis/version_{version_ablation_2}/config.yaml"))

    results_data_ablation_1 = data_ablation_1[1]
    results_data_ablation_2 = data_ablation_2[1]
    original_model_performance_dict = {}
    for benchmark_id, results in results_data_ablation_1.items():
        original_model_performance_dict[benchmark_id] = results.pop(("original", "original"))
        results_data_ablation_2[benchmark_id].pop(("original", "original"))
    source_paths_ablation_1, destination_paths_ablation_1, performance_arrays_ablation_1 = SingleNullLayersReplacementAnalysis.format_result_dictionary_to_plot(
        *SingleNullLayersReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_data_ablation_1), original_model_performance_dict)
    source_paths_ablation_2, destination_paths_ablation_2, performance_arrays_ablation_2 = SingleNullLayersReplacementAnalysis.format_result_dictionary_to_plot(
        *SingleNullLayersReplacementAnalysis.extract_and_sort_destination_source_labels_from_result_dictionary(
            results_data_ablation_2), original_model_performance_dict)

    for benchmark_id in results_data_ablation_1.keys():
        print(f"benchmark_id: {benchmark_id}")
        print(config_1.get("targets"))
        print(np.nanmean(performance_arrays_ablation_1[benchmark_id]))
        print(config_2.get("targets"))
        print(np.nanmean(performance_arrays_ablation_2[benchmark_id]))
"""


"""
def main():
    #model_id = "Llama-3.1-8B"
    model_id = "gemma-2-2b"
    store = False
    model_path = f"/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/{model_id}"
    version = "1"
    with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                           f"version_{version}", "storage.pkl"), "rb") as f:
        data_couples = pkl.load(f)

    with open(os.path.join(model_path, "single_null_layers_replacement_redundancy_analysis",
                           f"version_{version}", "storage.pkl"), "rb") as f:
        data_ablation = pkl.load(f)

    if "hellaswag" in data_couples[1] and store:
        data_couples[1]['hellaswag'][("original", "original")]['hellaswag']['acc_norm,none'] = 0.7296355307707628
        data_couples[1]['hellaswag'][("original", "original")]['hellaswag']['acc,none'] = 0.5501892053375822

    if "gsm8k" in data_couples[1] and store:
        data_couples[1]['gsm8k'][("original", "original")]['gsm8k']['exact_match,strict-match'] = 0.24564063684609552
        data_couples[1]['gsm8k'][("original", "original")]['gsm8k']['exact_match,flexible-extract'] = 0.24791508718726307

    for benchmark_id, results in data_couples[1].items():
        print(benchmark_id)
        for key, value in results.items():
            print(key)
            print(value)
            print()
            break
    print()

    if store:
        with open(os.path.join(model_path, "all_layer_couples_displacement_based_replacement_redundancy_analysis",
                               f"version_{version}", "storage.pkl"), "wb") as f:
            pkl.dump(data_couples, f)

    for benchmark_id, results in data_ablation[1].items():
        print(benchmark_id)
        for key, value in results.items():
            print(key)
            print(value)
            print()
            break
    print()
"""

if __name__ == "__main__":
    main()