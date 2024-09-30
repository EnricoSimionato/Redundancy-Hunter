import os
import pickle as pkl

from exporch.experiment import benchmark_id_metric_name_mapping
from exporch.utils.plot_utils import plot_heatmap
from redhunter.analysis.layer_replacement_analysis import LayerReplacementAnalysis, SingleNullLayersReplacementAnalysis


def main():
    with open("/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/Mistral-7B-v0.3/all_layer_couples_displacement_based_replacement_redundancy_analysis/version_0/storage.pkl", "rb") as f:
        data_couples = pkl.load(f)

    with open("/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results/Mistral-7B-v0.3/single_null_layers_replacement_redundancy_analysis/version_0/storage.pkl", "rb") as f:
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

    for benchmark_id in results_dict_couples.keys():
        plot_heatmap(
            [[performance_arrays_couples[benchmark_id] - performance_arrays_ablation[benchmark_id][:,1:]]],
            os.path.join("/Users/enricosimionato/Desktop/Redundancy-Hunter/src/experiments/results", f"heatmap_{benchmark_id}_couples_minus_ablation.png"),
            f"Results for the model Mistral-7B-v0.3 on the task {benchmark_id}",
            axis_titles=[f"Metric: {benchmark_id_metric_name_mapping[benchmark_id]}"],
            x_title="Overwritten layers labels",
            y_title="Duplicated layers labels",
            x_labels=[destination_paths_couples[benchmark_id]],
            y_labels=[source_paths_couples[benchmark_id]],
            cmap_str="seismic",
            fig_size=(40, 40),
            edge_color="white",
            fontsize=22,
            precision=3
        )


if __name__ == "__main__":
    main()