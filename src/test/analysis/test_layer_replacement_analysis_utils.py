import torch

from exporch import Config
from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.analysis.layer_replacement_analysis_utils import NullLayerReplacingModelWrapper


class TestNullLayerReplacingModelWrapper:
    def setup_method(
            self
    ) -> None:
        self.model = load_model_for_causal_lm(Config.convert_to_config({"model_id": "bert-base-uncased"}))
        print(self.model)
        self.destination_layer_paths = [
            {("encoder", "0", "query"): ("null",)},
            {("encoder", "1", "query"): ("null",)},
            {("encoder", "2", "query"): ("null",)},
            {("encoder", "3", "query"): ("null",)},
            {("encoder", "4", "query"): ("null",)},
            {("encoder", "5", "query"): ("null",)},
            {("encoder", "6", "query"): ("null",)},
            {("encoder", "7", "query"): ("null",)},
            {("encoder", "8", "query"): ("null",)},
            {("encoder", "9", "query"): ("null",)},
            {("encoder", "10", "query"): ("null",)},
            {("encoder", "11", "query"): ("null",)}
        ]
        print(self.destination_layer_paths)

    def test_wrapping(self):
        null_model_wrapper = NullLayerReplacingModelWrapper(self.model, None)

        for destination_layer_path in self.destination_layer_paths:
            print(destination_layer_path)
            tolerance = 1e-6

            weight = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data
            bias = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.bias.data
            print(weight.sum())
            print(bias.sum())

            null_model_wrapper.set_destination_layer_path_source_layer_path_mapping(destination_layer_path)

            weight = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data
            bias = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.bias.data
            print(weight.sum())
            print(bias.sum())

            assert torch.all(torch.isclose(weight, torch.tensor(0.), atol=tolerance)).item()
            assert torch.all(torch.isclose(bias, torch.tensor(0.), atol=tolerance)).item()


            try:
                previous_weight = null_model_wrapper.get_model().bert.encoder.layer[
                    int(list(destination_layer_path.keys())[0][1]) - 1].attention.self.query.weight.data
                previous_bias = null_model_wrapper.get_model().bert.encoder.layer[
                    int(list(destination_layer_path.keys())[0][1]) - 1].attention.self.query.bias.data
                assert not torch.all(torch.isclose(previous_weight, torch.tensor(0.), atol=tolerance)).item()
                assert not torch.all(torch.isclose(previous_bias, torch.tensor(0.), atol=tolerance)).item()
            except IndexError as e:
                pass

            try:
                next_weight = null_model_wrapper.get_model().bert.encoder.layer[
                    int(list(destination_layer_path.keys())[0][1]) + 1].attention.self.query.weight.data
                next_bias = null_model_wrapper.get_model().bert.encoder.layer[
                    int(list(destination_layer_path.keys())[0][1]) + 1].attention.self.query.bias.data
                assert not torch.all(torch.isclose(next_weight, torch.tensor(0.), atol=tolerance)).item()
                assert not torch.all(torch.isclose(next_bias, torch.tensor(0.), atol=tolerance)).item()
            except IndexError as e:
                pass

            null_model_wrapper.reset_replacement()

            weight = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data
            bias = null_model_wrapper.get_model().bert.encoder.layer[
                int(list(destination_layer_path.keys())[0][1])].attention.self.query.bias.data

            print(weight.sum())
            print(bias.sum())

        assert True