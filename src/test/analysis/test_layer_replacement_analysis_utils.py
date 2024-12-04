import torch

from exporch import Config
from exporch.utils.causal_language_modeling import load_model_for_causal_lm

from redhunter.utils.layer_replacement_wrapper.layer_replacement_wrapper import LayerReplacingModelWrapper, NullLayerReplacingModelWrapper


class TestLayerReplacingModelWrapper:
    def setup_method(
            self
    ) -> None:
        self.model = load_model_for_causal_lm(Config.convert_to_config({"model_id": "bert-base-uncased"}))
        self.num_layers = 12
        print(self.model)

    def test_wrapping(self):
        tolerance = 1e-6
        self.destination_layer_paths = [
            {("encoder", f"{i}", "query"): ("encoder", f"{i+1}", "query")} for i in range(self.num_layers - 1)
        ]

        model_wrapper = LayerReplacingModelWrapper(self.model, None)

        for destination_layer_path in self.destination_layer_paths:
            model_wrapper.set_destination_layer_path_source_layer_path_mapping(destination_layer_path)

            assert torch.all(torch.isclose(
                model_wrapper.get_model().bert.encoder.layer[int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data,
                model_wrapper.get_model().bert.encoder.layer[int(list(destination_layer_path.values())[0][1])].attention.self.query.weight.data,
                atol=tolerance
            )).item()

            for i in range(self.num_layers):
                if i != int(list(destination_layer_path.keys())[0][1]) and i != int(list(destination_layer_path.values())[0][1]):
                    assert not torch.all(torch.isclose(
                        model_wrapper.get_model().bert.encoder.layer[i].attention.self.query.weight.data,
                        model_wrapper.get_model().bert.encoder.layer[int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data,
                        atol=tolerance
                    )).item()

            model_wrapper.reset_replacement()

    def test_wrapping_with_weights(self):
        tolerance = 1e-6
        self.layer = torch.nn.Linear(768, 768)
        self.layer.weight.data = torch.ones(768, 768)
        print(self.layer.weight.data)
        self.destination_layer_paths = [
            {("encoder", f"{i}", "query"): self.layer} for i in range(self.num_layers - 1)
        ]

        model_wrapper = LayerReplacingModelWrapper(self.model, None)

        for destination_layer_path in self.destination_layer_paths:
            model_wrapper.set_destination_layer_path_source_layer_path_mapping(destination_layer_path)

            assert torch.all(torch.isclose(
                model_wrapper.get_model().bert.encoder.layer[int(list(destination_layer_path.keys())[0][1])].attention.self.query.weight.data,
                self.layer.weight.data,
                atol=tolerance
            )).item()

            for i in range(self.num_layers):
                if i != int(list(destination_layer_path.keys())[0][1]):
                    assert not torch.all(torch.isclose(
                        model_wrapper.get_model().bert.encoder.layer[i].attention.self.query.weight.data,
                        self.layer.weight.data,
                        atol=tolerance
                    )).item()

            model_wrapper.reset_replacement()


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
