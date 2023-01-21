"""
Wrapper for the Causical algorithm for causal discovery with a non-linear
additve SCM.
"""
from dodiscover.context import Context

from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch

import tempfile

torch.set_default_dtype(torch.float32)


class DefaultModelOptions(NamedTuple):
    base_distribution_type: Literal["gaussian", "spline"] = "spline"
    spline_bins: int = 8
    imputation: bool = False
    lambda_dag: float = 100.0
    lambda_sparse: float = 5.0
    tau_gumbel: float = 1.0
    var_dist_A_mode: Literal["simple", "enco", "true", "three"] = "three"
    imputer_layer_sizes: Optional[List[int]] = None
    mode_adjacency: Literal["upper", "lower", "learn"] = "learn"
    norm_layers: bool = True
    res_connection: bool = True
    encoder_layer_sizes: Optional[List[int]] = [32, 32]
    decoder_layer_sizes: Optional[List[int]] = [32, 32]
    cate_rff_n_features: int = 3000
    cate_rff_lengthscale: Union[int, float, List[float], Tuple[float, float]] = 1


class DeciTrainingOptions(NamedTuple):
    learning_rate: float = 3e-2
    batch_size: int = 512
    standardize_data_mean: bool = False
    standardize_data_std: bool = False
    rho: float = 10.0
    safety_rho: float = 1e13
    alpha: float = 0.0
    safety_alpha: float = 1e13
    tol_dag: float = 1e-3
    progress_rate: float = 0.25
    max_steps_auglag: int = 20
    max_auglag_inner_epochs: int = 1000
    max_p_train_dropout: float = 0.25
    reconstruction_loss_factor: float = 1.0
    anneal_entropy: Literal["linear", "noanneal"] = "noanneal"
    device: Literal["cpu", "gpu"] = "cpu"


class DECI:
    def __init__(self, model_params: dict):
        full_model_options = DefaultModelOptions()._asdict()
        full_model_options.update(model_params)
        self.full_model_options = full_model_options

    def fit(
        self,
        data: pd.DataFrame,
        context: Context,
        training_options: dict,
    ):
        """
        To speed up training you can try:
          increasing learning_rate
          increasing batch_size (reduces noise when using higher learning rate)
          decreasing max_steps_auglag (go as low as you can and still get a DAG)
          decreasing max_auglag_inner_epochs
        """
        from causica.datasets.dataset import Dataset
        from causica.datasets.variables import Variables
        from causica.models.deci.deci import DECI

        def _build_causica_dataset(self, data: pd.DataFrame) -> Dataset:
            self._encode_categorical_as_integers()
            numpy_data = self._prepared_data.to_numpy()
            data_mask = np.ones(numpy_data.shape)

            _causal_var_nature_to_causica_var_type = {
                "Discrete": "continuous",  # TODO: make categorical
                "Continuous": "continuous",
                "Categorical Ordinal": "continuous",  # TODO: make categorical
                "Categorical Nominal": "continuous",  # TODO: make categorical
                "Binary": "binary",
                "Excluded": "continuous",
            }

            variables = Variables.create_from_data_and_dict(
                numpy_data,
                data_mask,
                {
                    "variables": [
                        {
                            "name": name,
                            # TODO: this is currently mapping categorical to continuous
                            #       need to update the to properly handle
                            #       one-hot encoded values
                            "type": _causal_var_nature_to_causica_var_type.get(
                                self._nature_by_variable[name], "continuous"
                            ),
                            "lower": self._prepared_data[name].min(),
                            "upper": self._prepared_data[name].max(),
                        }
                        for name in self._prepared_data.columns
                    ]
                },
            )
            dataset = Dataset(train_data=numpy_data, train_mask=data_mask, variables=variables)
            return dataset

        def _build_model(variables: Dataset) -> DECI:
            """
            TODO: modify for constraints
            TODO: modify for interventions
            """
            with tempfile.TemporaryDirectory() as tmpdirname:
                deci_model = DECI.create(
                    model_id="DoDiscoverCausica",
                    save_dir=tmpdirname,
                    variables=variables,
                    model_config_dict=self.full_model_options,
                )
            return deci_model

        def _format_parameters(training_options):
            full_training_options_dict = DeciTrainingOptions()._asdict()
            full_training_options_dict.update(training_options)
            device = full_training_options_dict["device"]
            del full_training_options_dict["device"]
            return full_training_options_dict, device

        variables = _build_causica_dataset(data, context)
        if data.columns.size == 1:
            return nx.empty_graphs(n=list(data.columns))

        deci_model = _build_model(variables)

        full_training_options_dict, device = _format_parameters(training_options)

        deci_model.run_train(
            variables=variables, model_config_dict=full_training_options_dict, device=device
        )

        name_dict = {i: var.name for i, var in enumerate(variables)}

        dag = nx.relabel_nodes(deci_model.networkx_graph(), name_dict, copy=False)

        return dag
