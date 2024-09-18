# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Union
import warnings
import torch
import logging
from torch_geometric.data import Data as PygData
from pathlib import Path
import pandas as pd

from .mapping_utils import get_condition_lambda, map_tensor_handler, remap_from_mask
from .version_utils import get_pyg_data_keys


def load_datafiles_in_dir(input_dir, data_name=None, data_num=None):
    if data_name is not None:
        input_dir = os.path.join(input_dir, data_name)

    data_files = [str(path) for path in Path(input_dir).rglob("*.pyg")][:data_num]
    if len(data_files) == 0:
        warnings.warn(f"No data files found in {input_dir}")
    if data_num is not None:
        assert len(data_files) == data_num, (
            f"Number of data files found ({len(data_files)}) is less than the number"
            f" requested ({data_num})"
        )

    return data_files


def load_dataset_from_dir(input_dir, data_name, data_num):
    """
    Load in the PyG Data dataset from the data directory.
    """
    data_files = load_datafiles_in_dir(input_dir, data_name, data_num)

    return [torch.load(f, map_location="cpu") for f in data_files]


def run_data_tests(datasets: List, required_features, optional_features):
    for dataset in datasets:
        sample_event = dataset[0]
        assert sample_event is not None, "No data loaded"
        # Check that the event is the latest PyG format
        try:
            _ = len(sample_event)
        except RuntimeError:
            warnings.warn(
                "Data is not in the latest PyG format, so will be converted on-the-fly."
                " Consider re-saving the data in latest PyG Data type."
            )
            if dataset is not None:
                for i, event in enumerate(dataset):
                    dataset[i] = convert_to_latest_pyg_format(event)

        for feature in required_features:
            assert feature in sample_event or f"x_{feature}" in sample_event, (
                f"Feature [{feature}] not found in data, this is REQUIRED. Features"
                f" found: {get_pyg_data_keys(sample_event)}"
            )

        missing_optional_features = [
            feature
            for feature in optional_features
            if feature not in sample_event or f"x_{feature}" not in sample_event
        ]
        for feature in missing_optional_features:
            warnings.warn(f"OPTIONAL feature [{feature}] not found in data")

        # Check that the number of nodes is compatible with the edge indexing
        if "edge_index" in get_pyg_data_keys(sample_event) and "x" in get_pyg_data_keys(
            sample_event
        ):
            assert (
                sample_event.x.shape[0] >= sample_event.edge_index.max().item() + 1
            ), (
                "Number of nodes is not compatible with the edge indexing. Possibly an"
                " earlier stage has removed nodes, but not updated the edge indexing."
            )


def convert_to_latest_pyg_format(event):
    """
    Convert the data to the latest PyG format.
    """
    return PygData.from_dict(event.__dict__)

#import shutil

def handle_weighting(event, weighting_config):
    """
    Take the specification of the weighting and convert this into float values. The default is:
    - True edges have weight 1.0
    - Negative edges have weight 1.0

    The weighting_config can be used to change this behaviour. For example, we might up-weight target particles - that is edges that pass:
    - y == 1
    - primary == True
    - pt > 1 GeV
    - etc. As desired.

    We can also down-weight (i.e. mask) edges that are true, but not of interest. For example, we might mask:
    - y == 1
    - primary == False
    - pt < 1 GeV
    - etc. As desired.
    """

    # Set the default values, which will be overwritten if specified in the config
    weights = torch.zeros_like(event.y, dtype=torch.float)
    weights[event.y == 0] = 1.0

    for weight_spec in weighting_config:
        weight_val = weight_spec["weight"]
        '''
        if get_weight_mask(event, weight_spec["conditions"]).dim() != 1:
            print('Event:', event.event_id)
            src = '/mnt/data1/karres/cosmics_test/fully_connected_cosmic_michel/trainset/event'+str(event.event_id)+'.pyg'
            dst = '/mnt/data1/karres/cosmics_test/fully_connected_cosmic_michel/weird_events/event'+str(event.event_id)+'.pyg'
            shutil.move(src,dst)
            continue
        
        print(weights)
        print(weights[get_weight_mask(event, weight_spec["conditions"])])
        '''
        weights[get_weight_mask(event, weight_spec["conditions"])] = weight_val

    return weights


def handle_hard_node_cuts(
    event: Union[PygData, pd.DataFrame], hard_cuts_config: dict, passing_hit_ids=None
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    if isinstance(event, PygData):
        return handle_hard_node_cuts_pyg(event, hard_cuts_config)
    elif isinstance(event, pd.DataFrame):
        return handle_hard_node_cuts_pandas(event, hard_cuts_config, passing_hit_ids)
    else:
        raise ValueError(f"Data type {type(event)} not recognised.")


def handle_hard_node_cuts_pyg(event: PygData, hard_cuts_config: dict):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    Remap the track_edges to the new node list.
    """
    node_like_feature = [
        event[feature]
        for feature in get_pyg_data_keys(event)
        if event.is_node_attr(feature)
    ][0]
    node_mask = torch.ones_like(node_like_feature, dtype=torch.bool)

    # TODO: Refactor this to simply trim the true tracks and check which nodes are in the true tracks
    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        node_val_mask = map_tensor_handler(
            value_mask,
            output_type="node-like",
            track_edges=event.track_edges,
            num_nodes=node_like_feature.shape[0],
            num_track_edges=event.track_edges.shape[1],
        )
        node_mask = node_mask * node_val_mask

    logging.info(
        f"Masking the following number of nodes with the HARD CUT: {node_mask.sum()} /"
        f" {node_mask.shape[0]}"
    )

    num_nodes = event.num_nodes
    for feature in get_pyg_data_keys(event):
        if (
            isinstance(event[feature], torch.Tensor)
            and event[feature].shape
            and event[feature].shape[0] == num_nodes
        ):
            event[feature] = event[feature][node_mask]

    num_tracks = event.track_edges.shape[1]
    track_mask = node_mask[event.track_edges].all(0)
    node_lookup = torch.cumsum(node_mask, dim=0) - 1
    for feature in get_pyg_data_keys(event):
        if (
            isinstance(event[feature], torch.Tensor)
            and event[feature].shape
            and event[feature].shape[-1] == num_tracks
        ):
            event[feature] = event[feature][..., track_mask]

    event.track_edges = node_lookup[event.track_edges]
    event.num_nodes = node_mask.sum()

    return event


def handle_hard_node_cuts_pandas(
    event: pd.DataFrame, hard_cuts_config: dict, passing_hit_ids=None
):
    """
    Given set of cut config, remove nodes that do not pass the cuts.
    If passing_hit_ids is provided, only the hits with the given ids will be kept.
    """

    if passing_hit_ids is not None:
        event = event[event.hit_id.isin(passing_hit_ids)]
        return event

    # Create a temporary DataFrame with the same structure as the PyTorch event
    temp_event = {
        col: torch.tensor(event[col].values)
        for col in event.columns
        if pd.api.types.is_numeric_dtype(event[col])
        or pd.api.types.is_bool_dtype(event[col])
    }

    # Initialize the mask with all True values
    value_mask = torch.ones(len(event), dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert (
            condition_key in event.columns
        ), f"Condition key {condition_key} not found in event keys {event.columns}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)

        # Apply the condition lambda to the temporary event and update the mask
        value_mask &= condition_lambda(temp_event)

    # Convert the mask back to a Pandas-compatible format
    value_mask = value_mask.numpy().astype(bool)

    event = event[value_mask]

    return event


def handle_hard_cuts(event: PygData, hard_cuts_config: dict):
    """
    Given set of cut config, remove edges that do not pass the cuts.
    """
    true_track_mask = torch.ones_like(event.truth_map, dtype=torch.bool)

    for condition_key, condition_val in hard_cuts_config.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        true_track_mask = true_track_mask * value_mask

    graph_mask = torch.isin(
        event.edge_index, event.track_edges[:, true_track_mask]
    ).all(0)
    remap_from_mask(event, graph_mask)

    num_edges = event.edge_index.shape[1]
    for edge_key in get_pyg_data_keys(event):
        if (
            isinstance(event[edge_key], torch.Tensor)
            and num_edges in event[edge_key].shape
        ):
            event[edge_key] = event[edge_key][..., graph_mask]

    num_track_edges = event.track_edges.shape[1]
    for track_feature in get_pyg_data_keys(event):
        if (
            isinstance(event[track_feature], torch.Tensor)
            and num_track_edges in event[track_feature].shape
        ):
            event[track_feature] = event[track_feature][..., true_track_mask]


def reset_angle(angles):
    angles[angles > torch.pi] = angles[angles > torch.pi] - 2 * torch.pi
    angles[angles < -torch.pi] = angles[angles < -torch.pi] + 2 * torch.pi
    return angles


def handle_edge_features(event, edge_features):
    src, dst = event.edge_index

    for edge_feature in edge_features:
        if "dr" in edge_features and not ("dr" in get_pyg_data_keys(event)):
            event.dr = event.r[dst] - event.r[src]
        if "dphi" in edge_features and not ("dphi" in get_pyg_data_keys(event)):
            event.dphi = (
                reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            )
        if "dz" in edge_features and not ("dz" in get_pyg_data_keys(event)):
            event.dz = event.z[dst] - event.z[src]
        if "deta" in edge_features and not ("deta" in get_pyg_data_keys(event)):
            event.deta = event.eta[dst] - event.eta[src]
        if "phislope" in edge_features and not ("phislope" in get_pyg_data_keys(event)):
            dr = event.r[dst] - event.r[src]
            dphi = reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            phislope = dphi / dr
            event.phislope = phislope
        if "phislope" in edge_features:
            event.phislope = torch.nan_to_num(
                event.phislope, nan=0.0, posinf=100, neginf=-100
            )
            event.phislope = torch.clamp(event.phislope, -100, 100)
        if "rphislope" in edge_features and not (
            "rphislope" in get_pyg_data_keys(event)
        ):
            r_ = (event.r[dst] + event.r[src]) / 2.0
            dr = event.r[dst] - event.r[src]
            dphi = reset_angle((event.phi[dst] - event.phi[src]) * torch.pi) / torch.pi
            phislope = dphi / dr
            phislope = torch.nan_to_num(phislope, nan=0.0, posinf=100, neginf=-100)
            phislope = torch.clamp(phislope, -100, 100)
            rphislope = torch.multiply(r_, phislope)
            event.rphislope = rphislope  # features / norm / pre_proc once
        if "rphislope" in edge_features:
            event.rphislope = torch.nan_to_num(event.rphislope, nan=0.0)


def get_weight_mask(event, weight_conditions):
    graph_mask = torch.ones_like(event.y)

    for condition_key, condition_val in weight_conditions.items():
        assert condition_key in get_pyg_data_keys(
            event
        ), f"Condition key {condition_key} not found in event keys {get_pyg_data_keys(event)}"
        condition_lambda = get_condition_lambda(condition_key, condition_val)
        value_mask = condition_lambda(event)
        #print('Condition_key: ',condition_key)
        #print('Condition_val: ',condition_val)
        #print('Condition_lambda: ',condition_lambda)
        #print('Value_mask: ',value_mask)
        graph_mask = graph_mask * map_tensor_handler(
            value_mask,
            output_type="edge-like",
            num_nodes=event.num_nodes,
            edge_index=event.edge_index,
            truth_map=event.truth_map,
        )
    return graph_mask
