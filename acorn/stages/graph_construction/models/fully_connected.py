import os
import logging

# 3rd party imports
from ..graph_construction_stage import GraphConstructionStage

import torch
import numpy as np

from itertools import combinations
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial

class FullyConnected(GraphConstructionStage):
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        self.use_pyg = True
        self.use_csv = True

    def to(self, device):
        return self
    
    def build_graphs(self, dataset, data_name):
        output_dir = os.path.join(self.hparams['stage_dir'], data_name)
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Building graphs for {data_name}")

        max_workers = (self.hparams['max_workers'] if 'max_workers' in self.hparams else None)

        if max_workers != 1:
            print('Entered parallel mode.')
            process_map(
                partial(self._build_graph, output_dir=output_dir), 
                dataset, max_workers=max_workers, chunksize=1,
                desc=f"Building {data_name} fully connected graphs"
            )
        else:
            for event in tqdm(dataset, desc=f"Building {data_name} fully connected graphs"):
                if event[0] is None:
                    continue
                if os.path.exists(os.path.join(output_dir, f"event{event[0].event_id}.pyg")):
                    continue

                self._build_graph(dataset, output_dir=output_dir)

    def _build_graph(self, dataset, output_dir=None):
        graph = dataset[0]
        truth = dataset[2]
        graph = self._add_features(graph, truth)

        torch.save(graph, os.path.join(output_dir, f"event{graph.event_id}.pyg"))


    def _add_features(self, graph, truth):
        #Position features
        graph.node_pos = torch.tensor(truth[['x', 'y', 'z']].values)
        
        graph.r = torch.sqrt(graph.node_pos[:, 0]**2 + graph.node_pos[:, 1]**2)
        graph.phi = torch.atan2(graph.node_pos[:, 1], graph.node_pos[:, 0])
        graph.theta = torch.atan2(graph.r, graph.node_pos[:, 2])
        graph.eta = -torch.log(torch.tan(graph.theta/2))

        #Track features
        graph.num_nodes = len(graph.x)
        graph.batch = torch.zeros(graph.num_nodes)
        graph.ptr = torch.tensor([0, graph.num_nodes])

        #Edge/segment features
        fc_edges = np.array(list(combinations(truth['hit_id'],2)))
        fc_edges = self._fc_cuts(fc_edges, truth)

        graph.edge_index = torch.tensor(fc_edges.T)
        graph.y = self._truth_info(graph.edge_index, graph.track_edges)
        graph.truth_map = self._truth_mapping(graph.edge_index, graph.track_edges)

        #Append fc construction config
        graph.config.append(self.hparams)

        return graph
    
    def _truth_info(self, edge_index, track_edges):
        y = []

        for edge in edge_index.T: #check if edge is in track_edges
            found = False #reset found flag

            for segment in track_edges.T: #iterate through track_edges
                if torch.equal(edge, segment) or torch.equal(edge, segment.flip(0)):
                    y.append(True) #if edge is in track_edges, append True to y
                    found = True #change found flag to True
                    break
                
            if not found:
                y.append(False) #if edge is not in track_edges, append False to y

        return torch.tensor(y)
                
    def _truth_mapping(self, edge_index, track_edges):
        truth_map = []

        for segment in track_edges.T:
            found = False

            for index, edge in enumerate(edge_index.T):
                if torch.equal(edge, segment) or torch.equal(edge, segment.flip(0)):
                    truth_map.append(index)
                    found = True
                    break
            
            if not found:
                truth_map.append(-1)

        return torch.tensor(truth_map)
    
    def _fc_cuts(self, edges, truth):
        if self.hparams['ladder_cut']:
            edges = self._ladder_cut(edges, truth)
        if self.hparams['recurl_cut']:
            edges = self._recurl_cut(edges, truth)
        if self.hparams['distance_cut'] > 0:
            edges = self._distance_cut(edges, truth, self.hparams['distance_cut'])
        if self.hparams['angle_cut'] > 0:
            edges = self._angle_cut(edges, truth, self.hparams['angle_cut'])
        return edges
    
    def _ladder_cut(self, edges, truth):
        ladder_id_dict = truth.set_index('hit_id')['ladder_id'].to_dict()

        new_edges = []

        for edge in edges:
            ladder_id1 = ladder_id_dict.get(edge[0])
            ladder_id2 = ladder_id_dict.get(edge[1])

            if ladder_id1 != ladder_id2:
                new_edges.append(edge)

        edges = np.array(new_edges)

        return edges
    
    def _recurl_cut(self, edges, truth):
        station_id_dict = truth.set_index('hit_id')['station_id'].to_dict()

        new_edges = []

        for edge in edges:
            station_id1 = station_id_dict.get(edge[0])
            station_id2 = station_id_dict.get(edge[1])

            if not (station_id1 == 1 and station_id2 == 2):
                new_edges.append(edge)

        edges = np.array(new_edges)
    
        return edges
    
    def _distance_cut(self, edges, truth, cut_at=400):
        new_edges = []

        for edge in edges:
            hit1 = truth[truth['hit_id'] == edge[0]]
            hit2 = truth[truth['hit_id'] == edge[1]]

            distance = np.sqrt((hit1['x'].values - hit2['x'].values)**2 + (hit1['y'].values - hit2['y'].values)**2 + (hit1['z'].values - hit2['z'].values)**2)

            if distance < cut_at:
                new_edges.append(edge)

        edges = np.array(new_edges)

        return edges
    
    def _angle_cut(self, edges, truth, cut_at=np.pi):
        new_edges = []

        for edge in edges:
            hit1 = truth[truth['hit_id'] == edge[0]]
            hit2 = truth[truth['hit_id'] == edge[1]]

            x1 = hit1['x'].values
            y1 = hit1['y'].values
            z1 = hit1['z'].values
            
            x2 = hit2['x'].values
            y2 = hit2['y'].values
            z2 = hit2['z'].values  

            if y1 > y2:
                angle = np.arccos(y1-y2/np.sqrt((x1-x2)**2 + (y1-y2)**2) + (z1-z2)**2) 
            else:
                angle = np.arccos(y2-y1/np.sqrt((x1-x2)**2 + (y1-y2)**2) + (z1-z2)**2)

            if angle < cut_at:
                new_edges.append(edge)

        edges = np.array(new_edges)

        return edges