import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from ..data_reading_stage import EventReader
from torch_geometric.data import Data
from itertools import product, combinations



'''
Event IDs weird: Some are missing but not important -> Ask Tamasi

Wanted to add counter of edges cut for every event processed but it doesn't work
    -> List is not expanded -> csv is empty in the end
    -> Maybe because of multiprocessing?
'''

def split_list(list, train_size, val_size, test_size, seed=None):
    if seed != None:
        np.random.seed(seed)
    
    np.random.shuffle(list)
    length = list.shape[0]

    print('Full dataset:',length,'events')
    print('Split into:')
    print('Trainset:', int(train_size*length),'events')
    print('Valset:', int(val_size*length),'events')
    print('Testset:', int(test_size*length),'events')
    print('===========================================')

    trainset = list[:int(train_size*length)] 
    valset = list[int(train_size*length):int((train_size+val_size)*length)]
    testset = list[int((train_size+val_size)*length):]

    trainset = np.sort(trainset)
    valset = np.sort(valset)
    testset = np.sort(testset)

    return trainset, valset, testset

def process_hits(hits):
    #Rename detector ids and pid
    hits = hits.rename(columns={'tid': 'particle_id'})
    hits = hits.rename(columns={'station': 'station_id'})
    hits = hits.rename(columns={'layer': 'layer_id'})
    hits = hits.rename(columns={'ladder': 'ladder_id'})
    hits = hits.rename(columns={'module': 'module_id'})

    # Calculate the pT of the particle
    hits['pt'] = np.sqrt(hits["px"] ** 2 + hits["py"] ** 2)

    # Calculate the radius of the particle
    hits['radius'] = np.sqrt(hits["vx"] ** 2 + hits["vy"] ** 2)

    #Assign nhits
    hits['nhits'] = len(hits['x'])

    #Assign hit_ids
    hits['hit_id'] = list(range(0,len(hits['x'])))

    #Make ladder ids and module ids unique
    hits['ladder_id'] = hits['ladder_id'] + hits['station_id']*100
    hits['module_id'] = hits['module_id'] + hits['station_id']*1000 

    return hits

class Mu3eCosmicReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        
        input_file = self.config["input_file"]
        self.raw_events = pd.read_csv(input_file) # Dataframe of all events
        self.event_id_list = self.raw_events['event'].unique() # List of all unique eIDs

        # Split the data by 80/10/10: train/val/test -> train,val,test are lists of eIDs
        self.trainset, self.valset, self.testset = split_list(self.event_id_list, 0.8, 0.1, 0.1, self.config['seed'])

    def _build_all_csv(self, dataset, dataset_name):
        if self.config['skip_csv'] == True:
            print('Skipping building', dataset_name,'CSVs')
            return

        print('Building',dataset_name,'CSV files started!')

        dataset_dir = os.path.join(self.config["stage_dir"], dataset_name)
        csv_dir = os.path.join(dataset_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        # Build CSV files, optionally with multiprocessing
        max_workers = self.config["max_workers"] if "max_workers" in self.config else 1
        if max_workers != 1:
            process_map(
                partial(self._build_single_csv, output_dir=csv_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} CSV files",
            )
        else:
            for event in tqdm(dataset, desc=f"Building {dataset_name} CSV files"):
                self._build_single_csv(event, output_dir=csv_dir)

        print('Building',dataset_name,'CSV files completed!')

    # Build single csv files for each track
    def _build_single_csv(self, event, output_dir=None):
        # Filter data with same eID
        truth = self.raw_events[self.raw_events['event'] == event]
    
        # Check if file already exists
        if os.path.exists(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event)))
        ):
            print(f"File {event} already exists, skipping...")
            return

        # Save to CSV
        truth.to_csv(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event))),
            index=False,
        )

    # Build pyg files 
    def _build_all_pyg(self, dataset_name): #gets called by _convert_to_pyg
        if self.config['skip_pyg'] == True:
            print('Skipping building', dataset_name,'PYGs')
            return

        print('Building truth graphs started! ('+dataset_name+')')

        dataset_dir = os.path.join(self.config["stage_dir"], dataset_name)
        graph_dir = os.path.join(dataset_dir, 'truth_graphs')
        os.makedirs(graph_dir, exist_ok=True)

        csv_events = getattr(self, dataset_name)

        assert len(csv_events) > 0, "No CSV files found!"

        max_workers = (
            self.config["max_workers"] if "max_workers" in self.config else None
        )
        if max_workers != 1:
            process_map(
                partial(self._build_single_pyg_event, output_dir=graph_dir),
                csv_events,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} graphs",
            )
        else:
            for event in tqdm(csv_events, desc=f"Building {dataset_name} graphs"):
                self._build_single_pyg_event(event, output_dir=graph_dir)

        print('Building truth graphs completed!')

    def _build_single_pyg_event(self, event, output_dir=None, fc_output_dir=None):
        os.sched_setaffinity(0, range(1000))

        event_id = event
        single_pyg_dir = os.path.join(output_dir, "event{:09}-graph.pyg".format(int(event_id)))

        if os.path.exists(single_pyg_dir):
            if self.config['skip_fcg'] == True:
                print(f"Graph {event_id} already exists, skipping...")
                return
            
        dataset_dir = os.path.dirname(output_dir)
        event_path = os.path.join(dataset_dir, "csv/event{:09}-truth.csv".format(int(event_id)))

        hits = pd.read_csv(event_path)
        hits = process_hits(hits)
        
        hits, track_index_edges, flag = self._sort_radius(hits, event_id)

        #Skip this graph if track_index_edges is empty
        if flag == True:
            return
        
        #Two graph building methods: simple (connect edges based on distance to vertex) and normal (with remapping)
        if self.config['simple_graph'] == True:
            graph, positions_x_y, flag = self._build_truth_graph_simple(hits, track_index_edges) #positions_x_y are for plotting
        else:
            tracks, track_features, hits, flag = self._build_true_tracks(hits, track_index_edges) 
            graph = self._build_graph(hits, tracks, track_features)

        self._save_pyg_data(graph, output_dir, event_id)

    def _sort_radius(self, hits, event_id):
        # Sort by increasing distance from production
        hits = hits.assign(
            R=np.sqrt(
                (hits.x - hits.vx) ** 2
                + (hits.y - hits.vy) ** 2
                + (hits.z - hits.vz) ** 2
            )
        )
        signal = hits[(hits.particle_id != 0)]
        signal = signal.sort_values("R").reset_index(drop=False)

        #layer_id and station_id
        module_columns = self.config["module_columns"]

        signal_index_list = (
            signal.groupby(
                ["particle_id"] + module_columns,
                sort=False,
            )["index"]
            .agg(lambda x: list(x))
            .groupby(level=0)
            .agg(lambda x: list(x))
        )

        track_index_edges = []
        for row in signal_index_list.values:
            for i, j in zip(row[:-1], row[1:]):
                track_index_edges.extend(list(product(i, j)))

        track_index_edges = np.array(track_index_edges).T

        #Check if track_index_edges is empty -> Skip this graph if empty
        if track_index_edges.size == 0:
            print('Event', event_id, 'is a problem')
            hits = track_index_edges = None
            return hits, track_index_edges, True
        
        return hits, track_index_edges, False

    def _build_truth_graph_simple(self, hits, track_index_edges):
        '''
        This function takes in the edges of an event and outputs a pyg object containing the truth graph
        The truth graph is built by connecting succeeding hits after sorting by distance to the vertex location
        '''

        #Graph building
        feature_matrix = torch.from_numpy(hits[self.config["feature_sets"]['hit_features']].values)
        edge_index = torch.from_numpy(track_index_edges)
        graph_features = torch.from_numpy(hits[self.config["feature_sets"]['track_features']].values)

        graph=Data()
        graph.x = feature_matrix
        graph.edge_index = edge_index
        graph.y = graph_features

        #Get x,y positions for graph drawing later
        positions_x_y = hits[['x', 'y']].T.to_dict()

        positions_x_y = {node_num: np.array([value['x'], value['y']]) for node_num, value in positions_x_y.items()}

        return graph, positions_x_y, False

    def _build_true_tracks(self, hits, track_index_edges):
        #Get hit_ids <-> track_index_edges+1 due to definition of hit_id
        track_edges = hits.hit_id.values[track_index_edges]

        #Get the track features specified in config
        track_features = self._get_track_features(hits, track_index_edges, track_edges)

        # Remap
        track_edges, track_features, hits = self.remap_edges(
            track_edges, track_features, hits
        )
        
        return track_edges, track_features, hits, False
    
    def _build_all_fc_pyg(self, dataset_name):
        if self.config['skip_fcg'] == True:
            print('Skipping building ', dataset_name,' fully connected PYGs')
            return

        print('Building fully connected graphs started! ('+dataset_name+')')

        dataset_dir = os.path.join(self.config["stage_dir"], dataset_name)
        fc_graph_dir = os.path.join(dataset_dir, 'fc_graphs_ladder_cut')
        os.makedirs(fc_graph_dir, exist_ok=True)

        csv_events = getattr(self, dataset_name)

        assert len(csv_events) > 0, "No CSV files found!"

        max_workers = (
            self.config["max_workers"] if "max_workers" in self.config else None
        )
        if max_workers != 1:
            process_map(
                partial(self._build_single_fc_pyg_event, output_dir=fc_graph_dir),
                csv_events,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} fully connected graphs",
            )
        else:
            for event in tqdm(csv_events, desc=f"Building {dataset_name} fully connected graphs"):
                self._build_single_fc_pyg_event(event, output_dir=fc_graph_dir)

        print('Building fully connected graphs completed!')
        
    def _build_single_fc_pyg_event(self, event, output_dir=None):
        os.sched_setaffinity(0, range(1000))

        event_id = event
        single_fc_pyg_dir = os.path.join(output_dir, "event{:09}-fc_graph.pyg".format(int(event_id)))

        if os.path.exists(single_fc_pyg_dir):
            print(f"FC Graph {event_id} already exists, skipping...")
            return

        dataset_dir = os.path.dirname(output_dir)
        event_path = os.path.join(dataset_dir, "csv/event{:09}-truth.csv".format(int(event_id)))

        hits = pd.read_csv(event_path)
        hits = process_hits(hits)

        fc_graph, positions_x_y = self._build_fully_connected_graph(hits)

        self._save_fc_pyg_data(fc_graph, output_dir, event_id)

    def _build_fully_connected_graph(self, hits):
        #All unique combinations of hit_ids -> Fully connected graph undirected
        fc_edges = np.array(list(combinations(hits['hit_id'],2)))
        fc_edges = self._fc_cuts(fc_edges, hits)

        #Graph building
        feature_matrix = torch.from_numpy(hits[self.config["feature_sets"]['hit_features']].values)
        edge_index = torch.from_numpy(fc_edges.T)
        graph_features = torch.from_numpy(hits[self.config["feature_sets"]['track_features']].values)

        fc_graph=Data()
        fc_graph.x = feature_matrix
        fc_graph.edge_index = edge_index
        fc_graph.y = graph_features

        positions_x_y = hits[['x', 'y']].T.to_dict()

        positions_x_y = {node_num: np.array([value['x'], value['y']]) for node_num, value in positions_x_y.items()}

        return fc_graph, positions_x_y
    
    def _fc_cuts(self, fc_edges, hits):
        # Ladder cut
        ladder_id_dict = hits.set_index('hit_id')['ladder_id'].to_dict()

        new_fc_edges = []

        for edge in fc_edges:
            ladder_id1 = ladder_id_dict.get(edge[0])
            ladder_id2 = ladder_id_dict.get(edge[1])

            if ladder_id1 != ladder_id2:
                new_fc_edges.append(edge)

        fc_edges = np.array(new_fc_edges)
        '''
        #Recurl-recurl cut
        station_id_dict = hits.set_index('hit_id')['station_id'].to_dict()

        new_fc_edges = []

        for edge in fc_edges:
            station_id1 = station_id_dict.get(edge[0])
            station_id2 = station_id_dict.get(edge[1])

            if not (station_id1 == 1 and station_id2 == 2):
                new_fc_edges.append(edge)

        fc_edges = np.array(new_fc_edges)
        '''
        return fc_edges
    
    def _save_fc_pyg_data(self, graph, output_dir, event_id):
        torch.save(graph, os.path.join(output_dir, "event{:09}-fc_graph.pyg".format(int(event_id))))