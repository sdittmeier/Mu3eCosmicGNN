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
from itertools import product



'''
Build truth graphs stops after ~1500 graphs
    -> IndexError: arrays used as indices must be of integer (or boolean) type
    -> track_index_edges probably the error
    -> Other seeds also stop around that mark

    Reason: Event 13032 and 17761 only have hits in layer 3 -> track_index_edges empty

Event IDs weird: Some are missing -> Ask Tamasi

Next steps:
    -> Need additional module id for special cases like above
        -Add in simulation or manual?
    -> Implement nhits -> Done I think
    -> Graph visualization: -> 'GlobalStorage' object has no attribute 'edge_index'
                            -> Error in truth graph generation?
'''

def split_list(list, train_size, val_size, test_size, seed=None):
    if seed != None:
        np.random.seed(seed)
    
    np.random.shuffle(list)
    length = list.shape[0]

    print('In total',length,'events.')
    print('Split into:')
    print('Trainset:', int(train_size*length),'events')
    print('Valset:', int(val_size*length),'events')
    print('Testset:', int(test_size*length),'events')

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

    # Assign hit_ids
    hits['hit_id'] = list(range(1,len(hits['x'])+1))

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
            print('Skipping building ', dataset_name,' CSVs')
            return

        print('Building ',dataset_name,' csv files started!')

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

        print('Building ',dataset_name,' csv files completed!')

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
            print('Skipping building ', dataset_name,' PYGs')
            return

        print('Building truth graphs started!')

        dataset_dir = os.path.join(self.config["stage_dir"], dataset_name)
        graph_dir = os.path.join(dataset_dir, 'graphs')
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

    def _build_single_pyg_event(self, event, output_dir=None):
        os.sched_setaffinity(0, range(1000))

        event_id = event

        if os.path.exists(os.path.join(output_dir, "event{:09}-graph.pyg".format(int(event_id)))):
            print(f"Graph {event_id} already exists, skipping...")
            return

        dataset_dir = os.path.dirname(output_dir)
        event_path = os.path.join(dataset_dir, "csv/event{:09}-truth.csv".format(int(event_id)))

        hits = pd.read_csv(event_path)
        hits = process_hits(hits)
        
        hits, track_index_edges, flag = self._sort_radius(hits, event_id)

        #Skip this graph if track_index_edges is empty
        if flag== True:
            return
        
        #Two graph building methods: simple (connect edges based on distance to vertex) and normal (with remapping)
        if self.config['simple_graph'] == True:
            graph, positions_x_y, flag = self._build_truth_graph_simple(hits, track_index_edges)
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

        '''
        hits is a dataframe containing the entries of the raw data, the added track features, module columns
        signal is hits with added R (distance to vertex loc), filtered out noise and sorted by R

        ['particle_id']+module_columns is a list of pID and detector ids
        signal.groupby(...)['index'].agg(lambda x: list(x)) creates DataFrame of only index, pID and module IDs
            -> output is dataframe with column of all unique particle ids, then layer id then station id
            ->e.g. particle 1 hits in layer 3, station 0 twice -> row: 1,3,0, [0,1]
            ->e.g. particle 1 hits in layer 3, station 0 and station -1 ->  1,3,0,[0]
                                                                             ,3,-1,[1]
            ->e.g. particle 1 hits in layer 3, station 0 twice and once in layer 2 station 0
                ->  1,3,0,[0,3]
                    1,2,0,[1,2]

        .groupby(level=0).agg(lambda x: list(x)) groups by first level <-> pID
            ->Output is dataframe with pID column and lists of list of indices
            ->e.g. particle 1 hits in layer 3, station 0 twice and twice in layer 2 station 0
                -> 1 [[0,2],[1]]
            ->Get dataframe with rows corresponding to different particles

        signal_index_list is the final dataframe

        signal_index_list.values gives back a list of the lists in signal_index_values

        track_index_edges is list of tuples

        for row in signal_index_list.values:    -> picks a row
            for i,j in zip(row[:-1], row[1:]):  -> i:that row without last value, j: that row without first value
                track_index_edges.extend(list(product(i,j)))
        '''

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

        '''
        signal_index_list is a dataframe with rows containing the particle_id and one list for each unique combination of module_ids
        e.g. particle 58858 hits layer 3 twice (hit ids 0,2) and layer 2 (1,3) twice, all in the same station, the dataframe will be
        58858 [[0,2],[1,3]]
        '''

        track_index_edges = []
        for row in signal_index_list.values:
            for i, j in zip(row[:-1], row[1:]):
                track_index_edges.extend(list(product(i, j)))

        track_index_edges = np.array(track_index_edges).T

        '''
        track_index_edges is a list of all possible combinations of two values from signal_index_list
        e.g. signal_index_list = [[0,2],[1,3]] -> track_index_edges = [(0,1),(0,3),(2,1),(2,3)]
        '''

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