import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from ..data_reading_stage import EventReader

from itertools import product

'''
To Do:  -Create truth graphs
            ->Rewrite _build_single_pyg
            ->Check if _build_all_pyg needs modifications
'''

def split_list(list, train_size, val_size, test_size):
    #seed hinzufügen
    np.random.shuffle(list)
    length = list.shape[0]

    trainset = list[:int(train_size*length)] 
    valset = list[int(train_size*length):int((train_size+val_size)*length)]
    testset = list[int((train_size+val_size)*length):]

    return trainset, valset, testset

class Mu3eCosmicReader(EventReader):
    def __init__(self, config):
        super().__init__(config)
        
        input_file = self.config["input_file"]
        self.raw_events = pd.read_csv(input_file) # Dataframe of all events
        self.event_id_list = self.raw_events['event'].unique() # List of all unique eIDs

        # Split the data by 80/10/10: train/val/test -> train,val,test are lists of eIDs
        self.trainset, self.valset, self.testset = split_list(self.event_id_list, 0.8, 0.1, 0.1)
    
    def _build_all_csv(self, dataset, dataset_name):
        if self.config['skip_csv'] == True:
            return

        print('Building train, val and test csv files started!')

        output_dir = os.path.join(self.config["stage_dir"], dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Build CSV files, optionally with multiprocessing
        max_workers = self.config["max_workers"] if "max_workers" in self.config else 1
        if max_workers != 1:
            process_map(
                partial(self._build_single_csv, output_dir=output_dir),
                dataset,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} CSV files",
            )
        else:
            for event in tqdm(dataset, desc=f"Building {dataset_name} CSV files"):
                self._build_single_csv(event, output_dir=output_dir)

        print('Building train, val and test csv files completed!')

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
        print('Building truth graphs started!')

        stage_dir = os.path.join(self.config["stage_dir"], dataset_name)
        csv_events = getattr(self, dataset_name)

        assert len(csv_events) > 0, "No CSV files found!"

        max_workers = (
            self.config["max_workers"] if "max_workers" in self.config else None
        )
        if max_workers != 1:
            process_map(
                partial(self._build_single_pyg_event, output_dir=stage_dir),
                csv_events,
                max_workers=max_workers,
                chunksize=1,
                desc=f"Building {dataset_name} graphs",
            )
        else:
            for event in tqdm(csv_events, desc=f"Building {dataset_name} graphs"):
                self._build_single_pyg_event(event, output_dir=stage_dir)

        print('Building truth graphs completed!')

    def _build_single_pyg_event(self, event, output_dir=None):
        os.sched_setaffinity(0, range(1000))

        event_id = event

        if os.path.exists(os.path.join(output_dir, f"event{event_id}-graph.pyg")):
            print(f"Graph {event_id} already exists, skipping...")
            return

        event_path = os.path.join(output_dir, "event{:09}-truth.csv".format(int(event)))
        hits = pd.read_csv(event_path)
        hits = hits.rename(columns={'tid': 'particle_id'})
        hits = hits.assign(hit_id=list(range(1,len(hits['x'])+1)))

        #Find hit in event with largest y and use this position as vertex location -> Maybe requires redo for more complex events
        max_y_hit = hits.iloc[hits['y'].idxmax()]
        mother_x = max_y_hit['x']
        mother_y = max_y_hit['y']
        mother_z = max_y_hit['z']
        hits = hits.assign(vx=mother_x, vy=mother_y, vz=mother_z)

        #Need to preprocess s.t. hits contains pid, hid, x,y,z,vx,vy,vz
        # -> vertex position = x,y,z of mother particle? -> ask tamasi
        #extra particles files
        # -> used row number as hit_id -> correct?
        tracks, track_features, hits = self._build_true_tracks(hits) 
        '''
        Problems: Unsure what happens at/after grouping phase
            What is signal_index_list
        '''

        #Further processing -> Not yet implemented (needed at all?)
        hits, particles, tracks = self._custom_processing(hits, particles, tracks)

        graph = self._build_graph(hits, tracks, track_features, event_id)
        self._save_pyg_data(graph, output_dir, event_id)

    def _build_true_tracks(self, hits):
        assert all(
            col in hits.columns
            for col in ["particle_id", "hit_id", "x", "y", "z", "vx", "vy", "vz"]
        ), (
            "Need to add (particle_id, hit_id), (x,y,z) and (vx,vy,vz) features to hits"
            " dataframe in custom EventReader class"
        )

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

        #What does this do?
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
        track_edges = hits.hit_id.values[track_index_edges]

        assert (
            hits[hits.hit_id.isin(track_edges.flatten())].particle_id == 0
        ).sum() == 0, "There are hits in the track edges that are noise"

        track_features = self._get_track_features(hits, track_index_edges, track_edges)

        # Remap
        track_edges, track_features, hits = self.remap_edges(
            track_edges, track_features, hits
        )

        return track_edges, track_features, hits