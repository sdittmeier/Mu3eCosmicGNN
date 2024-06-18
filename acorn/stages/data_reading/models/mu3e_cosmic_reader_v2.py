import os
import numpy as np
import pandas as pd

from ..data_reading_stage import EventReader

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
    hits = hits.rename(columns={'module': 'module_id'})
    hits = hits.rename(columns={'ladder': 'ladder_id'})
    hits = hits.rename(columns={'pid': 'particle_type'})

    # Calculate the pT of the particle
    hits['pt'] = np.sqrt(hits["px"] ** 2 + hits["py"] ** 2)

    # Calculate the radius of the particle
    hits['radius'] = np.sqrt(hits["vx"] ** 2 + hits["vy"] ** 2)

    #Assign nhits
    hits['nhits'] = len(hits['x'])

    #Assign hit_ids
    hits['hit_id'] = list(range(0,len(hits['x'])))

    #Make ladder ids and module ids unique
    hits['module_id'] = hits['module_id'] + hits['ladder_id']*100 + hits['layer_id']*1000 + hits['station_id']*10000
    hits['ladder_id'] = hits['ladder_id'] + hits['layer_id']*1000 + hits['station_id']*10000
    hits['layer_id'] = hits['layer_id'] + hits['station_id']*10000
    
    #Assign charge
    hits['q'] = np.where(hits['particle_type'] > 0, 1, -1)

    return hits

class Mu3eCosmicReaderV2(EventReader):
    def __init__(self, config):
        super().__init__(config)
        
        input_file = self.config["input_file"]
        self.raw_events = pd.read_csv(input_file) # Dataframe of all events
        '''
        #Filter out electrons and positrons (pid=+-11)
        self.raw_events = self.raw_events[self.raw_events['pid'] != 11]
        self.raw_events = self.raw_events[self.raw_events['pid'] != -11]
        '''
        
        self.event_id_list = self.raw_events['event'].unique() # List of all unique eIDs

        # Split the data by 80/10/10: train/val/test -> train,val,test are lists of eIDs
        self.trainset, self.valset, self.testset = split_list(self.event_id_list, 0.8, 0.1, 0.1, self.config['seed'])

    def _build_single_csv(self, event, output_dir=None):
        raw = self.raw_events[self.raw_events['event'] == event]
        processed = process_hits(raw)

        truth = processed.loc[:, self.config['truth_features']]
        particles = processed.loc[:, self.config['particles_features']]
        particles = particles.drop_duplicates(subset='particle_id')
        
        # Check if file already exists
        if os.path.exists(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event)))
        ) and os.path.exists(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event)))
        ):
            print(f"File {event} already exists, skipping...")
            return

        # Save to CSV
        truth.to_csv(
            os.path.join(output_dir, "event{:09}-truth.csv".format(int(event))),
            index=False,
        )

        particles.to_csv(
            os.path.join(output_dir, "event{:09}-particles.csv".format(int(event))),
            index=False,
        )