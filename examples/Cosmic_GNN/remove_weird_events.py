#! /bin/env python
import torch
import glob
import shutil
import os
import pandas as pd

#Find problematic events
input_dir = '/mnt/data1/tkar/cosmic_new/fully_connected_graphs/cosmic_michel_f3/aa/'

i = 0
weird_dir= input_dir+'weird_events/'
for dataset in ['trainset', 'valset', 'testset']:
    weird_sub_dir = weird_dir+dataset+'/'
    os.makedirs(weird_sub_dir, exist_ok=True)

    for graph_dir in glob.glob(input_dir+dataset+'/*.pyg'):
        graph = torch.load(graph_dir)

        if graph.x.shape[0] == graph.track_edges.shape[1]:
            i += 1
            shutil.move(graph_dir, weird_sub_dir)
print(i)