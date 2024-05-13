import os
import torch
import glob

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from itertools import product, combinations
from torch_geometric.data import Data
from torch_geometric import utils


def visualize_mu3e_graph(graph, hits): 
    '''
    Input:  PYG graph
            CSV loaded as dataframe
    Output: Information about edges and nodes
            2D plot of the graph in x-y plane with nodes at correct position in space
    '''

    print(f'Number of nodes: {graph.num_nodes}')
    print(f'Number of edges: {graph.num_edges}')
    #print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}') 
    #print(f'Has isolated nodes: {graph.has_isolated_nodes()}')
    #print(f'Has self-loops: {graph.has_self_loops()}')
    #print(f'Is undirected: {graph.is_undirected()}')
    #print('Edges:',graph.edge_index)
    
    g = utils.to_networkx(graph, to_undirected=True)

    pos_x_y = hits[['x', 'y']].T.to_dict()
    pos_x_y = {node_num: np.array([value['x'], value['y']]) for node_num, value in pos_x_y.items()}
    
    pos_y_z = hits[['y', 'z']].T.to_dict()
    pos_y_z = {node_num: np.array([value['y'], value['z']]) for node_num, value in pos_y_z.items()}

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.set_title('x-y view')
    axs.set_aspect('equal')
    axs.scatter(x=0,y=0, color='red')
    axs.grid()
    axs.set_xlim(-100,100)
    axs.set_ylim(-100,100)

    '''
    axs[1].set_title('y-z view')
    axs[1].set_aspect('equal')
    axs[1].scatter(x=0,y=0, color='red')
    axs[1].grid()
    axs[1].set_xlim(-650,650)
    axs[1].set_ylim(-100,100)
    '''
    nx.draw_networkx(g, pos=pos_x_y, node_size=200, with_labels=True, edge_color="r", alpha=0.7, ax=axs)
    #nx.draw_networkx(g, pos=pos_y_z, node_size=50, with_labels=False, edge_color="r", alpha=0.7, ax=axs[1])
    
    plt.show()

def graph_difference(graph1, graph2):
    '''
    Input:  Two PYG graphs
    Output: Graph with edges that are present in graph1 but not in graph2
    '''
    edges_1 = graph1.edge_index.cpu().numpy()
    edges_2 = graph2.edge_index.cpu().numpy()

    set1 = {tuple(x) for x in edges_1.T}
    set2 = {tuple(x) for x in edges_2.T}

    difference = set1 - set2
    difference = np.array(list(difference)).T
    difference = torch.from_numpy(difference)

    graph1.edge_index = difference

    return graph1

def event_histogram(dataframe, name):
    '''
    Input:  Full dataset CSV loaded as dataframe
    Output: Histogram of the number of hits per event
            Additional statistics: mean, std, min, max
    '''

    event_count = dataframe['event'].value_counts()

    plt.hist(event_count.tolist(), bins=40)
    plt.ylabel('# of hits per event')
    plt.yscale('log')
    plt.title('Distribution of the number of hits per event ('+name+')')

    # Calculate statistics
    mean = np.mean(event_count)
    std = np.std(event_count)
    min_val = np.min(event_count)
    max_val = np.max(event_count)

    # Create legend
    legend_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMin: {min_val}\nMax: {max_val}"
    max_edges = max_val ** 2
    legend_text += f"\nMax Edges: {round(max_edges/1000,1)}k"
    plt.legend([legend_text])
    plt.show()

def edge_cuts(feature_store, set, sample):
    '''
    Input:  Path to feature store
            Dataset name (test/train/val)
            Sample name (cosmic/michel)
    Output: Histogram of the number of edges cut per graph
            Additional statistics: mean, max
            Purity of the graphs
            Returns the mean purity with and without cuts
    '''

    path = os.path.join(feature_store, set)

    path1 = os.path.join(path, 'fc_uncut') #fc graphs without cuts
    path2 = os.path.join(path, 'fc_both_cut') #fc graphs with both cuts
    path3 = os.path.join(path, 'truth_graphs') #truth graphs
    path4 = os.path.join(path, 'fc_ladder_cut') #fc graphs with ladder cut only
    
    edges_fc = []
    edges_fc_both_cut = []    
    edges_fc_ladder_cut = []
    edges_truth = []
    
    #Read out the number of edges for each graph in each set
    #FC graphs without cuts
    for fc_graph in glob.glob(path1+'/*.pyg'):
        data1 = torch.load(fc_graph)
        edges_fc.append(data1.num_edges)

    #FC graphs with both cuts
    for fc_graph_both_cut in glob.glob(path2+'/*.pyg'):
        data2 = torch.load(fc_graph_both_cut)
        edges_fc_both_cut.append(data2.num_edges)

    #Truth graphs
    for truth_graph in glob.glob(path3+'/*.pyg'):
        data3 = torch.load(truth_graph)
        edges_truth.append(data3.num_edges)

    #FC graphs with ladder cut only
    for fc_graph_ladder_cut in glob.glob(path4+'/*.pyg'):
        data4 = torch.load(fc_graph_ladder_cut)
        edges_fc_ladder_cut.append(data4.num_edges)

    #Difference between the number of edges in the FC graphs with both cuts and the FC graphs with ladder cut only
    edge_diff = np.array(edges_fc) - np.array(edges_fc_ladder_cut)
    mean = round(np.mean(edge_diff),2)

    #Calculate the percentage of edges cut
    ratio = 100 - np.array(edges_fc_ladder_cut)/np.array(edges_fc)*100
    ratio_mean = np.mean(ratio)

    plt.hist(edge_diff, bins=40, 
             label=set+': mean # of edges:'+str(mean)+'; mean percentage:'+str(round(ratio_mean,2))+'%', 
             alpha=0.5, edgecolor='black', histtype='stepfilled')

    # Customize the plot
    plt.xlabel('Edges cut')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.legend()
    plt.title('Distribution of edges cut per graph ('+sample+', ladder cut)')
    plt.grid(True)
    plt.legend()


    print(set)
    print('Maximum amount of edges cut:', np.max(edge_diff))
    print('Maximum percentage of edges cut:', round(np.max(ratio),2),'%')
    print()

    purity_cut = np.array(edges_truth)/np.array(edges_fc_ladder_cut)*100
    purity = np.array(edges_truth)/np.array(edges_fc)*100

    purity_cut_mean = round(np.mean(purity_cut),2)
    purity_mean = round(np.mean(purity),2)

    print('Purity:', purity_mean)
    print('Purity (with cut):', purity_cut_mean)

    return purity_cut_mean, purity_mean

def get_edges(feature_store, which_graphs):
    '''
    Input:  Path to feature store
            Name of the folder with the graphs
    Output: List of the number of edges in each graph
    '''

    dir = '/mnt/data1/karres/cosmics_test'
    path = os.path.join(dir,feature_store)
    
    edges = []

    for set in ['trainset', 'valset', 'testset']:
        graph_dir = os.path.join(path, set, which_graphs)

        for graph in glob.glob(graph_dir+'/*.pyg'):
            data = torch.load(graph)
            edges.append(data.num_edges)

    return edges

def build_edge_histogram(feature_store, name):
    '''
    Input:  Path to feature store
            Name of the dataset
    Output: Histogram of the number of edges in each graph
    '''

    uncut = get_edges(feature_store, 'fc_uncut')
    ladder = get_edges(feature_store, 'fc_ladder_cut')
    both = get_edges(feature_store, 'fc_both_cut')
    truth = get_edges(feature_store, 'truth_graphs')

    plt.hist(uncut, bins=40, alpha=0.4, edgecolor='black', histtype='stepfilled', 
            label='Uncut, mean # of edges: '+str(round(np.mean(uncut),2)))
    plt.hist(ladder, bins=40, alpha=0.4, edgecolor='black', histtype='stepfilled', 
            label='Ladder cut, mean # of edges: '+str(round(np.mean(ladder),2)))
    plt.hist(both, bins=40, alpha=0.4, edgecolor='black', histtype='stepfilled', 
            label='Both cuts, mean # of edges: '+str(round(np.mean(both),2)))
    plt.hist(truth, bins=40, alpha=0.4, edgecolor='black',histtype='stepfilled', 
            label='Truth, mean # of edges: '+str(round(np.mean(truth),2)))

    plt.xlabel('Edges')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.legend()
    plt.title('Distribution of edges per graph ('+name+')')
    plt.grid(True)
    plt.show()

def distance_between_nodes(path):
    '''
    Input:  Path to a single PYG graph
    Output: List of the distances between the nodes in the graph
    '''

    event = torch.load(path)
    distances = []

    for edge in event.edge_index.T:
        node1_pos = event.x[edge[0]][1:]
        node2_pos = event.x[edge[1]][1:]

        distance = np.linalg.norm(node1_pos-node2_pos)
        distances.append(distance)

    return distances

def angle_between_nodes(path):
    '''
    Input:  Path to a single PYG graph
    Output: List of the angles between the nodes in the graph (relative to y-axis)
    '''

    event = torch.load(path)
    angles = []

    for edge in event.edge_index.T:
        node1_pos = event.x[edge[0]][1:]
        node2_pos = event.x[edge[1]][1:]

        diff_pos = node2_pos - node1_pos
        angle = np.arccos(np.abs(diff_pos[1])/np.linalg.norm(diff_pos)) - np.pi/2

        angles.append(angle)

    return angles

def get_distances_and_angles(dir, feature_store, which_graphs):
    '''
    Input:  Path to dataset directory
            Feature store
            Name of the folder with the graphs
    Output: List of distances and angles between the nodes in the graphs for the whole dataset
    '''

    path = os.path.join(dir,feature_store)
    
    distances = []
    angles = []

    for set in ['trainset', 'valset', 'testset']:
        graph_dir = os.path.join(path, set, which_graphs)

        for graph in glob.glob(graph_dir+'/*.pyg'):
            distances.extend(distance_between_nodes(graph))
            angles.extend(angle_between_nodes(graph))

    return distances, angles

def get_distances_and_angles_multi(dir, feature_store, graph_type_list):
    '''
    Input:  Path to dataset directory
            Feature store
            List of the names of the folders with the graphs
            e.g.: graph_type_list = ['fc_uncut', 'fc_both_cut']
    Output: List of lists of distances and angles between the nodes in the graphs for the whole dataset
            Each sub-list corresponds to a different graph type
    '''
    distances = []
    angles = []

    for type in graph_type_list:
        distances.append(get_distances_and_angles(dir, feature_store, type)[0])
        angles.append(get_distances_and_angles(dir, feature_store, type)[1])

    return distances, angles

def plot_distance_angle(distances, angles, names, set_name):
    '''
    Input:  List of lists of distances and angles between the nodes in the graphs
            List of names of the graph types
            Name of the dataset
    Output: Histograms of the distances and angles between the nodes in the graphs for each graph type
    '''

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for distance, name in zip(distances, names):
        axs[0].hist(distance, bins=50, alpha=0.4, edgecolor='black', histtype='stepfilled', label=name)

    for angle, name in zip(angles, names):
        axs[1].hist(angle, bins=50, alpha=0.4, edgecolor='black', histtype='stepfilled', label=name)

    axs[0].legend()
    axs[0].set_title('Distribution of distances between nodes')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Distance [mm]')
    axs[0].set_ylabel('Frequency')

    axs[1].legend()
    axs[1].set_title('Distribution of angles between nodes')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Angle [rad]')
    axs[1].set_ylabel('Frequency')

    fig.suptitle('Edge properties of the '+set_name+' dataset', fontsize=16) 
    plt.savefig('dist_ang_'+set_name+'.png', dpi=400)
    plt.show()
   

def eff_pur_single_event(path_truth, path_fc):
    '''
    Input:  Path to the truth graph and the FC graph
    Output: Efficiency and purity of the FC graph
    '''

    event_truth = torch.load(path_truth)
    event_fc = torch.load(path_fc)

    edges_truth = event_truth.edge_index.cpu().numpy()
    edges_fc = event_fc.edge_index.cpu().numpy()

    truth_set = {tuple(x) for x in edges_truth.T}
    fc_set = {tuple(x) for x in edges_fc.T}
    
    truth_set_set = {frozenset(pair) for pair in truth_set}
    fc_set_set = {frozenset(pair) for pair in fc_set}

    intersection = truth_set_set.intersection(fc_set_set)
    intersection = {tuple(pair) for pair in intersection}
    
    efficiency = len(intersection)/len(truth_set)
    purity = len(intersection)/len(fc_set)

    return efficiency, purity

def get_eff_pur(dir, feature_store, which_graphs):
    '''
    Input:  Path to dataset directory
            Feature store 
            Name of the folder with the graphs
    Output: List of efficiencies and purities for the whole dataset
    '''

    path = os.path.join(dir,feature_store)
    
    eff_list = []
    pur_list = []

    for set in ['trainset', 'valset', 'testset']:
        fc_graph_dir = os.path.join(path, set, which_graphs)
        truth_graph_dir = os.path.join(path, set, 'truth_graphs')

        fc_graphs = glob.glob(fc_graph_dir+'/*.pyg')
        truth_graphs = glob.glob(truth_graph_dir+'/*.pyg')

        for fc_graph, truth_graph in zip(fc_graphs, truth_graphs):
            eff, pur = eff_pur_single_event(truth_graph, fc_graph)
            eff_list.append(eff)
            pur_list.append(pur)

    return eff_list, pur_list

def get_eff_pur_multi(dir, feature_store, graph_type_list):
    '''
    Input:  Path to dataset directory
            Feature store
            List of the names of the folders with the graphs
            e.g.: graph_type_list = ['fc_uncut', 'fc_both_cut']
    Output: List of lists of efficiencies and purities for the whole dataset
            Each sub-list corresponds to a different graph type
    '''

    eff_list = []
    pur_list = []

    for type in graph_type_list:
        eff_list.append(get_eff_pur(dir, feature_store, type)[0])
        pur_list.append(get_eff_pur(dir, feature_store, type)[1])

    return eff_list, pur_list

def plot_eff_pur(efficiencies, purities, names, set_name):
    '''
    Input:  List of efficiencies and purities for the graphs
            List of names of the graph types
            Name of the dataset
    Output: Histograms of the efficiencies and purities for each graph type
    '''
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for eff, name in zip(efficiencies, names):
        axs[0].hist(eff, bins=50, alpha=0.4, edgecolor='black', histtype='stepfilled', label=name+ ' mean: '+str(round(np.mean(eff),2)))

    for pur, name in zip(purities, names):
        axs[1].hist(pur, bins=50, alpha=0.4, edgecolor='black', histtype='stepfilled', label=name+ ' mean: '+str(round(np.mean(pur),2)))

    axs[0].legend()
    axs[0].set_title('Distribution of efficiencies')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Efficiency')
    axs[0].set_ylabel('Frequency')

    axs[1].legend()
    axs[1].set_title('Distribution of purities')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Purity')
    axs[1].set_ylabel('Frequency')

    fig.suptitle('Efficiency and purity of the '+set_name+' dataset', fontsize=16)
    plt.savefig('eff_pur_'+set_name+'.png', dpi=400)
    plt.show()