import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def hit_id_to_particle_id_map(graph):
    print('Building map')
    track_edges = graph.track_edges
    particle_types = graph.particle_type

    hit_to_particle_map = {}

    for edge, pid in zip(track_edges.T, particle_types):
        for hit_id in edge:
            hit_to_particle_map[hit_id] = pid
        
        missing_keys = set(range(len(graph.x) + 1)) - hit_to_particle_map.keys()
        if len(missing_keys) > 0:
            print('Error: Hit id not found in hit_to_particle_map')
            print('Setting missing hit ids to 0')
            print('Missing hit ids:', missing_keys)

            for key in missing_keys:
                hit_to_particle_map[key] = 0


    return hit_to_particle_map

def get_particle_id(hit_id, hit_to_particle_map):
    return hit_to_particle_map.get(hit_id, 11)
    
    #If hit id not found in hit_to_particle_map return 11 (electron)
    #Map based on truth graph -> Hit id not found <=> This particle only has 1 hit -> probably electron

def get_num_hits(graph):
    hit_to_particle_map = hit_id_to_particle_id_map(graph)

    num_muon_hits = 0

    for hit_id in hit_to_particle_map.keys():
        if hit_to_particle_map[hit_id]**2 == 13**2:
            num_muon_hits += 1

    num_electron_hits = graph.x.shape[0] - num_muon_hits

    return num_muon_hits, num_electron_hits

def remove_particle_edges(edges_list, particle_type, map):
    '''
    Input: edge_index or track_edges, |pdg ID| to be kept and mapping from hit id to pdg ID
    Output: edge_index or track_edges without edges corresponding to the specified pdg ID
    '''
    remove = []
    for index, edge in enumerate(edges_list.T):
        pid1 = get_particle_id(edge[0].item(), map)
        pid2 = get_particle_id(edge[1].item(), map)

        if pid1**2 == particle_type**2 or pid2**2 == particle_type**2: #keeps all edges not containing any particle_type hits
            remove.append(index)
        
    edges_list = np.delete(edges_list.T, remove, axis=0).T
    return edges_list

def sort_and_convert_to_set(edges_list):
    sorted_edges = torch.sort(edges_list.T, dim=1)[0]
    set_edges = set(tuple(row.tolist()) for row in sorted_edges)
    return set_edges

def convert_to_tensor(set_edges):
    return torch.tensor(list(set_edges)).T

def get_num_edges(graph):
    edge_index = graph.edge_index
    track_edges = graph.track_edges

    hit_to_particle_map = hit_id_to_particle_id_map(graph)
        
    true_muon_edges = remove_particle_edges(track_edges, 11, hit_to_particle_map) #remove all edges from track_edges containing electron hits -> leave truth muons
    num_truth_muon_edges = true_muon_edges.shape[1]

    #Remove truth muon edges from edge_index
    true_muon_edges_set = sort_and_convert_to_set(true_muon_edges)
    edge_index_set = sort_and_convert_to_set(edge_index)

    background_edges_set = edge_index_set - true_muon_edges_set
    background_edges = convert_to_tensor(background_edges_set)
    num_background_edges = background_edges.shape[1]

    return num_truth_muon_edges, num_background_edges

#Graph level control plots: num_nodes, num_edges

def compute_hits_edges_stats(dir, samplesize):

    muon_hits = []
    electron_hits = []

    truth_muon_edges = []
    fake_muon_edges = []
    background_edges = []

    i=0

    for graph in glob.glob(dir):
        graph = torch.load(graph)

        num_muon_hits, num_electron_hits = get_num_hits(graph)
        truth_muon, background = get_num_edges(graph)

        muon_hits.append(num_muon_hits)
        electron_hits.append(num_electron_hits)

        truth_muon_edges.append(truth_muon)
        background_edges.append(background)

        if i >samplesize:
            break

        i+=1
    return muon_hits, electron_hits, truth_muon_edges, background_edges

def plot_hits_edge_stats(dir, samplesize, titles=['Cosmics with Michel', 'Cosmics with Michel, fully connected'], filename='num_hits_edges'):
    '''
    Creates plots of the number of hits and edges in 'samplesize' graphs from the directory 'dir'
    Optional: titles for the plots and filename for saving the plots
    '''

    muon_hits, electron_hits, truth_muon_edges, background_edges = compute_hits_edges_stats(dir, samplesize)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(muon_hits, bins=np.logspace(0,2,50), edgecolor='black', linewidth=1.2, histtype='step', label='Muon hits')
    axs[0].hist(electron_hits, bins=np.logspace(0, 3, 50), edgecolor='red', linewidth=1.2, histtype='step', label='Electron hits')
    axs[0].set_xlabel('Number of hits')
    axs[0].set_ylabel('Frequency')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].legend()
    axs[0].set_title('Distribution of hits ('+titles[0]+')')
    axs[0].grid(True)

    axs[1].hist(truth_muon_edges, bins=np.logspace(0,2,50), edgecolor='black', linewidth=1.2, histtype='step', label='Signal edges (true muon edges)')
    axs[1].hist(background_edges, bins=np.logspace(0,6,50), edgecolor='red', linewidth=1.2, histtype='step', label='Background edges')
    axs[1].set_xlabel('Number of edges')
    axs[1].set_ylabel('Frequency')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].legend()
    axs[1].set_title('Distribution of edges ('+titles[1]+')')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('/home/mue/karres/git/Mu3eCosmicGNN/examples/Cosmic_GNN/control_plots/'+filename+'.png', dpi=400)
    plt.show()

    print('Mean number of muon hits:', np.mean(muon_hits))
    print('Mean number of electron hits:', np.mean(electron_hits))
    print('Mean number of truth muon edges:', np.mean(truth_muon_edges))
    print('Mean number of background edges:', np.mean(background_edges))

#Edge level control plots: angle and distance between hits

def edge_distance_angle(graph):
    edge_index = graph.edge_index
    track_edges = graph.track_edges

    truth_muon_distances = []
    truth_muon_angles = []
    
    background_distances = []
    background_angles = []

    hit_to_particle_map = hit_id_to_particle_id_map(graph)

    true_muon_edges = remove_particle_edges(track_edges, 11, hit_to_particle_map)

    truth_muon_set = sort_and_convert_to_set(true_muon_edges)
    edge_index_set = sort_and_convert_to_set(edge_index)

    background_set = edge_index_set - truth_muon_set

    background_edges = convert_to_tensor(background_set)
    
    for edge in background_edges.T:
        node1 = edge[0].item()
        node2 = edge[1].item()

        node1_pos = graph.node_pos[node1].numpy()
        node2_pos = graph.node_pos[node2].numpy()

        distance = np.linalg.norm(node1_pos - node2_pos)
        angle = np.arccos(np.dot(node1_pos, node2_pos)/(np.linalg.norm(node1_pos)*np.linalg.norm(node2_pos)))

        background_distances.append(distance)
        background_angles.append(angle)

    for edge in true_muon_edges.T:
        node1 = edge[0].item()
        node2 = edge[1].item()

        node1_pos = graph.node_pos[node1].numpy()
        node2_pos = graph.node_pos[node2].numpy()

        distance = np.linalg.norm(node1_pos - node2_pos)
        angle = np.arccos(np.dot(node1_pos, node2_pos)/(np.linalg.norm(node1_pos)*np.linalg.norm(node2_pos)))

        truth_muon_distances.append(distance)
        truth_muon_angles.append(angle)
    
    return truth_muon_distances, truth_muon_angles, background_distances, background_angles

def compute_distance_angle_stats(dir, samplesize):

    truth_muon_distances = []
    truth_muon_angles = []
    background_distances = []
    background_angles = []

    i=0

    for graph in glob.glob(dir):
        graph = torch.load(graph)
        truth_muon_d, truth_muon_a, background_d, background_a = edge_distance_angle(graph)

        truth_muon_distances.extend(truth_muon_d)
        truth_muon_angles.extend(truth_muon_a)
        background_distances.extend(background_d)
        background_angles.extend(background_a)

        if i >samplesize:
            break

        i+=1
    
    return truth_muon_distances, truth_muon_angles, background_distances, background_angles

def plot_distance_angle_stats(dir, samplesize, title='Cosmic Michel, fully connected', filename='edge_distance_angle'):

    truth_muon_distances, truth_muon_angles, background_distances, background_angles = compute_distance_angle_stats(dir, samplesize)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(truth_muon_distances, bins=50, edgecolor='black', linewidth=1.2, histtype='step', label='Signal edges (truth muon edges)')
    axs[0].hist(background_distances, bins=50, edgecolor='red', linewidth=1.2, histtype='step', label='Background edges')
    axs[0].set_xlabel('Distance between hits [mm]')
    axs[0].set_ylabel('Frequency')
    axs[0].set_yscale('log')
    axs[0].set_xlim(-20, 1220)
    axs[0].legend()
    axs[0].set_title('Distribution of edge distances ('+title+')')
    axs[0].grid(True)

    axs[1].hist(truth_muon_angles, bins=50, edgecolor='black', linewidth=1.2, histtype='step', label='Signal edges (true muon edges)')
    axs[1].hist(background_angles, bins=50, edgecolor='red', linewidth=1.2, histtype='step', label='Background edges')
    axs[1].set_xlabel('Angle between hits [rad]')
    axs[1].set_ylabel('Frequency')
    axs[1].set_yscale('log')
    axs[1].set_xlim(-0.1, np.pi+0.1)
    axs[1].legend()
    axs[1].set_title('Distribution of edge angles ('+title+')')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('/home/mue/karres/git/Mu3eCosmicGNN/examples/Cosmic_GNN/control_plots/'+filename+'.png', dpi=400)
    plt.show()
    
    print('Avg distance signal edges: ', np.mean(truth_muon_distances))
    print('Avg distance background edges: ', np.mean(background_distances))
    print('Avg angle signal edges: ', np.mean(truth_muon_angles))
    print('Avg angle background edges: ', np.mean(background_angles))

def find_weird_events(weird_path):
    ptots = []
    particle_types = []

    for weird_path in glob.glob(weird_path+'/*.pyg'):
        csv_path = weird_path.replace('.pyg', '-truth.csv')
        csv_path = csv_path.replace('fully_connected_cosmic_michel/weird_events', 'feature_store_cosmic_michel/trainset')

        if not os.path.exists(csv_path):
            csv_path = csv_path.replace('trainset', 'valset')

            if not os.path.exists(csv_path):
                csv_path = csv_path.replace('valset', 'testset')

        truth_df = pd.read_csv(csv_path)

        weird_hit = truth_df[truth_df.duplicated(subset=['particle_id', 'layer_id', 'ladder_id', 'module_id', 'station_id'], keep=False)]

        ptot = np.unique(np.sqrt(weird_hit['px'].values**2 + weird_hit['py'].values**2 + weird_hit['pz'].values**2))
        particle_id = np.unique(weird_hit['particle_id'].values)

        particle_path = csv_path.replace('-truth.csv', '-particles.csv')
        particle_df = pd.read_csv(particle_path)

        particle_df = particle_df[particle_df['particle_id'].isin(particle_id)]
        particle_type = particle_df['particle_type'].tolist()

        ptots.extend(ptot)
        particle_types.extend(particle_type)

    return ptots, particle_types

def plot_weird_events(dir):
    ptots, particle_types = find_weird_events(dir)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(ptots, bins=50, edgecolor='black', linewidth=1.2, histtype='step')
    axs[0].set_xlabel('p_tot [MeV]')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Distribution of total momenta of problematic hits (Cosmic Michel)')
    axs[0].grid(True)

    axs[1].hist(particle_types, bins=27, edgecolor='black', linewidth=1.2, histtype='step')
    axs[1].set_xlabel('pdgID')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of pdgIDs of problematic hits (Cosmic Michel)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('/home/mue/karres/git/Mu3eCosmicGNN/examples/Cosmic_GNN/control_plots/weird_hits.png', dpi=400)
    plt.show()

def get_eff_pur(graph):
    edge_index = graph.edge_index
    track_edges = graph.track_edges

    hit_to_particle_map = hit_id_to_particle_id_map(graph)

    true_muon_edges = remove_particle_edges(track_edges, 11, hit_to_particle_map)

    num_true_muon_edges = true_muon_edges.shape[1]
    num_edges = edge_index.shape[1]

    #Find intersection of edge index and true muon edges
    edge_index_set = sort_and_convert_to_set(edge_index)
    true_muon_set = sort_and_convert_to_set(true_muon_edges)
   
    true_muon_edges_in_edge_index_set = edge_index_set.intersection(true_muon_set)
    true_muon_edges_in_edge_index = convert_to_tensor(true_muon_edges_in_edge_index_set)

    num_true_muon_edges_in_edge_index = true_muon_edges_in_edge_index.shape[1]
    
    efficiency = num_true_muon_edges_in_edge_index / num_true_muon_edges
    purity = num_true_muon_edges / num_edges

    edge_info = np.array([num_edges, num_true_muon_edges, num_edges-num_true_muon_edges_in_edge_index]) #total edges, true muon edges, background edges
    return efficiency, purity, edge_info

def compute_eff_pur(dir, samplesize):
    efficiencies = []
    purities = []
    edge_info = np.empty((0,3))

    i=0

    for graph in glob.glob(dir):
        graph = torch.load(graph)
        eff, pur, edge = get_eff_pur(graph)

        efficiencies.append(eff)
        purities.append(pur)
        edge_info = np.vstack((edge_info,edge))

        if i > samplesize:
            break

        i+=1
    
    return np.array(efficiencies), np.array(purities), edge_info

def get_avg_eff_pur(stage, cuts, samplesize):
    '''
    Takes in a stage dir e.g. /mnt/data1/karres/cosmics_test/fully_connected_cosmic_michel_distance_cut_
    and a list of cuts e.g. ['500', '450', '400', '350', '300', '250', '200', '150', '100', '50']
    and outputs the average efficiency, purity and #edges for each cut
    '''

    dirs = [stage+cut+'/trainset/*.pyg' for cut in cuts]
    efficiencies = np.empty((0,samplesize+2))
    purities = np.empty((0,samplesize+2))
    edge_infos = np.empty((samplesize+2,3,0))

    for dir in dirs:
        print(dir)
        eff, pur, edge = compute_eff_pur(dir, samplesize)
        efficiencies = np.vstack((efficiencies, eff))
        purities = np.vstack((purities, pur))
        edge_infos = np.dstack((edge_infos, edge))

    avg_efficiencies = np.mean(efficiencies, axis=1)
    avg_purities = np.mean(purities, axis=1)
    avg_edge_infos = np.mean(edge_infos, axis=0)
    edge_infos_std = np.std(edge_infos, axis=0)

    avg_num_edges = avg_edge_infos[0,:]
    avg_num_muon_edges = avg_edge_infos[1,:]
    avg_num_background_edges = avg_edge_infos[2,:]

    efficiencies_std = np.std(efficiencies, axis=1)
    purities_std = np.std(purities, axis=1)
    num_edges_std = edge_infos_std[0,:]
    num_muon_edges_std = edge_infos_std[1,:]
    num_background_edges_std = edge_infos_std[2,:]

    return avg_efficiencies, avg_purities, avg_num_edges, avg_num_muon_edges, avg_num_background_edges, efficiencies_std, purities_std, num_edges_std, num_muon_edges_std, num_background_edges_std