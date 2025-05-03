##
## Copied from https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_08en_chicago.html
## 

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from uxsim import *

############################################
# tntp parser customized for Chicago-sketch
############################################

def import_links(filename):
    """
    Modified from `parsing networks in Python.ipynb` in https://github.com/bstabler/TransportationNetworks
    Credit: Transportation Networks for Research Core Team. Transportation Networks for Research. https://github.com/bstabler/TransportationNetworks. Accessed 2021-10-01.
    """
    net = pd.read_csv(filename, skiprows=8, sep='\t')

    trimmed= [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop(['~', ';'], axis=1, inplace=True)
    return net

def import_nodes(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse lines that are not empty or comments
    data = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('~'):
            parts = line.split()
            if len(parts) >= 3:  # Make sure there are enough elements
                data.append(parts[:3])

    df = pd.DataFrame(data, columns=["node", "x", "y"])
    df = df.astype({'node': int, 'x': float, 'y': float})
    return df


def import_odmatrix_phila(matfile, nodefile):
    """
    Modified from `parsing networks in Python.ipynb` in https://github.com/bstabler/TransportationNetworks
    Credit: Transportation Networks for Research Core Team. Transportation Networks for Research. https://github.com/bstabler/TransportationNetworks. Accessed 2021-10-01.
    """
    f = open(matfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig=int(orig[0])

        d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())

    # mat = np.zeros((zones, zones))
    # for i in range(zones):
    #     for j in range(zones):
    #         # We map values to a index i-1, as Numpy is base 0
    #         mat[i, j] = matrix.get(i+1,{}).get(j+1,0)

    mat = [[0 for i in range(zones)] for j in range(zones)]
    for i in range(zones):
        for j in range(zones):
            # We map values to a index i-1, as Python is base 0
            mat[i][j] = matrix.get(i+1,{}).get(j+1,0)

    index = np.arange(zones) + 1

    nodelist = [i for i in range(1525)]#list(import_nodes(nodefile)["node"])

    df = pd.DataFrame(mat, columns=nodelist)
    df["origin"] = nodelist
    df = df.reindex(columns=["origin"]+nodelist)
    return df

df_nodes = import_nodes(r"./TransportationNetworks/Philadelphia/Philadelphia_node.tntp")
df_links = import_links(r"./TransportationNetworks/Philadelphia/Philadelphia_net.tntp")
df_demand = import_odmatrix_phila(r"./TransportationNetworks/Philadelphia/Philadelphia_trips.tntp", r"./TransportationNetworks/Philadelphia/Philadelphia_node.tntp")

print(df_nodes.columns)

############################################
# prepare for deleting dummy links/nodes
############################################

map_dummy2real = {}
map_real2dummy = {}
for i in lange(df_links):
    if df_links["capacity"][i] == 49500:    #dummy link between dummy node and real node
        if df_links["init_node"][i] not in map_dummy2real.keys() and df_links["init_node"][i] < df_links["term_node"][i]:
            #print("delete: link", (df_links["init_node"][i], df_links["term_node"][i]), end=", ")
            map_dummy2real[df_links["init_node"][i]] = df_links["term_node"][i]
print("")

############################################
# uxsim setup
############################################


W = World(
    name="",
    deltan=30,
    tmax=10000,
    print_mode=1, save_mode=1, show_mode=1,
    random_seed=42,
    vehicle_logging_timestep_interval=1,
    meta_data = {"DATA SOURCE AND LICENCE": "Philadelphia network. This is based on https://github.com/bstabler/TransportationNetworks/tree/master/Chicago-Sketch by Transportation Networks for Research Core Team. Users need to follow their licence. Especially, this data is for academic research purposes only, and users must indicate the source of any dataset they are using in any publication that relies on any of the datasets provided in this web site."}
)

for i in range(len(df_nodes)):
    if i+1 not in map_dummy2real.keys():
        name = str(i+1)
        x = float(df_nodes["x"][i])
        y = float(df_nodes["y"][i])
        W.addNode(name, x, y)

for i in range(len(df_links)):
    start_node = df_links["init_node"][i]
    end_node = df_links["term_node"][i]

    if start_node in map_dummy2real.keys() or end_node in map_dummy2real.keys():    #delete dummy node
        continue

    length = df_links["length"][i]*1609.34  #mile to m
    if length < 100:
        length = 100

    capacity = df_links["capacity"][i]/3600
    n_lanes = int(capacity/0.5) #based on the assumption that flow capacity per lane is 0.5 veh/s in arterial roads
    if n_lanes < 1:
        n_lanes = 1
    if df_links["link_type"][i] == 2:   #highway
        n_lanes = 3

    free_flow_time = df_links["free_flow_time"][i]*60
    if free_flow_time > 0:
        free_flow_speed = length/free_flow_time
    else:
        free_flow_speed = 10
    if free_flow_speed < 10:
        free_flow_speed = 10
    elif free_flow_speed > 40:
        free_flow_speed = 40

    W.addLink(str(i), str(start_node), str(end_node), length=length, free_flow_speed=free_flow_speed, number_of_lanes=n_lanes)


demand_multipiler = 1/0.85  #compensete some demands that are filtered out by pre-processing
demand_threthhold = 30  #delete too small demand (<= deltan) as they cause peculiar traffic pattern at the beginning of simulation
for i in tqdm(range(len(df_demand))):
    origin = str(i+1)
    if int(origin) in map_dummy2real.keys():
        origin = str(map_dummy2real[int(origin)])
    for j in range(len(df_demand)):
        destination = str(j+1)

        if int(destination) in map_dummy2real.keys():
            destination = str(map_dummy2real[int(destination)])

        if origin == destination:
            continue

        demand = df_demand.loc[i, j]*demand_multipiler
        if demand > demand_threthhold:
            try:
                W.adddemand(origin, destination, 0, 3600, volume=demand)
            except:
                print("inconsistent demand:", origin, destination, demand)

W.save_scenario(r"./TransportationNetworks/Philadelphia/Philadelphia_sketch.uxsim_scenario")

W.exec_simulation()
W.analyzer.print_simple_stats()

W.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.2, interval=5, trace_length=10,  figsize=6, antialiasing=False)
