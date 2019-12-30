import pandas as pd
import numpy as np
from math import *
from scipy import spatial
from statistics import stdev
import matplotlib.pyplot as plt
import networkx as nx
import random
#%matplotlib inline
plt.style.use('seaborn-poster')

def filter_list_util(lst1, lst2, common=True):
    filtered_list = []
    for item in lst1:
        if(common==True):
            if(item in lst2):
                filtered_list.append(item)
        if(common==False):
            if (item not in lst2):
                filtered_list.append(item)
    return filtered_list

def construct_task_queue(starting_task, goal_task, job, G):
    tasks_not_placed = []
    topo_path_to_goal = list(nx.topological_sort(G))
    print("Entire topological path to goal: ",topo_path_to_goal)

    #starting_task = input("Enter starting task: ")
    starting_task_prereqs = nx.ancestors(G, starting_task)
    print("prerequisites of the starting task: ", starting_task_prereqs)
    filtered_topo_path_to_goal = filter_list_util(topo_path_to_goal, starting_task_prereqs, common=False)

    #goal_task = input("Enter goal task: ")
    path_to_goal = nx.ancestors(G, goal_task)
    print("path to goal: ",path_to_goal)

    final_filtered_path = filter_list_util(filtered_topo_path_to_goal, path_to_goal)
    print("final filtered path: ",final_filtered_path)

    for task in job:
        if(task not in final_filtered_path):
            tasks_not_placed.append(task)
    return final_filtered_path, tasks_not_placed

#--------------------------------------------------------------------------------------------------------#
#                                   HELPER METHODS                                                       #
#--------------------------------------------------------------------------------------------------------#

def to_Cartesian(lat, lng):
    '''
    function to convert latitude and longitude to 3D cartesian coordinates
    '''
    R = 6371 # radius of the Earth in kilometers

    x = R * cos(lat) * cos(lng)
    y = R * cos(lat) * sin(lng)
    z = R * sin(lat)
    return x, y, z


def deg2rad(degree):
    '''
    function to convert degree to radian
    '''
    rad = degree * 2 * np.pi / 360
    return (rad)


def rad2deg(rad):
    '''
    function to convert radian to degree
    '''
    degree = rad / 2 / np.pi * 360
    return (degree)


def distToKM(x):
    '''
    function to convert cartesian distance to real distance in km
    '''
    R = 6371  # earth radius
    gamma = 2 * np.arcsin(deg2rad(x / (2 * R)))  # compute the angle of the isosceles triangle
    dist = 2 * R * sin(gamma / 2)  # compute the side of the triangle
    return (dist)


def kmToDIST(x):
    '''
    function to convert real distance in km to cartesian distance
    '''
    R = 6371  # earth radius
    gamma = 2 * np.arcsin(x / 2. / R)

    dist = 2 * R * rad2deg(sin(gamma / 2.))
    return (dist)

def Average(lst):
    if (len(lst)==0): return 0
    else:
        cartesian_dist_average = sum(lst) / len(lst)
        km_average_dist = distToKM(cartesian_dist_average)
        return km_average_dist

def standard_deviation(lst):
    if (len(lst) == 0): return 0
    else:
        cartesian_stdev = stdev(lst)
        km_stdev = distToKM(cartesian_stdev)
        return km_stdev

def requestDensity(r, requests):
    PI = 3.142
    if (r==0 or len(requests)==0): return 0
    else:
        area = PI * (r * r)
        density = (len(requests))/area
        return density

def max_dist_from_POIs(POIs_dist_list):
    if (len(POIs_dist_list)==0): return 0
    else:
        cartesian_dist = max(POIs_dist_list)
        km_dist = distToKM(cartesian_dist)
        return km_dist

def remove_outliers(POIs_dist, n):
    if (len(POIs_dist) == 0): return 0
    else:
        for i in range(n):
            cartesian_dist = max(POIs_dist)
            POIs_dist.remove(cartesian_dist)
        final_km_dist = max_dist_from_POIs(POIs_dist)
        return final_km_dist

def main():
    #-------------------------------------------------------------------------------------------------------------#
    #                           READING DATASET AND FILTERING OUT SUSPICIOUS RECORDS                              #
    #-------------------------------------------------------------------------------------------------------------#
    initialData = pd.read_csv('/Users/Tausal21/Desktop/EQ_Works/project/ws-data-spark/data/DataSample.csv')
    num_of_suspicious_records = initialData.duplicated(subset=[' TimeSt', 'Latitude', 'Longitude']).sum()
    print("Number of suspicious records dropped: ", num_of_suspicious_records)
    filteredData = initialData.drop_duplicates(keep=False, subset=[' TimeSt', 'Latitude', 'Longitude']) #filtering out suspicious logins
    print(filteredData.head())

    POIData = pd.read_csv('/Users/Tausal21/Desktop/EQ_Works/project/ws-data-spark/data/POIList.csv')
    print(POIData.head())

    #-------------------------------------------------------------------------------------------------------------#
    #                           CONVERTING LAT/LONG POINTS TO 3D CARTESIAN AND BUILDING KD-TREE                   #
    #-------------------------------------------------------------------------------------------------------------#

    POI_lats = POIData[[' Latitude']].values
    POI_longs = POIData[['Longitude']].values

    POI_x, POI_y, POI_z = zip(*map(to_Cartesian, POI_lats, POI_longs))


    POI_coordinates = list(zip(POI_x, POI_y, POI_z))
    tree = spatial.cKDTree(POI_coordinates)


    POI_01_dist = [] #storing min distances from logins to POI_01
    POI_02_dist = [] #  " " " POI_02
    POI_03_dist = [] #  " " " POI_03
    POI_04_dist = [] #  " " " POI_04
    POI_list = [] #we append this list as a column at the end of the original filtered dataset
    for index, row in filteredData.iterrows():
        x_req, y_req, z_req = to_Cartesian(row['Latitude'], row['Longitude'])
        dist, POI_index = tree.query((x_req, y_req, z_req), 1)

        POI_list.append(POI_index)

        if(POI_index==0): POI_01_dist.append(dist)
        if(POI_index==1): POI_02_dist.append(dist)
        if(POI_index==2): POI_03_dist.append(dist)
        if(POI_index==3): POI_04_dist.append(dist)

    filteredData['Nearest_POI'] = POI_list
    print(filteredData.head())

    #-------------------------------------------------------------------------------------------------------------#
    #                       CALCULATING AVERAGE AND STANDARD DEVIATION FROM EACH POI TO REQ POINT                 #
    #-------------------------------------------------------------------------------------------------------------#
    print("-----------CALCULATING AVERAGE AND STANDARD DEVIATION FROM EACH POI TO REQ POINT---------")

    POI_01_avg_dist_requestPoints = Average(POI_01_dist)
    POI_02_avg_dist_requestPoints = Average(POI_02_dist)
    POI_03_avg_dist_requestPoints = Average(POI_03_dist)
    POI_04_avg_dist_requestPoints = Average(POI_04_dist)
    print("Average real distance in km from POI_01 through 04 \n "
          "to their nearest request points, respectively: ",
          POI_01_avg_dist_requestPoints, POI_02_avg_dist_requestPoints,
          POI_03_avg_dist_requestPoints, POI_04_avg_dist_requestPoints)


    POI_01_stdev_requestPoints = standard_deviation(POI_01_dist)
    POI_02_stdev_requestPoints = standard_deviation(POI_02_dist)
    POI_03_stdev_requestPoints = standard_deviation(POI_03_dist)
    POI_04_stdev_requestPoints = standard_deviation(POI_04_dist)
    print("Standard Deviation in km from POI_01 through 04 \n "
          "to their nearest request points, respectively: ",
          POI_01_stdev_requestPoints, POI_02_stdev_requestPoints,
          POI_03_stdev_requestPoints, POI_04_stdev_requestPoints)

    #-------------------------------------------------------------------------------------------------------------#
    #                       CALCULATING RADIUS AND DENSITY FOR EACH POI CIRCLE                                    #
    #-------------------------------------------------------------------------------------------------------------#
    print("--------CALCULATING RADIUS AND DENSITY FOR EACH POI CIRCLE-------")

    POI_01_circle_rad_km = max_dist_from_POIs(POI_01_dist)
    POI_01_req_density = requestDensity(POI_01_circle_rad_km, POI_01_dist)

    POI_02_circle_rad_km = max_dist_from_POIs(POI_02_dist)
    POI_02_req_density = requestDensity(POI_02_circle_rad_km, POI_02_dist)

    POI_03_circle_rad_km = max_dist_from_POIs(POI_03_dist)
    POI_03_req_density = requestDensity(POI_03_circle_rad_km, POI_03_dist)

    POI_04_circle_rad_km = max_dist_from_POIs(POI_04_dist)
    POI_04_req_density = requestDensity(POI_04_circle_rad_km, POI_04_dist)

    print("Radius of the POIs encompassing all of their nearest req points \n"
          "in real distance(km), respectively: ",
          POI_01_circle_rad_km, POI_02_circle_rad_km,
          POI_03_circle_rad_km, POI_04_circle_rad_km)

    print("Request density of each POI respectively: ",
          POI_01_req_density, POI_02_req_density,
          POI_03_req_density, POI_04_req_density)

    #----------------------------------------------------------------------------------------------------------#
    #                           DRAWING CIRCLES WITH POIs                                                      #
    #----------------------------------------------------------------------------------------------------------#

    POI_01_lat_long = list(POIData.loc[0, [' Latitude', 'Longitude']])
    POI_02_lat_long = list(POIData.loc[1, [' Latitude', 'Longitude']])
    POI_03_lat_long = list(POIData.loc[2, [' Latitude', 'Longitude']])
    POI_04_lat_long = list(POIData.loc[3, [' Latitude', 'Longitude']])

    POI_01_circle = plt.Circle((POI_01_lat_long[1], POI_01_lat_long[0]), POI_01_circle_rad_km, color='blue', alpha=0.1)
    POI_02_circle = plt.Circle((POI_02_lat_long[1], POI_02_lat_long[0]), POI_02_circle_rad_km, color='brown', alpha=0.1)
    POI_03_circle = plt.Circle((POI_03_lat_long[1], POI_03_lat_long[0]), POI_03_circle_rad_km, color='green', alpha=0.1)
    POI_04_circle = plt.Circle((POI_04_lat_long[1], POI_04_lat_long[0]), POI_04_circle_rad_km, color='yellow', alpha=0.1)
    fig, ax = plt.subplots()
    ax.set_xlim((-500, 500))
    ax.set_ylim((-500, 500))
    ax.add_artist(POI_01_circle)
    ax.add_artist(POI_03_circle)
    ax.add_artist(POI_04_circle)
    fig.savefig('POI_circles.png')

    #-------------------------------------------------------------------------------------------------------------#
    #                              REQUEST LOGIN AND POI SCATTERPLOT                                              #
    #-------------------------------------------------------------------------------------------------------------#

    #we extract lat/long for each group of points closest to their respective POIs
    req_lat_long_POI_01 = filteredData.loc[filteredData['Nearest_POI'] == 0]
    req_lat_long_POI_02 = filteredData.loc[filteredData['Nearest_POI'] == 1]
    req_lat_long_POI_03 = filteredData.loc[filteredData['Nearest_POI'] == 2]
    req_lat_long_POI_04 = filteredData.loc[filteredData['Nearest_POI'] == 3]


    p1 = plt.scatter(req_lat_long_POI_01['Longitude'], req_lat_long_POI_01['Latitude'],s=10, c='blue', alpha=0.4)
    POI_01 = plt.scatter(POI_01_lat_long[1], POI_01_lat_long[0], s=800 , c='red', marker='x',edgecolor='black',linewidth=2, alpha=1)

    p2 = plt.scatter(req_lat_long_POI_02['Longitude'], req_lat_long_POI_02['Latitude'],s=10,c='brown', alpha=0.4)
    POI_02 = plt.scatter(POI_02_lat_long[1], POI_02_lat_long[0], s=800 , c='red', marker='x',edgecolor='black',linewidth=2, alpha=1)

    p3 = plt.scatter(req_lat_long_POI_03['Longitude'], req_lat_long_POI_03['Latitude'],s=10,c='green', alpha=0.4)
    POI_03 = plt.scatter(POI_03_lat_long[1], POI_03_lat_long[0], s=800 , c='red', marker='x',edgecolor='black',linewidth=2, alpha=1)

    p4 = plt.scatter(req_lat_long_POI_04['Longitude'], req_lat_long_POI_04['Latitude'],s=10,c='yellow', alpha=0.4)
    POI_04 = plt.scatter(POI_04_lat_long[1], POI_04_lat_long[0], s=800 , c='red', marker='x',edgecolor='black',linewidth=2, alpha=1)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Request login and POI Scatterplot')
    plt.annotate(('POI_01'),xy=(POI_01_lat_long[1], POI_01_lat_long[0]),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='brown', alpha=0.4),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                 )
    '''
    plt.annotate(('POI_02'),xy=(POI_02_lat_long[1], POI_02_lat_long[0]),
                 textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                 )
    '''
    plt.annotate(('POI_03'),xy=(POI_03_lat_long[1], POI_03_lat_long[0]),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='brown', alpha=0.4),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                 )
    plt.annotate(('POI_04'),xy=(POI_04_lat_long[1], POI_04_lat_long[0]),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='brown', alpha=0.4),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                 )
    plt.legend((p1,p2,p3,p4), ('Closest to POI_01', 'Closest to POI_02', 'Closest to POI_03', 'Closest to POI_04'),
               loc='lower left',
               ncol=3,
               fontsize=10)
    plt.show()
    plt.savefig('request_POI_scatterplot.png')

    #----------------------------------------------------------------------------------------------------------#
    #                                         DAG SCHEDULING                                                   #
    #----------------------------------------------------------------------------------------------------------#

    print("------------PIPELINE DEPENDENCY-----------")
    G = nx.DiGraph()

    with open("/Users/Tausal21/PycharmProjects/interview_task3/task_ids.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
    task_list = currentline

    for node in currentline:
        G.add_node(node)
    print('Graph nodes: ',G.nodes)

    with open("/Users/Tausal21/PycharmProjects/interview_task3/relations.txt", "r") as filestream:
        for line in filestream:
            currentline = line.rstrip('\n').split("->")
            #print(currentline[0], currentline[1])
            G.add_edge(currentline[0], currentline[1])

    #starting and goal tasks are hardcoded 73 and 36 respectively
    m = random.randrange(2,7,1)
    job = random.sample(task_list, m)
    output_task_queue_hardcoded, tasks_not_placed_hardcoded = construct_task_queue('73', '36', job, G)
    print("----this is run with starting and goal tasks as 73 and 36 respectively----")
    print("output task queue: ", output_task_queue_hardcoded)
    print("tasks from the new job not placed: ", tasks_not_placed_hardcoded)

    #user inputs starting and goal task
    starting_task = input("Enter starting task: ")
    goal_task = input("Enter goal task: ")
    output_task_queue, tasks_not_placed = construct_task_queue(starting_task, goal_task, job, G)
    print("-----this is run with user giving starting and goal tasks as input----")
    print("output task queue: ", output_task_queue)
    print("tasks from the new job not placed: ", tasks_not_placed)

if __name__ == '__main__':
    main()