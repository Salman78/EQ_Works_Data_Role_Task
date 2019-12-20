import pandas as pd
import numpy as np
from math import *
from scipy import spatial
from statistics import stdev
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

%matplotlib inline
plt.style.use('seaborn-poster')

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
    return sum(lst) / len(lst)

initialData = pd.read_csv('/Users/Tausal21/Desktop/EQ_Works/project/ws-data-spark/data/DataSample.csv')
POIData = pd.read_csv('/Users/Tausal21/Desktop/EQ_Works/project/ws-data-spark/data/POIList.csv')
#OPTIONAL: checking the column names
#for col in initialData.columns:
#   print(col)


#print(initialData.shape)
#print(initialData.duplicated(subset=[' TimeSt', 'Latitude', 'Longitude']).sum()) #number of suspicious logins
filteredData = initialData.drop_duplicates(keep=False, subset=[' TimeSt', 'Latitude', 'Longitude']) #filtering out suspicious logins
print(filteredData.head())

print(POIData.head())

#converting POI DF's lat/long to numpy arrary
#POIs_lat_long = POIData[[' Latitude', 'Longitude']].values
POI_lats = POIData[[' Latitude']].values
POI_longs = POIData[['Longitude']].values
#print(POI_lats)

#Converted to 3D coordinates
POI_x, POI_y, POI_z = zip(*map(to_Cartesian, POI_lats, POI_longs))

#creating cKDTree using POI coordinates
POI_coordinates = list(zip(POI_x, POI_y, POI_z))
tree = spatial.cKDTree(POI_coordinates)


POI_01 = [] #storing min distances from logins to POI_01
POI_02 = [] #  " " " POI_02
POI_03 = [] #  " " " POI_03
POI_04 = [] #  " " " POI_04
POI_list = [] #we append this list as a column at the end of the original filtered dataset
for index, row in filteredData.iterrows():
    x_req, y_req, z_req = to_Cartesian(row['Latitude'], row['Longitude'])
    dist, POI_index = tree.query((x_req, y_req, z_req), 1)

    POI_list.append(POI_index)

    if(POI_index==0): POI_01.append(dist)
    if(POI_index==1): POI_02.append(dist)
    if(POI_index==2): POI_03.append(dist)
    if(POI_index==3): POI_04.append(dist)

    #i+=1
    #if(i==3): break
filteredData['Nearest_POI'] = POI_list
print(filteredData.head())

POI_01_avg_dist_requestPoints = Average(POI_01)
POI_02_avg_dist_requestPoints = Average(POI_02)
POI_03_avg_dist_requestPoints = Average(POI_03)
POI_04_avg_dist_requestPoints = Average(POI_04)

POI_01_stdev_requestPoints = stdev(POI_01)
POI_02_stdev_requestPoints = stdev(POI_02)
POI_03_stdev_requestPoints = stdev(POI_03)
POI_04_stdev_requestPoints = stdev(POI_04)











