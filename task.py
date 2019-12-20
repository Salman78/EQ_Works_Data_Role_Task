import pandas as pd
import numpy as np
from math import *
from scipy import spatial

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


i=0
dist_list = []
POI_index_list = []
login_index_list = []
for index, row in filteredData.iterrows():
    x_req, y_req, z_req = to_Cartesian(row['Latitude'], row['Longitude'])
    dist, POI_index = tree.query((x_req, y_req, z_req), 1)

    login_index_list.append(row['_ID'])
    dist_list.append(dist)
    POI_index_list.append(POI_index)


    i+=1
    if(i==3): break

login_and_dist_list = list(zip(login_index_list, POI_index_list, dist_list))

print(login_and_dist_list)








