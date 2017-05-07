# -*- coding: utf-8 -*-

import subway_utils as su, csv , numpy as np
from rtree import index
from math import cos, radians, pi
import shapely.geometry as shpgeo, shapefile

db, cursor = su.opendb()

class station:
    
    def __init__(self, lat, lon, datadict = {}):
        self._data = datadict
        self._data['lat'] = float(lat)
        self._data['lon'] = float(lon)
        
    def __getitem__(self, key):
        if key not in self._data:
            raise IndexError(key)
        return self._data[key]
        
    def __setitem__(self, key, value):
        self._data[key] = value
        
    def __hash__(self):
        return hash((self._data['lat'], self._data['lon']))
        
    def __iter__(self):
        for key in self._data:
            yield key
        
    def __eq__(self, other):
        return True if self._data['lat'] == other._data['lat'] and  self._data['lon'] == other._data['lon'] else False
        
    def getfields(self):
        return [k for k in self._data]

def near_station(station, idx, pts):
    return [next(idx.nearest((*p, *p), 1, objects='raw')) == station for p in pts]   
    
def in_shape(state, pts):
    return [state.contains(shpgeo.Point(*p)) for p in pts]
    
def build_station_index(filename):

    with open(filename, 'r') as csvin:
        
        rdr = csv.reader(csvin, delimiter = ';')
        
        idx = index.Index()
        stations = []
        
        for row in rdr:
            
            if len(row) > 3:
                       
                d = su.est_density(cursor, float(row[2]), float(row[1]))
                s = station(row[1], row[2], {**d, **{'name': row[0]}, **{'parking': row[3]}})     
                stations.append(s)
                idx.insert(0, (s['lon'], s['lat'], s['lon'], s['lat']), s)
                
            else:
                print(row)
                
    return stations, idx
    
def network_map()
    
def calculate_areas(stations, idx, shp):
    for s in stations:
        n = 100 # done 1000
        # calculate lon degrees to 1km
        ew_deg = 1.0/111.320/cos(radians(s['lat']))
    
        Theta = np.random.randn(n, 2)
        R = np.sqrt(np.random.rand(n))
        vectors = Theta * np.stack([R*ew_deg / np.linalg.norm(Theta, axis=1), R*ns_deg / np.linalg.norm(Theta, axis=1)], axis=1)
        
        # pts in a 1km area
        pts = [(v[0] + s['lon'], v[1] + s['lat']) for v in vectors]
        walk = [1 if a else 0 for a in in_shape(shp, pts)]
        wcount = sum(walk)
        near = sum(1 if a and b else 0 for a, b in zip(near_station(s, idx, pts), walk))

        s['narea'] = near*pi/n # area of a 1km radius circle is pi
        s['warea'] = wcount*pi/n
        
        # pts in 15km area
        pts = [(v[0]*15 + s['lon'], v[1]*15 + s['lat']) for v in vectors]
        drive = sum(1 if a and b else 0 for a, b in zip(near_station(s, idx, pts), in_shape(shp, pts)))
        s['darea'] = drive*225*pi/n
        
        print(s['name'], "{0:.2f}".format(wcount*pi/n), "{0:.2f}".format(near*pi/n), "{0:.2f}".format(drive*225*pi/n))
        
        
def write_stations(stations, filename):
    fields = ['name', 'lat', 'lon', 'popnear', 'housenear', 'empnear', 'paynear', 'popwalk', 'housewalk', 'empwalk', 'paywalk', 'popdrive', 'housedrive', 'parking']
    with open(filename , 'w') as outfile:
        wrtr = csv.writer(outfile) 
        wrtr.writerow(fields)
        for s in stations:
            wrtr.writerow([s[f] for f in fields])
            
def calculate_totals(stations):
    fields = stations[0].getfields()
    dfields = [f for f in fields if f.endswith('density')]
    for s in stations:
        for d in dfields:
            name = d.split('density')[0]
            
            s[name + 'near'] = int(s[d] * s['narea'])
            s[name + 'walk'] = int(s[d] * s['warea'])
            s[name + 'drive'] = int(s[d] * s['darea'])
        
    
            
################################################
# Load station geo data; build station lists; build station geoindices
    
bstations, bidx = build_station_index('/opt/school/stat672/subway/boston_subwaygeo.csv')
cstations, cidx = build_station_index('/opt/school/stat672/subway/chicago_subwaygeo.csv')
print("Density estimate and index built")

# load station network maps
bnet = network_map()

#############################################
# Load shapefiles, calculate available walking (1km) and driving (15km) areas
    
# calculate latitude degrees to 1 km
ns_deg = 1.0/110.574

# load shapefile of states in question (index 32 = MA, 19 = IL)
sf = shapefile.Reader('./shapes/cb_2015_us_state_20m')

calculate_areas(bstations, bidx, shpgeo.shape(sf.shape(32).__geo_interface__))
calculate_areas(cstations, cidx, shpgeo.shape(sf.shape(19).__geo_interface__))
print("Areas calculated")


###############################################
# Multiply densities by areas to get counts

calculate_totals(bstations)
calculate_totals(cstations)

           
###############################################
# Write staions to csv files
           
write_stations(bstations, "/opt/school/stat672/subway/boston_stations.csv")    
write_stations(cstations, "/opt/school/stat672/subway/chicago_stations.csv")  



        


