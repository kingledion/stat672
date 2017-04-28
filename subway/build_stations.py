# -*- coding: utf-8 -*-

import subway_utils as su, csv, numpy as np
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

    def getDict(self):
        return self._data

def is_station(station, idx, pts):
    return [next(idx.nearest((*p, *p), 1, objects='raw')) == s for p in pts]   
    
def in_shape(state, pts):
    return [state.contains(shpgeo.Point(*p)) for p in pts]

with open('/opt/school/stat672/subway/boston_subwaygeo.csv', 'r') as csvin:
    rdr = csv.reader(csvin, delimiter = ';')
    
    idx = index.Index()
    stations = []
    
    for row in rdr:
        
        if len(row) > 2:
                   
            d = su.est_density(cursor, float(row[2]), float(row[1]))
            s = station(row[1], row[2], {**d, **{'name': row[0]}})     
            stations.append(s)
            idx.insert(0, (s['lon'], s['lat'], s['lon'], s['lat']), s)
            
    print("Density estimate and index built")
    
# calculate latitude degrees to 1 km
ns_deg = 1.0/110.574

# load shapefile of state in question (index 32 = MA, 19 = IL)
sf = shapefile.Reader('./shapes/cb_2015_us_state_20m')
state = shpgeo.shape(sf.shape(32).__geo_interface__)

            
for s in stations:
    n = 1000
    # calculate lon degrees to 1km
    ew_deg = 1.0/111.320/cos(radians(s['lat']))

    Theta = np.random.randn(n, 2)
    R = np.sqrt(np.random.rand(n))
    vectors = Theta * np.stack([R*ew_deg / np.linalg.norm(Theta, axis=1), R*ns_deg / np.linalg.norm(Theta, axis=1)], axis=1)
    
    pts = [(v[0] + s['lon'], v[1] + s['lat']) for v in vectors]
    count = sum(1 if a and b else 0 for a, b in zip(is_station(s, idx, pts), in_shape(state, pts)))
    # area of a 1km radius circle is pi
    print(s['name'], "{0:.2f}".format(count*pi/n))
    s['area'] = count*pi/n
     

    
arealist = sorted(stations, key=lambda x: x['area'])
with open("/opt/school/stat672/subway/boston_stations.csv" , 'w') as outfile:
    fields = arealist[0].getDict().keys()
    wrtr = csv.DictWriter(outfile, fieldnames = fields)
    
    wrtr.writeheader()
    for s in arealist:
        wrtr.writerow(s.getDict())
    

    
#printlist = sorted(stations, key=lambda x: x['area'] * (x['popdensity'] + x['empdensity']))
#
#keys = ['popdensity', 'empdensity', 'paydensity', 'housedensity']
#for s in printlist:
#    print(", ".join([s['name']] + [str(int(s[k]*s['area'])) for k in keys]))


        


