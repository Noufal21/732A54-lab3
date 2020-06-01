from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext
sc = SparkContext(appName="lab_kernel")


# In[166]:



h_distance = 500# Up to you
h_date = 10# Up to you
h_time = 3# Up to you
a = 60.3097 # Up to you
b = 12.6959 # Up to you
date = "2014-07-04" # Up to you
date = datetime.strptime(date, "%Y-%m-%d")

# In[167]:


temp = sc.textFile("BDALab3/input/temperature-readings.csv")
lines = temp.map(lambda line: line.split(";"))
station_temperature = lines.map(lambda x: (x[0], (datetime.strptime(x[1], '%Y-%m-%d' ),datetime.strptime(x[2], '%H:%M:%S'),float(x[3]) )))
station_temperature = station_temperature.filter(lambda x:(x[1][0] < date))


# In[168]:


stations = sc.textFile("BDALab3/input/stations.csv")
lines = stations.map(lambda line: line.split(";"))
station_location = lines.map(lambda x: (x[0],(float(x[4]),float(x[3]) )))



# In[169]:


station_distinct = station_location.collectAsMap()
broadcast_station = sc.broadcast(station_distinct)


# # Functions

# In[170]:


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

def bind_location_temp(station):
    
    station_location = list(broadcast_station.value[station[0]])
    station_temp  = list(station[1])
    station_temp.extend((station_location[0] , station_location[1]))
    complete_obj = (station[0] , tuple(station_temp))
    return complete_obj

def date_difference(date1,date2):
    delta = abs((date1 - date2).days)
    return min(delta,abs(365-delta))
    #return 1

def time_difference(time1,time2):
    delta = abs((time1 - time2).seconds) / 3600
    return min(delta,abs(24-delta))

def dist_kernal(distDiff,h_distance):
    return(exp(-(distDiff**2)/h_distance**2))

def date_kernal(dateDiff,h_date):
    return(exp(-(dateDiff**2)/h_date**2))

def time_kernal(timeDiff,h_time):
    return(exp(-(timeDiff**2)/h_time**2))


# # Logic

# In[171]:


binded_data = station_temperature.map(lambda x: bind_location_temp(x))


# In[172]:


basic_kernal = binded_data.map(lambda x:(x[0],(date_kernal(date_difference(date,x[1][0]),h_date),x[1][1],x[1][2], dist_kernal(haversine(x[1][3],x[1][4],b,a),h_distance)    )))


# In[196]:

cached_kernal = basic_kernal
cached_kernal.cache()

PredictSumKernal =[]
PredictProdKernal =[]
for time in ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00","12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]:
    t =  datetime.strptime(time, '%H:%M:%S')
    add_timeKernal = basic_kernal.map(lambda x: ((x[0]),   (x[1][0],   time_kernal(time_difference(x[1][1],t),h_time), x[1][2], x[1][3])))
    
    
    Weight_add = add_timeKernal.map(lambda x : ( 1  ,  ( (x[1][0] + x[1][1] + x[1][3]),x[1][2])  )).map(lambda x:[x[1][0],x[1][0]*x[1][1]]) .reduce(lambda x,y : (x[0]+y[0],x[1]+y[1]))
    Weight_add = Weight_add[1]/Weight_add[0]
    PredictSumKernal.append(Weight_add)
    
    
    Weight_prod = add_timeKernal.map(lambda x : ( 1  ,  ( (x[1][0] * x[1][1] * x[1][3]),x[1][2])  )).map(lambda x:[x[1][0],x[1][0]*x[1][1]]) .reduce(lambda x,y : (x[0]+y[0],x[1]+y[1]))
    Weight_prod = Weight_prod[1]/Weight_prod[0]
    PredictProdKernal.append(Weight_prod)


# In[197]:

rdd_sumKernal = sc.parallelize(PredictSumKernal,1)
rdd_sumKernal.saveAsTextFile("BDALab3/SumKernal")


# In[198]:


rdd_prodKernal = sc.parallelize(PredictProdKernal,1)
rdd_prodKernal.saveAsTextFile("BDALab3/ProdKernal")

