import numpy as np
from sklearn.cluster import KMeans
import numpy.random as np_rnd

class DataCreator(object):
    def __init__(self):
        self.name = 'DataCreator Class'
        self.model = None
        self.events_per_centroid = None
        
    def fit(self, data, n_clusters=2):
        self.model = KMeans(n_clusters=n_clusters)
        self.model.fit(data)
        event_per_centroid = []
        output = self.model.predict(data)
        for icenter in range(n_clusters):
            event_per_centroid = (np.append(event_per_centroid,
                                            float(sum(output==icenter))/
                                            float(len(data))))
        self.events_per_centroid = event_per_centroid
            
    def create_events(self, data, n_events=100):
        output = self.model.predict(data)
        for icenter in range(len(self.events_per_centroid)):
            qtd_events = (np.float(n_events)*np.ceil(self.events_per_centroid[icenter])).astype(int)
            if qtd_events == 0:
                continue
            select_data = data[output==icenter,:]
            return select_data [np.random.randint(0, select_data.shape[0]-1, size=qtd_events),:]