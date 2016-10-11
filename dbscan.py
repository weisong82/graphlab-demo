import graphlab as gl
from sklearn.datasets import make_moons

data = make_moons(n_samples=200, shuffle=True, noise=0.1, random_state=19)
sf = gl.SFrame(data[0]).unpack('X1')

dbscan_model = gl.dbscan.create(sf, radius=0.25)
dbscan_model.summary()

dbscan_model['cluster_id'].head(5)
dbscan_model['cluster_id'].tail(5)
