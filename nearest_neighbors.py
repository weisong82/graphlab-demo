import graphlab as gl
import os
'''
The reference dataset that is used to create the nearest neighbors model cannot have missing data.
 Please use the SFrame.fillna and SFrame.dropna utilities to preprocess missing values before
 creating a nearest neighbors model.
'''
if os.path.exists('houses.csv'):
    sf = gl.SFrame.read_csv('houses.csv')
else:
    data_url = 'https://static.turi.com/datasets/regression/houses.csv'
    sf = gl.SFrame.read_csv(data_url)
    sf.save('houses.csv')

print sf.head(5)
'''
+------+---------+------+--------+------+-------+
| tax  | bedroom | bath | price  | size |  lot  |
+------+---------+------+--------+------+-------+
| 590  |    2    | 1.0  | 50000  | 770  | 22100 |
'''
##!!! it is important to normalize the features
for c in sf.column_names():
    sf[c] = (sf[c] - sf[c].mean()) / sf[c].std()

model = gl.nearest_neighbors.create(sf)
model = gl.nearest_neighbors.create(sf, features=['bedroom', 'bath', 'size'])
model.summary()

knn = model.query(sf[:5], k=5)
print knn.head()

'''
By default, the similarity_graph method returns an SGraph whose vertices are the rows of the reference dataset
and whose edges indicate a nearest neighbor match. Specifically, the destination vertex of an edge is a nearest
neighbor of the source vertex.
similarity_graph can also return results in the same form as the query method if so desired.
'''
sim_graph = model.similarity_graph(k=3)
sim_graph.show(vlabel='id', arrows=True)


##distance

model = gl.nearest_neighbors.create(sf, features=['bedroom', 'bath', 'size'],
                                    distance=gl.distances.manhattan)
knn = model.query(sf[:3], k=3)
knn.print_rows()

sf_check = sf[['bedroom', 'bath', 'size']]
print "distance check 1:", gl.distances.manhattan(sf_check[2], sf_check[10])
print "distance check 2:", gl.distances.manhattan(sf_check[2], sf_check[14])