import os
import graphlab as gl
from math import sqrt

if os.path.exists('schizophrenia_clean'):
    sf = gl.SFrame('schizophrenia_clean')
else:
    sf_functional = gl.SFrame.read_csv(
        'https://static.turi.com/datasets/mlsp_2014/train_FNC.csv')
    sf_morphometry = gl.SFrame.read_csv(
        'https://static.turi.com/datasets/mlsp_2014/train_SBM.csv')

    sf = sf_functional.join(sf_morphometry, on="Id")
    sf = sf.remove_column('Id')

    sf.save('schizophrenia_clean')


K = int(sqrt(sf.num_rows() / 2.0))

kmeans_model = gl.kmeans.create(sf, num_clusters=K)
kmeans_model.summary()


#The cluster_info SFrame indicates the final cluster centers, one per row, in terms of the same features used to create the model.

kmeans_model['cluster_info'].print_rows(num_columns=5, max_row_width=80,
                                        max_column_width=10)

#The last three columns of the cluster_info SFrame indicate metadata about the corresponding cluster: ID number, number of points in the cluster, and the within-cluster sum of squared distances to the center.

kmeans_model['cluster_info'][['cluster_id', 'size', 'sum_squared_distance']]

#The cluster_id field of the model shows the cluster assignment for each input data point, along with the Euclidean distance from the point to its assigned cluster's center.

kmeans_model['cluster_id'].head()


#predict
new_clusters = kmeans_model.predict(sf[:5])
print new_clusters

