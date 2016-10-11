import os
import graphlab as gl
data_file = 'US_business_links'
if os.path.exists(data_file):
    sg = gl.load_sgraph(data_file)
    # sg.save('1', format='csv')
else:
    url = 'https://static.turi.com/datasets/' + data_file
    sg = gl.load_sgraph(url)
    sg.save(data_file)
print sg.summary()

pr = gl.pagerank.create(sg, max_iterations=10)
# print pr['pagerank']
print pr.summary()

pr_out = pr['pagerank']
print pr_out.topk('pagerank', k=10)

##Triangle counting
##The number of triangles in a vertex's immediate neighborhood is a measure of the "density" of the vertex's neighborhood.
tri = gl.triangle_counting.create(sg)
print tri.summary()

tri_out = tri['triangle_count']
print tri_out.topk('triangle_count', k=10)

##Because GraphLab Create SGraphs use directed edges, the shortest path toolkit also finds the shortest directed paths to a source vertex.

sssp = gl.shortest_path.create(sg, source_vid='Microsoft')
sssp.get_path(vid='Weyerhaeuser', show=True,
              highlight=['Microsoft', 'Weyerhaeuser'], arrows=True, ewidth=1.5)