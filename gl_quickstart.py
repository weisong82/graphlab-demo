import graphlab as gl

# url = 'https://static.turi.com/datasets/millionsong/song_data.csv'
# songs = gl.SFrame.read_csv(url)
# songs.show()
#
# songs['year'].mean()
#
# songs['num_words'] = songs['title'].apply(lambda x: len(x.split(' ')))
#
# songs.groupby('artist_name', {'total': gl.aggregate.COUNT})

url = 'https://static.turi.com/datasets/regression/Housing.csv'
x = gl.SFrame.read_csv(url)
m = gl.linear_regression.create(x, target='price')

eval()