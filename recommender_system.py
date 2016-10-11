import graphlab as gl

ratings = gl.SFrame.read_csv('./ml-20m/ratings.csv')
movies = gl.SFrame.read_csv('./ml-20m/movies.csv')


# training_data,validatation_data = gl.recommender.util.random_split_by_user(ratings, 'userId', 'movieId')
# model = gl.recommender.create(training_data, 'userId', 'movieId')
#
# gl.Model.save(model,'ml-20m-reco-model')

model = gl.load_model('ml-20m-reco-model')

# You can now make recommendations for all the users you've just trained on
#results = model.recommend()
#print results
param = [1,2]
print model.recommend(users=param,k=2)