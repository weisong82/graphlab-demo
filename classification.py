import graphlab as gl

# Load the data
##data =  gl.SFrame('https://static.turi.com/datasets/regression/yelp-data.csv')
data =  gl.SFrame('/Users/wei/code/pycharm/yelp-data.csv')

# Restaurants with rating >=3 are good
data['is_good'] = data['stars'] >= 3

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Automatically picks the right model based on your data.
model = gl.classifier.create(train_data, target='is_good',
                             features = ['user_avg_stars',
                                         'business_avg_stars',
                                         'user_review_count',
                                         'business_review_count'])

# Generate predictions (class/probabilities etc.), contained in an SFrame.
predictions = model.classify(test_data)

# Evaluate the model, with the results stored in a dictionary
results = model.evaluate(test_data)