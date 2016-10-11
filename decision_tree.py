import graphlab as gl

# Load the data
data =  gl.SFrame.read_csv('https://static.turi.com/datasets/xgboost/mushroom.csv')

# Label 'p' is edible
data['label'] = data['label'] == 'p'

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create a model.
model = gl.decision_tree_regression.create(train_data, target='label',
                                           max_depth =  3)

# Save predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and save the results into a dictionary
results = model.evaluate(test_data)

model.show(view="Tree", tree_id=0)
