#-*- coding: UTF-8 -*-
import graphlab as gl

# Load the MNIST data (from an S3 bucket)
# data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/train')
# test_data = gl.SFrame('https://static.turi.com/datasets/mnist/sframe/test')
#
#
# data.save('mnist-data')
# test_data.save('mnist_test')

data = gl.SFrame('mnist-data')
test_data = gl.SFrame('mnist_test')
# Random split the training-data
training_data, validation_data = data.random_split(0.8)

# Make sure all images are of the same size (Required by Neuralnets) so nb!!
for sf in [training_data, validation_data, test_data]:
  sf['image'] = gl.image_analysis.resize(sf['image'], 28, 28, 1)

net = gl.deeplearning.get_builtin_neuralnet('mnist')

print "Layers of the network "
print "--------------------------------------------------------"
print net.layers

print "Parameters of the network "
print "--------------------------------------------------------"
print net.params


# model = gl.neuralnet_classifier.create(training_data, target='label',
#                                        network = net,
#                                        validation_set=validation_data,
#                                        metric=['accuracy', 'recall@2'],
#                                        max_iterations=3)
#
# model.save('./model_neuralnet_classifier_model')

model = gl.load_model('./model_neuralnet_classifier_model')
##用模型来预测
predictions = model.classify(test_data)
print predictions

##最高范围值推荐
pred_top2 = model.predict_topk(test_data, k=2)
print pred_top2


##评估
result = model.evaluate(test_data)
print "Accuracy         : %s" % result['accuracy']
print "Confusion Matrix : \n%s" % result['confusion_matrix']



##We also provide a model trained on Imagenet
# imagenet_path = 'https://static.turi.com/models/imagenet_model_iter45'
# imagenet_model = gl.load_model(imagenet_path)
# imagenet_model.save('./imagenet_model_imagenet')

imagenet_model = gl.load_model('./imagenet_model_imagenet')

data['image'] = gl.image_analysis.resize(data['image'], 256, 256, 3)
data['imagenet_features'] = imagenet_model.extract_features(data)

# Now, let's build a new classifier on top of extracted features
m = gl.classifier.create(data,
                         features = ['imagenet_features'],
                         target='label')