import graphlab as gl
from graphlab import mxnet as mx
import os
##mx.pretrained_model.download_model('https://static.turi.com/models/mxnet_models/release/image_classifier/imagenet1k_inception_bn-1.0.tar.gz')

mx.pretrained_model.list_models()

image_classifier = mx.pretrained_model.load_model('imagenet1k_inception_bn', ctx=mx.gpu(0))

# Load image data into SFrame
data_file = 'cats_dogs_sf'
if os.path.exists(data_file):
    sf = gl.load_sframe(data_file)
else:
    url = 'https://static.turi.com/datasets/' + data_file
    sf = gl.load_sframe(url)
    sf.save(data_file)


# Predict using the pretrained image classifier
prediction = image_classifier.predict_topk(sf['image'], k=1)

# Extract features from images
features = image_classifier.extract_features(sf['image'])