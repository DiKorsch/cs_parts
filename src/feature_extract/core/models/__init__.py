import numpy as np
# import logging

from chainer_addons.models import PrepareType
from chainer.cuda import to_cpu

# from chainer_addons.models import ResnetLayers, VGG19Layers, InceptionV3
# from cvargparse.utils.enumerations import BaseChoiceType
# from functools import partial

from feature_extract.core.models import tf_inception #import InceptionV3 as InceptionV3_TF

from abc import ABC, abstractmethod

class ModelWrapper(object):
	def __init__(self, model, **kwargs):
		super(ModelWrapper, self).__init__()
		self.model = model
		self.load(**kwargs)

	@property
	def is_tf_model(self):
		return isinstance(self.model, tf_inception.InceptionV3)

	def __call__(self, batch):
		X = np.array(batch)
		n, n_parts, c, h, w = X.shape
		X = X.reshape((-1, c, h, w))

		if self.is_tf_model:
			import pdb; pdb.set_trace()
			# transpose from [n, c, h, w] to [n, h, w, c]
			batch_feats = self.model(X.transpose(0, 2, 3, 1))
		else:
			var = self.model.xp.array(X)
			batch_feats = self.model(var, self.model.meta.feature_layer)
			if isinstance(batch_feats, tuple):
				batch_feats = batch_feats[0]

			pred = self.model.clf_layer(batch_feats)
			batch_feats = to_cpu(batch_feats.array)
			pred = to_cpu(pred.array)

		feat = batch_feats.reshape((n, -1, self.model.meta.feature_size))
		y = pred.argmax(axis=-1).reshape((n, -1))
		return feat, y


	def load(self, weights, n_classes, device=-1):
		if self.is_tf_model:
			import pdb; pdb.set_trace()
		else:
			self.model.load_for_inference(
				weights=weights,
				n_classes=n_classes,
				strict=True,
			)
			if device >= 0:
				self.model.to_gpu(device)


# class ModelType(BaseChoiceType):
# 	ResNet = ChainerModelWrapper(ResnetLayers)
# 	VGG19 = ChainerModelWrapper(VGG19Layers)
# 	Inception = ChainerModelWrapper(InceptionV3)
# 	Inception_TF = TFModelWrapper(InceptionV3_TF)

# 	Default = ChainerModelWrapper(ResnetLayers)

