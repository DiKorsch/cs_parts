import abc
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from cvdatasets import AnnotationType
from cvdatasets.utils import new_iterator

from part_estimation.utils import topk_decision

def evaluate_data(opts, clf, data, subset, scaler):

	X = scaler.transform(data.features[:, -1])
	y = data.labels - opts.label_shift
	pred = clf.decision_function(X).argmax(axis=1)
	logging.info("Accuracy on {} subset: {:.4%}".format(subset, (pred == y).mean()))

	topk_preds, topk_accu = topk_decision(X, y, clf=clf, topk=opts.topk)
	logging.info("Top{}-Accuracy on {} subset: {:.4%}".format(opts.topk, subset, topk_accu))

class IdentityScaler(object):
	"""
		Do not scale the data, just return itself
	"""
	transform = lambda self, x: x

class Data(abc.ABC):

	@abc.abstractmethod
	def __init__(self):
		super(Data, self).__init__()

	@classmethod
	def new(self, opts, clf=None):

		assert opts.parts == "GLOBAL", \
			f"Wrong parts selected: {opts.parts}. Should be \"GLOBAL\""
		parts_key = f"{opts.dataset}_{opts.parts}"

		annot = AnnotationType.new_annotation(opts, load_strict=False)
		logging.info("Loading {} annotations from \"{}\"".format(
			annot.__class__, opts.data))
		logging.info("Using \"{}\"-parts".format(parts_key))


		ds_info = annot.dataset_info
		model_info = annot.info.MODELS[opts.model_type]

		n_classes = ds_info.n_classes + opts.label_shift

		data = annot.new_dataset(subset=None)
		train_data, val_data = map(annot.new_dataset, ["train", "test"])

		if annot.labels.max() > n_classes:
			_, annot.labels = np.unique(annot.labels, return_inverse=True)

		logging.info("Minimum label value is \"{}\"".format(data.labels.min()))

		assert train_data.features is not None and val_data.features is not None, \
			"Features are not loaded!"

		assert val_data.features.ndim == 2 or val_data.features.shape[1] == 1, \
			"Only GLOBAL part features are supported here!"

		if opts.scale_features:
			logging.info("Scaling data on training set!")
			scaler = MinMaxScaler()
			scaler.fit(train_data.features[:, -1])
		else:
			scaler = IdentityScaler()

		it, _ = new_iterator(data,
			opts.n_jobs, opts.batch_size,
			repeat=False, shuffle=False
		)

		if clf is not None:
			for _data, subset in [(train_data, "training"), (val_data, "validation")]:
				evaluate_data(opts, clf, _data, subset, scaler)

		return scaler, it, model_info, n_classes


