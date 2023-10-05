import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from svm_training.core.training.classifiers import ClfInitializer


def l2_norm_feats(feats):
	assert feats.ndim == 3, "Wrong number of dimensions!"
	l2_norm = np.sqrt((feats ** 2).sum(axis=-1, keepdims=True))
	feats /= l2_norm

def shuffle_feats(feats):
	for f in feats:
		np.random.shuffle(f[:-1])

class Trainer(object):

	@classmethod
	def new(cls, opts, train, val, key):
		if opts.sparse:
			opts.classifier = "svm"

		return cls(train, val, key,
			class_init=ClfInitializer.new(opts),

			shuffle_parts=opts.shuffle_part_features,
			l2_norm=opts.l2_norm,
			sparse=opts.sparse,
			scale_features=opts.scale_features,
			eval_local_parts=opts.eval_local_parts,
			no_dump=opts.no_dump,
			output=opts.output,
		)

	@property
	def n_parts(self):
		return self.train_feats.shape[1]

	def __init__(self, train_data, val_data, key, *,
		class_init,
		output="svms",
		sparse=False,
		scale=False,
		shuffle_parts=False,
		eval_local_parts=False,
		l2_norm=False,
		scale_features=False,
		no_dump=False,
		):
		super(Trainer, self).__init__()

		self.train_feats = train_data.features
		self.val_feats = val_data.features

		self.scaler = None
		if scale_features:
			logging.info("Initialized MinMaxScaler")
			self.scaler = MinMaxScaler()

		self.suffix, self.X, self.X_val = self.prepare_features(
			l2_norm=l2_norm,
			sparse=sparse,
			shuffle_parts=shuffle_parts,
			eval_local_parts=eval_local_parts
		)

		self.y, self.y_val = train_data.labels, val_data.labels

		self.key = key
		self.sparse = sparse

		self.output = output
		self.no_dump = no_dump


		self.class_init = class_init

	def prepare_features(self, *, l2_norm, shuffle_parts, eval_local_parts, sparse):

		feats = self.train_feats, self.val_feats

		if self.scaler is not None:
			logging.info("Scaling data...")
			self.scaler.fit(feats[0])
			feats = self.scaler.transform(feats[0]), self.scaler.transform(feats[1])

		if l2_norm:
			l2_norm_feats(feats[0])
			l2_norm_feats(feats[1])

		suffix = "glob_only"
		if self.n_parts == 1:
			if sparse:
				suffix += "_sparse_coefs"

			return suffix, feats[0][:, -1], feats[1][:, -1]

		assert not sparse, "Sparsity is not supported for part features!"

		suffix = "all_parts"

		if eval_local_parts:
			feats = self.train_feats[:, :-1], self.val_feats[:, :-1]
			suffix = "local_parts"

		if shuffle_parts:
			logging.info("Shuffling features")
			feats = shuffle_feats(feats[0].copy()), shuffle_feats(feats[1].copy())
			suffix += "_shuffled"

		n_train, n_val = feats[0].shape[0], feats[1].shape[0]
		return suffix, feats[0].reshape(n_train, -1), feats[1].reshape(n_val, -1)


	def new_clf(self, **kwargs):
		return self.class_init(n_parts=self.n_parts, **kwargs)

	def train_score(self, **kwargs):
		logging.debug(f"Training: {self.X.shape} | {self.y}")
		logging.debug(f"Validation: {self.X_val.shape} | {self.y_val}")

		clf = self.new_clf(**kwargs)

		logging.info("Training {} Classifier...".format(clf.__class__.__name__))
		clf.fit(self.X, self.y)
		train_accu = clf.score(self.X, self.y)
		logging.info("Training Accuracy: {:.4%}".format(train_accu))
		return clf, clf.score(self.X_val, self.y_val)

	def evaluate(self):

		clf, score = self.train_score()

		if not self.no_dump:
			self.class_init.dump(clf, self.output, key=self.key, suffix=self.suffix)

		logging.info("Accuracy {}: {:.4%}".format(self.suffix, score))

		if self.sparse:
			sparsity = (clf.coef_ != 0).sum(axis=1)
			n_feats = clf.coef_.shape[1]

			logging.info("===== Feature selection sparsity ====")
			logging.info("Absolute:   {:.2f} +/- {:.4f}".format(
				sparsity.mean(), sparsity.std()))
			logging.info("Percentage: {:.2%} +/- {:.4%}".format(
				sparsity.mean()/n_feats, sparsity.std()/n_feats))
			logging.info("=====================================")

		return clf
