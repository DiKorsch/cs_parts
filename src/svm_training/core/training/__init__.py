
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from svm_training.core.training.classifiers import ClfInitializer

def train_score(X, y, X_val, y_val, clf_class, n_parts=1, scale=False, **kwargs):
	logging.debug("Training")
	logging.debug(X.shape)
	logging.debug(y)

	logging.debug("Validation")
	logging.debug(X_val.shape)
	logging.debug(y_val)

	if scale:
		logging.info("Scaling Data...")
		scaler = MinMaxScaler()
		X = scaler.fit_transform(X)
		X_val = scaler.transform(X_val)
	clf = clf_class(n_parts=n_parts, **kwargs)

	logging.info("Training {} Classifier...".format(clf.__class__.__name__))
	clf.fit(X, y)
	train_accu = clf.score(X, y)
	logging.info("Training Accuracy: {:.4%}".format(train_accu))
	return clf, clf.score(X_val, y_val)


def evaluate_parts(opts, train, val, key, shuffle=False):

	assert not opts.sparse, "Sparsity is not supported here!"

	train_feats = train.features
	val_feats = val.features

	def inner(X, y, X_val, y_val, suffix):
		if shuffle:
			suffix += "_shuffled"

		class_init = ClfInitializer.new(opts)

		clf, score = train_score(X, y, X_val, y_val, class_init,
			n_parts=train_feats.shape[1],
			scale=opts.scale_features)

		logging.info("Accuracy {}: {:.4%}".format(suffix, score))

		if not opts.no_dump:
			class_init.dump(clf, opts.output, key=key, suffix=suffix)

	if shuffle:
		logging.info("Shuffling features")
		train_feats = train_feats.copy()
		val_feats = val_feats.copy()

		for f in train_feats:
			np.random.shuffle(f[:-1])

		for f in val_feats:
			np.random.shuffle(f[:-1])

	y, y_val = train.labels, val.labels

	X = train_feats.reshape(len(train), -1)
	X_val = val_feats.reshape(len(val), -1)
	inner(X, y, X_val, y_val, "all_parts")

	if opts.eval_local_parts:
		X = train_feats[:, :-1, :].reshape(len(train), -1)
		X_val = val_feats[:, :-1, :].reshape(len(val), -1)
		inner(X, y, X_val, y_val, "local_parts")

def evaluate_global(opts, train, val, key):

	X, y = train.features[:, -1, :], train.labels
	X_val, y_val = val.features[:, -1, :], val.labels
	suffix = "glob_only"

	if opts.sparse:
		suffix += "_sparse_coefs"
		opts.classifier = "svm"

	class_init = ClfInitializer.new(opts)

	clf, score = train_score(X, y, X_val, y_val, class_init,
		n_parts=1,
		scale=opts.scale_features)

	logging.info("Accuracy {}: {:.2%}".format(suffix, score))
	if opts.sparse:
		sparsity = (clf.coef_ != 0).sum(axis=1)
		n_feats = clf.coef_.shape[1]

		logging.info("===== Feature selection sparsity ====")
		logging.info("Absolute:   {:.2f} +/- {:.4f}".format(
			sparsity.mean(), sparsity.std()))
		logging.info("Percentage: {:.2%} +/- {:.4%}".format(
			sparsity.mean()/n_feats, sparsity.std()/n_feats))
		logging.info("=====================================")

	if not opts.no_dump:
		class_init.dump(clf, opts.output, key=key, suffix=suffix)


def l2_norm_feats(feats):
	assert feats.ndim == 3, "Wrong number of dimensions!"
	l2_norm = np.sqrt((feats ** 2).sum(axis=-1, keepdims=True))
	feats /= l2_norm

def evaluate(opts, train, val, key):
	n_parts = train.features.shape[1]

	if opts.l2_norm:
		l2_norm_feats(train.features)
		l2_norm_feats(val.features)

	if n_parts == 1:
		evaluate_global(opts, train, val, key)
	else:
		evaluate_parts(opts, train, val, key, shuffle=opts.shuffle_part_features)
