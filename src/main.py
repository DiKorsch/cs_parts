#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")  # noqa: E701
# ruff: noqa: E402

import chainer
import cv2
cv2.setNumThreads(1)
import joblib
import logging
import numpy as np
import shutil
import typing as T
import yaml

from chainer.dataset.convert import concat_examples
from chainer_addons.models import PrepareType
from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import GPUParser
from cvdatasets import AnnotationType
from cvdatasets import Annotations
from cvdatasets.dataset.image import Size
from cvdatasets.utils import new_iterator
from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args
from cvmodelz.models import ModelFactory

from cluster_parts.core import BoundingBoxPartExtractor
from cluster_parts.core import Corrector
from cluster_parts.utils import ClusterInitType
from cluster_parts.utils import FeatureComposition
from cluster_parts.utils import FeatureType
from cluster_parts.utils import ThresholdType

from bdb import BdbQuit
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


def parse_args(**kwargs):

	parser = GPUParser([
		Arg.flag("--vacuum",
			help="Set this flag, to delete the output folder after a failed"\
			"or a quitted (from a debugging session) training"),
		Arg("--checkpoint"),
		Arg("--output", default="output"),
	])

	parser.add_args([
		Arg("--augment_positions", action="store_true"),
		Arg("--center_crop_on_val", action="store_true"),
	], group_name="Extraction arguments")

	parser.add_args([
		Arg.flag("--shuffle_part_features"),

		Arg.flag("--no_dump"),

	], group_name="SVM arguments")
	parser.add_args([
		Arg.int("--n_parts", default=4),
		Arg.int("--topk", default=1),
		Arg.flag("--fit_object"),

		FeatureType.as_arg("feature_composition",
			nargs="+", default=FeatureComposition.Default,
			help_text="composition of features"),

		ThresholdType.as_arg("thresh_type",
			help_text="type of gradient thresholding"),

		Arg("--gamma", type=float, default=0.7,
			help="Gamma-Correction of the gradient intesities"),

		Arg("--sigma", type=float, default=5,
			help="Gaussian smoothing strength"),

		Arg.flag("--show_parts_only"),

	], group_name="Part estimation arguments")

	add_dataset_args(parser)
	add_model_args(parser)

	parser.add_args(ArgFactory()\
		.batch_size()\
		.debug()\
		.seed())

	return parser.parse_args(**kwargs)


def load_data(args, annot, size, prepare):
	from feature_extract.core.dataset import Dataset

	@dataclass
	class Datasets:
		annot: Annotations
		train: Dataset
		test: Dataset
		entire_data: Dataset

		def __iter__(self):
			return iter([("train", self.train), ("test", self.test)])

	kwargs = dict(dataset_cls=Dataset,
		opts=args,
		prepare=prepare,
		size=size)

	data = Datasets(annot, *annot.new_train_test_datasets(**kwargs),
		entire_data=annot.new_dataset(subset=None, **kwargs))

	n_train = len(data.train)
	n_test = len(data.test)
	logging.info(
		f"Loaded '{args.parts}'-parts dataset with {n_train} train and {n_test}" \
		f" test samples from '{annot.root}'")


	logging.info("Profiling image processing: ")
	ds = data.train
	with ds.enable_img_profiler():
		ds[np.random.randint(len(ds))]

	return data

def load_model(args, n_classes, model_info, model_root: Path, device):
	from feature_extract.core.models import ModelWrapper

	model = ModelFactory.new(
		model_type=args.model_type,
		input_size=Size(args.input_size))

	n_param = model.count_params()
	logging.info(f"Created model {args.model_type} with {n_param:,d} parameters")
	# logging.debug(model)
	size = Size(model.meta.input_size)

	prepare = partial(PrepareType[args.prepare_type](model),
		swap_channels=args.swap_channels,
		keep_ratio=args.center_crop_on_val,
	)
	logging.info(f"Input size: {size} | PrepareType: {args.prepare_type}")

	if args.weights:
		weights_file = Path(f"ft_{args.dataset}", args.weights)
	else:
		assert args.pretrained_on in model_info.weights, \
				f"Weights for \"{args.pretrained_on}\" pre-training were not found!"
		weights_file = Path(model_info.weights[args.pretrained_on])

	# is absolute path
	if weights_file.is_absolute():
		weights = weights_file
	else:
		weights = model_root / model_info.folder / weights_file

	assert weights.is_file(), f"Could not find weights \"{weights}\""
	logging.info(f"Loading weights from \"{weights}\"")
	n_classes = n_classes + args.label_shift
	wrapped_model = ModelWrapper(model,
		weights=str(weights),
		n_classes=n_classes,
		device=device)

	return wrapped_model, prepare, size

def main(args):

	output_root = Path(args.output)
	output_root.mkdir(exist_ok=True, parents=True)

	with open(output_root / "args.yml", "w") as f:
		yaml.dump(args.__dict__, f, sort_keys=True)

	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()

	annot = AnnotationType.new_annotation(args, load_strict=False)

	info = annot.info

	model, prepare, size = load_model(args,
		n_classes=annot.dataset_info.n_classes,
		model_info=info.MODELS[args.model_type],
		model_root=Path(info.BASE_DIR, info.MODEL_DIR),
		device=GPU)

	data = load_data(args, annot=annot, size=size, prepare=prepare)

	clf_opts = ClfOptions(
		classifier="svm",
		key=f"{args.dataset}_{args.parts}_{args.model_type}",
		shuffle_part_features=args.shuffle_part_features,
		sparse=True,
		l2_norm=False,
		eval_local_parts=False,
		no_dump=args.no_dump,
		scale_features=False,
		output=args.output,
	)


	if args.checkpoint:
		ckpt = Checkpoint.load(args.checkpoint, clf_opts)
		features = ckpt.features
		l1_svm, scaler = ckpt.svm, ckpt.scaler
	else:
		features = l1_svm = scaler = None

	if features is None:
		with chainer.using_config("train", False), chainer.no_backprop_mode():
			features = extract_features(model, data, batch_size=128, n_jobs=args.n_jobs)

	# eval_clf(features, model)
	np.savez(output_root / "features.npz", **{
		"train/features": features["train"].features,
		"train/labels": features["train"].labels,
		"test/features": features["test"].features,
		"test/labels": features["test"].labels,
	})


	# logging.info(f"Training baseline classifier with following options: {clf_opts}")
	# _ = train_svm(features, clf_opts)

	if l1_svm is None:
		logging.info(f"Training L1-classifier with following options: {clf_opts}")
		l1_svm, scaler = train_svm(features, clf_opts)
	else:
		X, y = features["train"].features.squeeze(axis=1), features["train"].labels
		X_val, y_val = features["test"].features.squeeze(axis=1), features["test"].labels

		if scaler is not None:
			X, X_val = scaler(X), scaler(X_val)
		train_score = l1_svm.score(X, y)
		test_score = l1_svm.score(X_val, y_val)

		logging.info("Score of the loaded classifier: " \
				f"{train_score:.2%} training | {test_score:.2%} test")
		joblib.dump(l1_svm, output_root / ckpt.svm_fname)

	part_opts = PartOptions(
		K = args.n_parts,
		feature_composition = args.feature_composition,
		fit_object = args.fit_object,

		topk = args.topk,
		swap_channels = args.swap_channels,
		gamma = args.gamma,
		sigma = args.sigma,
		thresh_type = args.thresh_type,
	)

	logging.info(f"Estimating parts with following options: {part_opts}")
	with chainer.using_config("train", False):
		it, n_batches = new_iterator(data.entire_data,
			n_jobs=args.n_jobs, batch_size=args.batch_size,
			repeat=False, shuffle=False)


		keys = ["CS_parts", "noCS_parts"]
		dest = [output_root / out / "parts/part_locs.txt" for out in keys]

		for d in dest:
			d.parent.mkdir(exist_ok=True, parents=True)

		estimate_parts(model, part_opts, clf=l1_svm, iterator=it,
			scaler=scaler,
			prepare=prepare, device=GPU,
			show_parts_only=args.show_parts_only,
			dest=dest
		)



########################
## Feature extraction ##
########################

@dataclass
class Features:
	features: np.ndarray
	labels: np.ndarray

def extract_features(wrapped_model, data, *, n_jobs: int = 3, batch_size: int = 32):

	result = {}
	for key, subset in data:
		it, n_batches = new_iterator(subset, n_jobs, batch_size,
			repeat=False, shuffle=False)

		# logging.info(f"Extracting {key} features")
		bar = tqdm(enumerate(it), total=n_batches, desc=f"Extracting {key} features")
		n_samples = len(subset)

		feats = np.zeros((n_samples, subset.n_crops, wrapped_model.model.meta.feature_size),
			dtype=np.float32)
		preds = np.zeros((n_samples, subset.n_crops), dtype=np.int32)

		labs = np.expand_dims(subset.labels, axis=1).repeat(subset.n_crops, axis=1)

		for batch_i, batch in bar:
			X, y = concat_examples(batch)
			batch_feats, pred = wrapped_model(X)
			i = batch_i * it.batch_size
			n = batch_feats.shape[0]
			feats[i : i + n] = batch_feats
			preds[i : i + n] = pred - it.dataset.label_shift

			curr_accu = (preds[:i+n] == labs[:i+n]).mean()
			bar.set_description(f"Extracting {key} features (Accuracy: {curr_accu:.2%})")

		result[key] = Features(feats, subset.labels)

	return result

################################
## Linear classifier training ##
################################

@dataclass
class ClfOptions:
	classifier: str = "svm"
	C: float = 0.1
	max_iter: int = 200

	sparse: bool = False
	l2_norm: bool = False
	eval_local_parts: bool =False
	no_dump: bool = False
	scale_features: bool = False
	shuffle_part_features: bool = False

	load: str = None
	output: str = None

	key: str = "KEY"

def train_svm(features, opts: ClfOptions):
	from svm_training.core import Trainer
	from svm_training.core.training.classifiers import ClfInitializer

	trainer = Trainer(
		features["train"], features["test"],
		key=opts.key,
		class_init=ClfInitializer.new(opts),
		sparse=opts.sparse,
		l2_norm=opts.l2_norm,
		eval_local_parts=opts.eval_local_parts,
		no_dump=opts.no_dump,
		scale_features=opts.scale_features,
		shuffle_parts=opts.shuffle_part_features,
		output=opts.output
	)
	return trainer.evaluate(), trainer.scaler

#########################
## CS-Parts estimation ##
#########################

@contextmanager
def outputs(destinations):
	if destinations:
		assert destinations is not None, \
			"For extraction output files are required!"
		outputs = [open(out, "w") for out in destinations]
		yield outputs
		[out.close() for out in outputs]
	else:
		logging.warning("Extraction is disabled!")
		yield None, None

@dataclass
class PartOptions:
	K: int
	feature_composition: T.List[str]
	fit_object: bool

	topk: int
	swap_channels: bool
	gamma: float
	sigma: float
	thresh_type: str

def estimate_parts(wrapped_model, opts: PartOptions, *, clf,
	scaler: T.Callable = None,
	scale_features: bool = False,
	show_parts_only: bool = False,
	dest: T.Tuple[str] = None,
	**kwargs
	):

	from part_estimation.core import Propagator
	from part_estimation.core import ExtractionPipeline
	from part_estimation.core import VisualizationPipeline

	logging.info(f"Using following feature composition: {opts.feature_composition}")
	model = wrapped_model.model

	propagator = Propagator(model, clf,
		scaler=scaler,
		topk=opts.topk,
		swap_channels=opts.swap_channels,
		n_jobs=1,
	)

	extractor = BoundingBoxPartExtractor(
		corrector=Corrector(gamma=opts.gamma, sigma=opts.sigma),

		K=opts.K,
		fit_object=opts.fit_object,

		thresh_type=opts.thresh_type,
		cluster_init=ClusterInitType.MAXIMAS,
		feature_composition=opts.feature_composition,

	)


	if show_parts_only:

		pipeline = VisualizationPipeline(
			model=model,
			extractor=extractor,
			propagator=propagator,
			**kwargs
		)
		pipeline.run()
	else:

		with outputs(dest) as files:
			pipeline = ExtractionPipeline(
				model=model,
				extractor=extractor,
				propagator=propagator,
				files=files,
				**kwargs
			)
			pipeline.run()

############################
## Main methods / classes ##
############################

@dataclass
class Checkpoint:
	features: T.Dict[str, Features]
	svm: object
	svm_fname: str
	scaler: T.Callable

	@classmethod
	def load(cls, ckpt, clf_opts: ClfOptions):
		ckpt = Path(ckpt)
		features, svm, scaler = None, None, None
		svm_fname = None

		_feats = ckpt / "features.npz"
		if _feats.exists():
			logging.info(f"Loading features from {_feats}")
			cont = np.load(_feats)

			features = {subset: Features(features=cont[f"{subset}/features"],
										labels=cont[f"{subset}/labels"])
				for subset in ["train", "test"]}

		name = clf_opts.classifier
		key = clf_opts.key
		_clf = ckpt / f"clf_{name}_{key}_glob_only_sparse_coefs.npz"
		if _clf.exists():
			svm_fname = _clf.name
			logging.info(f"Loading classifier from {_clf}")

			svm = joblib.load(_clf)

			if clf_opts.scale_features:
				scaler = MinMaxScaler()

		return cls(features=features, svm=svm, scaler=scaler, svm_fname=svm_fname)


MB = 1024**2
GB = 1024 * MB
chainer.cuda.set_max_workspace_size(1 * GB)
chainer.config.cv_resize_backend = "cv2"

def _move(outfolder: Path, move_to: str, reason: str):
	logging.warning(f"Training did not finish properly: reason={reason}")
	dst = outfolder.parent / move_to / outfolder.name
	if outfolder.exists():
		dst.parent.mkdir(exist_ok=True, parents=True)
		logging.warning(f"Moving training logs to {dst}")
		shutil.move(outfolder, dst)
	print(outfolder, "->", dst)

def _delete(outfolder: Path, vacuum: bool = False):
	if vacuum and outfolder.exists():
		logging.warning(f"Deleting all files from \"{outfolder}\""\
			"because vacuum=True; remove --vacuum to disable this!")
		shutil.rmtree(outfolder)

args = parse_args()

try:
	main(args)

except KeyboardInterrupt:
	_move(Path(args.output), "interrupted", "KeyboardInterrupt")
	raise

except BdbQuit:
	_delete(Path(args.output), args.vacuum)
	raise

except Exception as e:
	_move(Path(args.output), "failed", str(e))
	raise


else:
	logging.info("Finished")
