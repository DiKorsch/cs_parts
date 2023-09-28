#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")  # noqa: E701

import chainer
import logging
import numpy as np
import sys
import typing as T

from chainer_addons.models import PrepareType
from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import GPUParser
from cvdatasets import AnnotationType
from cvdatasets.dataset.image import Size
from cvdatasets.utils import new_iterator
from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args
from cvmodelz.models import ModelFactory

from functools import partial
from pathlib import Path
from tqdm.auto import tqdm


def add_modules(paths: T.List[str]):
	for path in paths:
		logging.info(f"Adding {path} to search paths")
		sys.path.append(path)

def extract_features(it, wrapped_model, n_batches):
	logging.info("Extracting features")
	bar = tqdm(enumerate(it), total=n_batches,
		desc="Extracting features")
	data = it.dataset
	n_samples = len(data)

	feats = np.zeros((n_samples, data.n_crops, wrapped_model.model.meta.feature_size), dtype=np.float32)
	preds = np.zeros((n_samples, data.n_crops), dtype=np.int32)

	labs = np.expand_dims(data.labels, axis=1).repeat(data.n_crops, axis=1)



	for batch_i, batch in bar:
		batch_feats, pred = wrapped_model(batch)
		i = batch_i * it.batch_size
		n = batch_feats.shape[0]
		feats[i : i + n] = batch_feats
		preds[i : i + n] = pred - it.dataset.label_shift

		# print(preds[i:i+n].ravel())
		# print(labs[i:i+n].ravel())
		# import pdb; pdb.set_trace()
		curr_accu = (preds[:i+n] == labs[:i+n]).mean()
		bar.set_description(f"Extracting features (Accuracy: {curr_accu:.2%})")

	return feats

def train_svm(args):
	from svm_training.core import Trainer


def parse_args():

	parser = GPUParser([
		Arg("--load_from", nargs="*",
			default=[
				"../01_svm_training",
				"../02_cs_parts_estimation",
				"../03_feature_extraction",
			]),
	])

	parser.add_args([
		Arg("--subset", choices=["train", "test"], default=None),
		Arg("--augment_positions", action="store_true"),
		Arg("--center_crop_on_val", action="store_true"),
	], group_name="Extraction arguments")

	add_dataset_args(parser)
	add_model_args(parser)

	parser.add_args(ArgFactory()\
		.batch_size()\
		.debug()\
		.seed())

	return parser.parse_args()

def load_dataset(args, annot, size, prepare):
	from feature_extract.core.dataset import Dataset

	data = annot.new_dataset(
		subset=args.subset,
		dataset_cls=Dataset,

		opts=args,
		prepare=prepare,
		size=size
	)

	n_samples = len(data)
	logging.info("Loaded \"{}\"-parts dataset with {} samples from \"{}\"".format(
		args.parts, n_samples, annot.root))

	it, n_batches = new_iterator(data,
		args.n_jobs, args.batch_size,
		repeat=False, shuffle=False)

	return it, n_batches

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
	add_modules(args.load_from)
	print(args)
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

	it, n_batches = load_dataset(args, annot=annot, size=size, prepare=prepare)


	with chainer.using_config("train", False), chainer.no_backprop_mode():
		extract_features(it, model, n_batches)

	train_svm(args)


MB = 1024**2
chainer.cuda.set_max_workspace_size(1024 * MB)
chainer.config.cv_resize_backend = "cv2"
main(parse_args())
