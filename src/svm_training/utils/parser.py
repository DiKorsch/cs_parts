from chainer_addons.models import PrepareType

from cvargparse import BaseParser, Arg

from cvdatasets.utils import read_info_file
from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args

def _base_parser():
	parser = BaseParser()

	add_dataset_args(parser)
	# add_model_args(parser)
	return parser

def train_args():
	parser = _base_parser()

	parser.add_args([
		Arg("--feature_model",
			default="resnet", choices=["inception", "resnet"],
			help="type of the model"),

		Arg("--classifier", "-clf", default="svm",
			choices=["svm", "logreg", "fv"]),

		Arg("--C", type=float, default=0.1,
			help="classifier regularization parameter"),

		Arg("--max_iter", type=int, default=200,
			help="maximum number of training iterations"),

		Arg("--load", type=str,
			help="initial weights and biases to use as initialization"),

		Arg("--show_feature_stats", action="store_true"),
		Arg("--shuffle_part_features", action="store_true"),

		Arg("--sparse", action="store_true",
			help="Use LinearSVC with L1 regularization for sparse feature selection"),

		Arg("--scale_features", action="store_true"),
		Arg("--l2_norm", action="store_true"),

		Arg("--eval_local_parts", action="store_true"),
		Arg("--no_dump", action="store_true"),


		Arg("--output", default=".out"),

	], group_name="SVM options")

	return parser.parse_args()


def predict_args():
	parser = _base_parser()

	parser.add_args([
		Arg("weights"),

		Arg("--feature_model",
			default="resnet", choices=["inception", "resnet"],
			help="type of the model"),

		Arg("--subset", choices=[
			"train", "test"
		]),

		Arg("--evaluate", action="store_true"),
		Arg("--scale_features", action="store_true"),

		Arg("--no_export", action="store_true"),

		Arg("--label_shift", type=int, default=0),

		Arg("--output", default="predictions.csv"),

	], group_name="Prediction options")

	return parser.parse_args()
