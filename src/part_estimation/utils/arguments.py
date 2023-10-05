from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import GPUParser


from cluster_parts.utils import FeatureComposition
from cluster_parts.utils import FeatureType
from cluster_parts.utils import ThresholdType

from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args

def parse_args():
	parser = GPUParser()

	add_dataset_args(parser)
	add_model_args(parser)

	factory = ArgFactory([

		Arg("trained_svm", type=str,
			help="Trained L1 SVM"),

		Arg("--scale_features", action="store_true"),
		Arg("--visualize_coefs", action="store_true"),

		Arg("--topk", type=int, default=5),
		Arg("--extract", type=str, nargs=2,
			help="outputs to store extracted part locations"),

		Arg("--fit_object", action="store_true"),
		Arg("--no_center_crop_on_val", action="store_true"),

		ThresholdType.as_arg("thresh_type",
			help_text="type of gradient thresholding"),

		FeatureType.as_arg("feature_composition",
			nargs="+", default=FeatureComposition.Default,
			help_text="composition of features"),

		Arg("--K", type=int, default=4),

		Arg("--gamma", type=float, default=0.7,
			help="Gamma-Correction of the gradient intesities"),

		Arg("--sigma", type=float, default=5,
			help="Gaussian smoothing strength"),
	])\
	.batch_size()

	parser.add_args(factory, group_name="Part estimation arguments")

	return parser.parse_args()
