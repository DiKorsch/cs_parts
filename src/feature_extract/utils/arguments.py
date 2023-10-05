from cvargparse import Arg
from cvargparse import ArgFactory
from cvargparse import GPUParser

from chainer_addons.links import PoolingType
from chainer_addons.models import PrepareType

from cvdatasets.utils import read_info_file
from cvfinetune.parser import add_dataset_args
from cvfinetune.parser import add_model_args

def extract_args():
	parser = GPUParser()

	add_dataset_args(parser)
	add_model_args(parser)

	parser.add_args(ArgFactory()\
			.batch_size()\
			.debug()\
			.seed())

	parser.add_args([
		Arg("output", help="output folder for the extracted features"),

		Arg("--subset", choices=["train", "test"], default=None),

		Arg("--center_crop_on_val", action="store_true"),

		Arg("--augment_positions", action="store_true"),
		Arg("--compress_output", action="store_true"),

		Arg("--is_bbox_parts", action="store_true"),


	], group_name="Extraction options")

	return parser.parse_args()
