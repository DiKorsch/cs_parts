import logging
import numpy as np

from abc import ABC

# from chainer_addons.dataset import PreprocessMixin, AugmentationMixin

from chainercv import transforms as tr
from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import BasePartMixin
from cvdatasets.dataset import IteratorMixin
from cvdatasets.dataset import ImageProfilerMixin
from cvdatasets.dataset import TransformMixin

from feature_extract.utils.preprocessing import augmented_positions

class Dataset(
	ImageProfilerMixin,
	TransformMixin,
	BasePartMixin,
	IteratorMixin,
	AnnotationsReadMixin,
	):
	label_shift = None

	def __init__(self, *, opts, prepare, **kwargs):
		super(Dataset, self).__init__(**kwargs)

		self._crop_scales = self._annot.dataset_info.scales or []

		self._augment_positions = opts.augment_positions
		self._center_crop_on_val = opts.center_crop_on_val
		self.prepare = prepare
		self.label_shift = opts.label_shift

		logging.info("There will be {} crops (on {} scales) from {} parts".format(
			self.n_crops, self.n_scales, self.n_parts
		))

	@property
	def n_parts(self):
		return len(self._annot.part_names)

	@property
	def n_scales(self):
		return len(self._crop_scales)

	@property
	def n_positions(self):
		return 1 if not self._augment_positions else 4

	@property
	def n_crops(self):
		return self.n_parts * self.n_scales * self.n_positions + 1

	def generate_crops(self, im_obj):
		for scale in self._crop_scales:
			if self._augment_positions:
				for aug_im_obj in augmented_positions(im_obj, scale):
					for crop in aug_im_obj.visible_crops(scale):
						yield crop
			else:
				for crop in im_obj.visible_crops(scale):
					yield crop

		yield im_obj.im_array

	def transform(self, im_obj):
		crops = []
		self._profile_img(im_obj.im_array, "input image")
		for i, crop in enumerate(self.generate_crops(im_obj)):
			if i == 0:
				self._profile_img(crop, "before prepare")
			crop = self.prepare(crop)
			if i == 0:
				self._profile_img(crop, "after prepare")

			if self._center_crop_on_val:

				crop = tr.center_crop(crop, size=self._size)
				if i == 0:
					self._profile_img(crop, "center cropped")

			crops.append(crop)

		return crops, im_obj.label + self.label_shift

	def get_example(self, i):
		ims, label = super(Dataset, self).get_example(i)
		ims = np.array(ims) * 2 - 1
		self._profile_img(ims, "result")
		return ims, label

from chainer.dataset import DatasetMixin
class TFDataset(DatasetMixin):

	def __init__(self, opts, annot, prepare, **foo):
		raise RuntimeError("FIX ME!")
		super(TFDataset, self).__init__()
		assert callable(prepare), "prepare must be callable!"
		self.uuids = annot.uuids
		self._annot = annot
		self._crop_scales = opts.scales
		self.prepare = prepare

	@property
	def n_parts(self):
		return len(self._annot.part_names)

	@property
	def n_scales(self):
		return len(self._crop_scales)

	@property
	def n_crops(self):
		return self.n_parts * self.n_scales + 1


	def __len__(self):
		return len(self.uuids)


	def _get(self, method, i):
		return getattr(self._annot, method)(self.uuids[i])

	def get_example(self, i):
		methods = ["image", "parts", "label"]
		im_path, parts, label = [self._get(m, i) for m in methods]

		im = self.prepare(im_path)
		part_crops = [np.zeros_like(im) for _ in range(len(parts))]
		return np.array(part_crops + [im])#, label
