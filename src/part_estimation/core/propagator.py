import chainer.functions as F
import logging
import numpy as np

from chainer.backends import cuda
from contextlib import contextmanager
from multiprocessing.dummy import Pool
from multiprocessing.pool import AsyncResult

from part_estimation.utils import topk_decision
from part_estimation.utils import prepare_back
from part_estimation.utils import saliency_to_im

class ImageGradient(object):
	"""
		Computes image gradients from given features w.r.t. image
		based on a model and an optional coeffiecient mask
	"""

	def __init__(self, model, feats, ims):
		super(ImageGradient, self).__init__()
		self.model = model
		self.feats = feats
		self.ims = ims

	def __call__(self, coefs=None):

		self.ims.grad = None
		self.model.cleargrads()

		if coefs is None:
			F.sum(self.feats).backward()
		else:
			F.sum(self.feats[np.where(coefs)]).backward()

		assert self.ims.grad is not None, "Backprop mode is off?"
		return self.ims.grad


class Propagator(object):

	def __init__(self, model, clf, scaler, topk, swap_channels=True, n_jobs=1):
		super(Propagator, self).__init__()
		self.model = model
		self.clf = clf
		self.topk = topk
		self.swap_channels = swap_channels
		self.scaler = scaler

		self.pool = Pool(n_jobs) if n_jobs >= 1 else None

		self.reset()

	def __getstate__(self):
		self_dict = self.__dict__.copy()
		del self_dict['pool']
		return self_dict

	def __setstate__(self, state):
		self.__dict__.update(state)

	@property
	def coefs(self):
		return self.clf.coef_

	def reset(self):
		self.ims = self.labs = self.topk_preds = None
		self.full_im_grad = self.pred_im_grad = None

	@contextmanager
	def __call__(self, feats, ims, labs):

		self.ims = ims
		self.labs = labs

		_feats = cuda.to_cpu(feats.array)
		labs = cuda.to_cpu(labs)

		self.topk_preds = topk_preds = self.evaluate_batch(_feats, labs)

		topk_pred_coefs = [self.coefs[p] for p in topk_preds.T]
		pred_coefs = topk_pred_coefs[-1]

		im_grad = ImageGradient(self.model, feats, ims)

		if self.pool is None:

			self.full_im_grad = im_grad()

			self.pred_im_grad = im_grad(topk_pred_coefs[-1] != 0)

		else:

			self.full_im_grad = self.pool.apply_async(im_grad)

			self.pred_im_grad = self.pool.apply_async(im_grad,
				args=(topk_pred_coefs[-1] != 0,))


		yield self

		self.reset()

	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.i >= len(self.ims):
			raise StopIteration

		im = self.prepare_back(self.ims)
		pred_grad = self.prepare_back(self.pred_im_grad, is_grad=True)
		full_grad = self.prepare_back(self.full_im_grad, is_grad=True)

		pred, gt = self.topk_preds[self.i, -1], self.labs[self.i]

		self.i += 1
		return self.i - 1, im, (pred_grad, full_grad), (int(pred), int(gt))


	def prepare_back(self, batch, is_grad=False):
		if isinstance(batch, AsyncResult):
			batch = batch.get()

		im = batch[self.i]
		if is_grad:
			im = saliency_to_im(im, xp=self.model.xp)
		return prepare_back(im, swap_channels=self.swap_channels)

	def evaluate_batch(self, feats, gt):

		if self.scaler is not None:
			feats = self.scaler.transform(feats)

		topk_preds, topk_accu = topk_decision(feats, gt,
			clf=self.clf, topk=self.topk)

		logging.debug(" | ".join([
			f"Batch Accuracy: {np.mean(topk_preds[:, -1] == gt):.4%} (Top1)",
			f"| {topk_accu:.4%} (Top{self.topk})" \
			f"{np.sum(topk_preds[:, -1] == gt): 3d} / {len(feats): 3d}"
		]))

		return topk_preds
