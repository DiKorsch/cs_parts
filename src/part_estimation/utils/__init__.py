import numpy as np
import chainer.functions as F

from chainer.backends import cuda

def saliency_to_im(saliency, chan_axis=0, xp=np, keepdims=True):
	"""Compute absolute mean over the channel axis"""
	return xp.abs(saliency).mean(axis=chan_axis, keepdims=keepdims)

def normalize(im, axis=(1,2)):
	im = im - im.min(axis=axis, keepdims=True)
	chan_max = im.max(axis=axis, keepdims=True)
	if 0 in chan_max:
		return im
	else:
		return im / chan_max

def prepare_back(im, swap_channels=True):
	im = im.array if hasattr(im, "array") else im
	im = normalize(cuda.to_cpu(im))
	if swap_channels:
		im = im[::-1]
	return im.transpose(1, 2, 0)


def topk_decision(X, y, clf, topk):
	decs = clf.decision_function(X)
	topk_preds = np.argsort(decs)[:, -topk:]
	topk_accu = (topk_preds == np.expand_dims(y, 1)).max(axis=1).mean()
	return topk_preds, topk_accu
