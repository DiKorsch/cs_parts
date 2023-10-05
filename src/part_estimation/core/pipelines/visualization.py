import logging
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from part_estimation.core.pipelines.base import BasePipeline

class VisualizationPipeline(BasePipeline):

	def __init__(self, *, plot_topk_grads=False, peak_size=None, **kwargs):
		super(VisualizationPipeline, self).__init__( **kwargs)

		self.plot_topk_grads = plot_topk_grads
		self.peak_size = peak_size

		K = self.extractor.K

		self.n_cols = int(np.ceil(np.sqrt(K)))
		self.n_rows = int(np.ceil(K / self.n_cols))

	def __call__(self, propagator):

		for i, im, (pred_grad, full_grad), (pred, gt) in propagator:

			logging.debug("predicted class: {}, GT class: {}".format(pred, gt))
			title ="Original Image [predicted: {}, GT: {}]".format(pred, gt)

			spec = GridSpec(2 * self.n_rows, 2 * self.n_cols)
			fig = plt.figure(figsize=(16, 9))
			ax0 = plt.subplot(spec[ 0:self.n_rows, 0:self.n_cols ])

			self.imshow(im, ax=ax0, title=title)
			self.plot_gradient(im, pred_grad, spec=spec)

			plt.tight_layout()
			plt.show()
			plt.close()

			############### Plots Top-k Gradients ###############
			if self.plot_topk_grads:
				grads_coefs_preds = list(zip(
					topk_pred_im_grad,
					topk_pred_coefs,
					topk_preds[i]
				))

				_coefs = np.array([c[i] for c in topk_pred_coefs])
				fig, ax = plt.subplots(figsize=(16, 9))
				# reverse first axis, so that top prediction is displayed first
				ax.imshow(_coefs[::-1] != 0, aspect="auto")
				for k, (_grad, _coef, _pred) in enumerate(reversed(grads_coefs_preds), 1):
					_plot_gradient(
						im, _grad[i],
						ax=None,
						title="Pred #{} Gradient (Class {})".format(k, _pred),
					)
				plt.show()
				plt.close()
			#####################################################


	def imshow(self, im, ax=None, title=None, figsize=(32, 18), **kwargs):
		if ax is None:
			fig, ax = plt.subplots(figsize=figsize)
		ax.imshow(im, **kwargs)
		ax.axis("off")

		if title is not None:
			ax.set_title(title)

		return ax



	def plot_gradient(self, im, grad, spec=None):

		ax1 = plt.subplot(spec[self.n_rows:, :self.n_cols])
		ax2 = plt.subplot(spec[:self.n_rows, self.n_cols:2*self.n_cols])

		grad = self.extractor.corrector(grad)

		ax2 = self.imshow(im, ax=ax2, title="Gradient")
		ax2 = self.imshow(grad, ax=ax2, alpha=0.7)

		thresh_mask = self.extractor.thresh_type(im, grad)
		new_grad = np.zeros_like(grad)
		new_grad[thresh_mask] = grad[thresh_mask]

		# new_grad = grad.copy()
		ax1 = self.imshow(new_grad, ax=ax1, cmap=plt.cm.gray, alpha=1.0)
		# ax = self.imshow(thresh_mask, ax=ax, cmap=plt.cm.Reds, alpha=0.4)

		if self.extractor.K is None or self.extractor.K <= 0:
			return

		cmap = plt.cm.viridis_r

		ys, xs = init_coords = self.extractor.cluster_init(grad, self.extractor.K)
		ax1.scatter(xs, ys, marker="x", color="black")

		centers, labs = self.extractor.cluster_saliency(im, grad)
		boxes = self.extractor.get_boxes(centers, labs, grad)

		for c, box in boxes:
			ax1.add_patch(Rectangle(
				*box, fill=False,
				linewidth=3,
				color=cmap(c / len(boxes))))

		self.imshow(labs, ax1, cmap=cmap, alpha=0.3)

		for i in range(self.extractor.K):
			row, col = np.unravel_index(i, (self.n_rows, self.n_cols))
			_ax = plt.subplot(spec[row + self.n_rows, col + self.n_cols])
			_c, ((x, y), w, h) = boxes[i]
			x,y,w,h = map(int, [x,y,w,h])
			self.imshow(im[y:y+h, x:x+w], _ax, title="Part #{}".format(i+1))

		if self.peak_size is not None:
			peaks = peak_local_max(grad, min_distance=self.peak_size, exclude_border=False)

			ys, xs = peaks.T
			ax.scatter(xs, ys, marker="x", c="blue")

		return ax1


def visualize_coefs(coefs, **kwargs):
	fig, ax = plt.subplots(3, 1)
	ax[0].set_title("Selected features per class")
	ax[0].set_xlabel("Feature Dimension")
	ax[0].set_ylabel("Class")
	ax[0].imshow(coefs != 0, aspect="auto")

	ax[1].set_title("Selections per feature dimension")
	ax[1].set_xlabel("Feature Dimension")
	ax[1].set_ylabel("# of selections")
	# ax[1].imshow((coefs != 0).sum(axis=0, keepdims=True), aspect="auto")
	ax[1].scatter(range(coefs.shape[1]), (coefs != 0).sum(axis=0))

	ax[2].set_title("Number of selected features per class")
	ax[2].set_xlabel("Class")
	ax[2].set_ylabel("# of features")
	ax[2].bar(range(len(coefs)), (coefs != 0).sum(axis=1))

	plt.show()
	plt.close()

