import numpy as np

def augmented_positions(im_obj, scale):
	im, parts, labels = im_obj.as_tuple()
	h, w = im.shape[:2]

	crop_h, crop_w = h * scale, w * scale

	d_h, d_w = int(crop_h / 4), int(crop_w / 4)

	diff = np.array([
		[-d_w,-d_h],
		[ d_w,-d_h],
		[-d_w, d_h],
		[ d_w, d_h],
	])

	for d in diff:
		aug_im_obj = im_obj.copy()

		xy = aug_im_obj.parts[:, 1:3]
		xy += d
		xy = np.maximum(xy, 0)
		xy = np.minimum(xy, np.array([w, h]) - 1)

		aug_im_obj.parts[:, 1:3] = xy
		yield aug_im_obj

