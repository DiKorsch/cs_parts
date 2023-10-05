import matplotlib.pyplot as plt
import numpy as np

def show_feature_stats(feats, key=None, part_names=None):
	fig, axs = plt.subplots(2, figsize=(16,9))

	axs[0].set_title("Average Feature Values ({})".format(key))
	axs[0].scatter(range(feats.shape[-1]), feats.mean(axis=(0,1)))

	feature_norms = np.sqrt(np.sum(feats ** 2, axis=-1))

	axs[1].set_title("Average Part Feature Lengths ({})".format(key))
	axs[1].boxplot(feature_norms, showfliers=False)

	if part_names is not None:
		axs[1].set_xticklabels(part_names, rotation=75)

	plt.show()
	plt.close()
