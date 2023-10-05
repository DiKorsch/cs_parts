import threading
import multiprocessing as mp
import time
import queue

from part_estimation.core.pipelines.base import BasePipeline


def try_to_put(Q, item, n_tries = 8, t0: float = 0.1):
	for n in range(n_tries):
		try:
			Q.put(item, block=False)
		except queue.Full:
			time.sleep(t0)
			t0 = t0 * 2

		else:
			return

	raise queue.Full(f"Failed to put item after {n+1} tries [qsize: {Q.qsize()}]")

class ExtractionPipeline(BasePipeline):

	def __init__(self, *, files, iterator, **kwargs):
		super(ExtractionPipeline, self).__init__(iterator=iterator, **kwargs)
		assert None not in files

		self.pred_out, self.full_out = files
		self.uuids = iterator.dataset.uuids
		self.batch_size = iterator.batch_size


	def to_out(self, im_id, part_id, box, out):
		(x, y), w, h = box

		print(im_id, int(part_id+1), *map(int, map(round, [x,y,w,h])),
			file=out)

	def to_pred_out(self, *args, **kwargs):
		self.to_out(*args, **kwargs, out=self.pred_out)

	def to_full_out(self, *args, **kwargs):
		self.to_out(*args, **kwargs, out=self.full_out)

	def __getstate__(self):
		# self_dict = self.__dict__.copy()
		# del self_dict['pred_out']
		# del self_dict['full_out']
		return dict(
			extractor=self.extractor,
			inqueue=self.inqueue,
			outqueue=self.outqueue,
			worker_done=self.worker_done,
			writer_done=self.writer_done,
		)

	def __setstate__(self, state):
		self.__dict__.update(state)

	def error_callback(self, exc):
		print(f"Error occured: {exc}")

	def run(self):
		n_jobs = self.batch_size // 2
		PoolCls = mp.Pool
		# PoolCls = mp.pool.ThreadPool
		with PoolCls(n_jobs) as pool, mp.Manager() as m:
			self.worker_done = m.Value("b", False)
			self.writer_done = m.Value("b", False)
			self.inqueue = m.Queue(maxsize=self.batch_size * 2)
			self.outqueue = m.Queue(maxsize=self.batch_size * 4)

			self.writer_thread = threading.Thread(target=self.write_result)
			self.writer_thread.deamon = True
			self.writer_thread._state = 0
			self.writer_thread.start()

			results = [pool.apply_async(self.extract, error_callback=self.error_callback)
				for _ in range(n_jobs)]

			super(ExtractionPipeline, self).run()

			self.worker_done.value = True
			for result in results:
				result.wait()

			self.writer_done.value = True
			self.writer_thread.join()

	def __call__(self, prop_iter):

		for i, im, grads, _ in prop_iter:

			im_idx = i + self.batch_i * self.batch_size
			im_uuid = self.uuids[im_idx]

			# try_to_put(self.outqueue, (im_uuid, self.extractor(im, grads)))

			try_to_put(self.inqueue, [im_uuid, im, grads])

	def extract(self):
		while True:
			if self.worker_done.value and self.inqueue.empty():
				break

			if self.inqueue.empty():
				time.sleep(0.1)
				continue

			im_uuid, im, grads = self.inqueue.get()
			# print(f"Processing item {im_uuid} [Qsize: {self.inqueue.qsize()}]")
			try:
				result = im_uuid, self.extractor(im, grads)
				# print(f"Putting result back [Qsize: {self.outqueue.qsize()}]")
			except Exception as e:
				print(f"Processing failed: {e}!")
				result = im_uuid, None

			try_to_put(self.outqueue, result)
			self.inqueue.task_done()

		# print("Exiting worker", self.worker_done.value, self.inqueue.qsize())

	def write_result(self):
		while True:
			if self.writer_done.value and self.outqueue.empty():
				break

			if self.outqueue.empty():
				time.sleep(0.1)
				continue

			im_uuid, parts = self.outqueue.get()

			if parts is not None:
				for pred_part, full_part in zip(*parts):
					self.to_pred_out(im_uuid, *pred_part)
					self.to_full_out(im_uuid, *full_part)

			self.outqueue.task_done()

		# print("Writer exists... {} | {} | {}".format(self.writer_done.value, self.inqueue.qsize(), self.outqueue.qsize()))
