# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

# from .my_checkmate import BestCheckpointSaver, get_best_checkpoint
# from .my_data_types import train_modes
from .checkmate import BestCheckpointSaver, get_best_checkpoint
from .data_types import train_modes
checkpoint_dir = "./checkpoint"
if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

num_checkpoints = 1 # Number of checkpoints to keep around

# Keeps only the most recently saved checkpoint
#
# max_to_keep is deliberatly set to 1 in order to provide for the case when more recent checkpoint
# has a smaller global_step. tf.train.Saver() orders by global_step.

class MRUCheckpoint():
	def __init__(self, path, session, variables):
		'''
		Func Desc:
		Initializes the class variables

		Input:
		self
		path - file path
		session 
		variables

		Output:

		'''
		self.ckpt_path = path
		self.ckpt_file = os.path.join(path, 'checkpoint')
		self.checkpoint_prefix = os.path.join(self.ckpt_path, 'hls-model')
		self.sess = session
		# max_to_keep
		self.saver = tf.train.Saver(variables, max_to_keep=1)
		# self.saver = tf.train.Saver()

	def save(self, global_step=None):
		'''
		Func Desc:
		saves the obtained checkpoint

		Input:
		self
		global step (Default - none)

		Output:

		'''
		path = self.saver.save(self.sess, self.checkpoint_prefix, global_step)
		print('Saved MRU checkpoint to path: ', path)
		
	def restore(self):
		'''
		Func Desc:
		Restores the last checkpoint

		Input:
		self

		Output:

		'''
		last_checkpoint = tf.train.latest_checkpoint(self.ckpt_path, 'checkpoint')
		#if self.saver.last_checkpoints:
		#    last_checkpoint = self.saver.last_checkpoints[0]
		#    print('All saved checkpoints: ', self.saver.last_checkpoints)
		#else:
		if not last_checkpoint:
			last_checkpoint = self.checkpoint_prefix

		print('Restoring checkpoint from path: ', last_checkpoint)
		self.saver.restore(self.sess, last_checkpoint)

	def restore_if_checkpoint_exists(self):
		'''
		Func Desc:
		checks if there exists any checkpoint for the file 

		Input:
		self

		Output:
		Boolean (True or False)
		'''
		if os.path.exists(self.ckpt_file):
			self.restore()
			return True
		return False

def test_mru_checkpoints(num_to_keep):
	'''
	Func Desc:
	Runs different sessions while changing the checkpoint number that is currently being worked with and tests the same
	
	Input:
	num_to_keep(int) - a limit on the size of the global step for checkpoint traversal

	Output:

	'''
	global_step = tf.get_variable(name='mru_global_step_%d' % num_to_keep, initializer=10, dtype=tf.int32)
	inc = tf.assign_add(global_step, 1)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	assert sess.run(global_step) == 10

	sess.run(inc)
	assert sess.run(global_step) == 11
	ckpt_path = '/tmp/checkpoints_%.6f' % np.random.rand()
	ckpt = MRUCheckpoint(ckpt_path, sess, tf.global_variables())
	ckpt.save(global_step)

	sess.run(inc)
	assert sess.run(global_step) == 12

	ckpt.restore_if_checkpoint_exists()
	assert sess.run(global_step) == 11

	assgn_op = tf.assign(global_step, 5)
	sess.run(assgn_op)
	assert sess.run(global_step) == 5
	ckpt.save(global_step)

	sess.run(inc)
	sess.run(inc)
	assert sess.run(global_step) == 7

	ckpt.restore_if_checkpoint_exists()
	assert sess.run(global_step) == 5

def test_checkpoint():
	'''
	Func Desc:
	tests whether the checkpoints stored are as expected

	Input:

	Output:

	'''
	v = tf.get_variable(name='v', initializer=12, dtype=tf.int32)
	v1 = tf.assign_add(v, 1)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	assert sess.run(v) == 12
	sess.run(v1)
	assert sess.run(v) == 13
	ckpt = MRUCheckpoint('/tmp/checkpoints', sess, tf.global_variables())
	ckpt.save()

	sess1 = tf.Session()
	sess1.run(tf.global_variables_initializer())
	assert sess1.run(v) == 12
	ckpt1 = MRUCheckpoint('/tmp/checkpoints', sess1, tf.global_variables())
	ckpt1.restore()
	assert sess1.run(v) == 13
	sess1.run(v1)
	assert sess1.run(v) == 14

	ckpt2 = MRUCheckpoint('/tmp/bad-ckpt-path', sess1, tf.global_variables())
	restored = ckpt2.restore_if_checkpoint_exists()
	assert restored == False

	restored = ckpt1.restore_if_checkpoint_exists()
	assert restored
	assert sess1.run(v) == 13

class BestCheckpoint():
	def __init__(self, path, prefix, session, num_checkpoints, variables, global_step):
		'''
		Func Desc:
		Initializes the class member variables to find the Best checkpoint so far

		Input:
		self
		path
		prefix
		session
		num_checkpoints
		variables
		global_step

		Output:

		'''
		self.ckpt_path = os.path.join(path, prefix)
		#self.ckpt_file = os.path.join(self.ckpt_path, 'checkpoint')
		#self.checkpoint_prefix = os.path.join(self.ckpt_path, prefix)
		self.sess = session
		# max_to_keep is None. Number of checkpoints is handled separately by BestCheckpointSaver
		self.saver = tf.train.Saver(variables, max_to_keep=None, save_relative_paths=True) 
		# self.saver = tf.train.Saver()
		self.best_ckpt_saver = BestCheckpointSaver(
			save_dir=self.ckpt_path,
			num_to_keep=num_checkpoints,
			maximize=True,
			saver=self.saver
			)
		self.global_step = global_step

	def save_if_best(self, metric):
		'''
		Func Desc:
		save if the current checkpoint is the best so far

		Input:
		self
		metric

		Output:

		'''
		saved = self.best_ckpt_saver.handle(metric, self.sess, self.global_step)
		path = tf.train.latest_checkpoint(self.ckpt_path, 'checkpoint')
		if saved:
			print('Saved new best checkpoint to path: ', path)
		else:
			print('No new best checkpoint. Did not save a new best checkpoint. Last checkpointed file: ', path)
		
	def restore_best_checkpoint(self):
		'''
		Func Desc:
		Restore the best checkpoint so far

		Input:
		self

		Output:

		'''
		best_ckpt_file = get_best_checkpoint(self.ckpt_path, select_maximum_value=True)
		print('Restoring best checkpoint from path: ', best_ckpt_file)
		self.saver.restore(self.sess, best_ckpt_file)

	def restore_best_checkpoint_if_exists(self):
		'''
		Func Desc:
		Restore the best checkpoint so far only if it exists

		Input:
		self

		Output:

		'''
		try:
			self.restore_best_checkpoint()
			return True
		except ValueError as e:
			print(str(e))
			return False

def test_best_ckpt():
	'''
	Func Desc:
	test for the best checkpoint so far

	Input:

	Output:

	'''
	global_step = tf.get_variable(name='global_step', initializer=50, dtype=tf.int32)
	inc_global_step = tf.assign_add(global_step, 1)
	sess1 = tf.Session()
	sess2 = tf.Session()
	
	sess1.run(tf.global_variables_initializer())
	sess2.run(tf.global_variables_initializer())

	# We'll save using sess1 and restore in sess2
	best_checkpoint_dir = '/tmp/best_ckpt_%.6f' % np.random.rand()
	best1 = BestCheckpoint(best_checkpoint_dir, 'foo-bar', sess1, 3, tf.trainable_variables(), global_step)
	best2 = BestCheckpoint(best_checkpoint_dir, 'foo-bar', sess2, 3, tf.trainable_variables(), global_step)

	restored = best2.restore_best_checkpoint_if_exists()
	assert not restored

	sess1.run(inc_global_step) ## 51
	best1.save_if_best(0.1)

	assert sess2.run(global_step) == 50
	restored = best2.restore_best_checkpoint_if_exists()
	assert restored
	assert sess2.run(global_step) == 51

	sess1.run(inc_global_step) ## 52
	best1.save_if_best(0.05)

	sess2.run(inc_global_step) # 52
	sess2.run(inc_global_step) # 53
	sess2.run(inc_global_step) # 54
	assert sess2.run(global_step) == 54
	restored = best2.restore_best_checkpoint_if_exists()
	assert restored
	assert sess2.run(global_step) == 51

	sess1.run(inc_global_step) ## 53
	best1.save_if_best(0.2)
	sess1.run(inc_global_step) ## 54
	best1.save_if_best(0.15)

	sess2.run(inc_global_step) # 52
	sess2.run(inc_global_step) # 53
	sess2.run(inc_global_step) # 54
	sess2.run(inc_global_step) # 55
	assert sess2.run(global_step) == 55
	restored = best2.restore_best_checkpoint_if_exists()
	assert restored
	assert sess2.run(global_step) == 53

def test_checkmate():
	'''
	Func Desc:
	test whether the checkmate model is working fine

	Input:

	Output:

	'''
	global_step = tf.get_variable(name='checkmate_global_step', initializer=12, dtype=tf.int32)
	inc_global_step_op = tf.assign_add(global_step, 1)
	sess = tf.Session()
	sess1 = tf.Session()
	
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=None)
	# saver = tf.train.Saver()
	best_checkpoint_dir = '/tmp/best_ckpt_%.6f' % np.random.rand()
	best_ckpt_saver = BestCheckpointSaver(
			save_dir=best_checkpoint_dir,
			num_to_keep=3,
			maximize=True,
			saver=saver
			)

	accuracy = 0.1 # 12
	best_ckpt_saver.handle(accuracy, sess, global_step)

	accuracy = 0.2
	sess.run(inc_global_step_op) # 13
	best_ckpt_saver.handle(accuracy, sess, global_step)

	accuracy = 0.05
	sess.run(inc_global_step_op) # 14
	best_ckpt_saver.handle(accuracy, sess, global_step)

	ckpt_path = get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True)
	print('Best ckpt path: ', ckpt_path)
	saver.restore(sess1, ckpt_path)
	assert sess1.run(global_step) == 13
	
	accuracy = 0.12
	sess.run(inc_global_step_op) # 15
	best_ckpt_saver.handle(accuracy, sess, global_step)

	ckpt_path = get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True)
	print('Best ckpt path: ', ckpt_path)
	saver.restore(sess1, ckpt_path)
	assert sess1.run(global_step) == 13

	accuracy = 0.45
	sess.run(inc_global_step_op) # 16
	best_ckpt_saver.handle(accuracy, sess, global_step)

	ckpt_path = get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True)
	print('Best ckpt path: ', ckpt_path)
	saver.restore(sess1, ckpt_path)
	assert sess1.run(global_step) == 16

	# Now select lowest value
	ckpt_path = get_best_checkpoint(best_checkpoint_dir, select_maximum_value=False)
	print('Best ckpt path: ', ckpt_path)
	saver.restore(sess1, ckpt_path)
	assert sess1.run(global_step) == 15

# Loading of checkpoints happens only once - at the end of HLSModel initialization.
#
# Saving of checkpoints happens during training. We have only one MRU checkpoint saver 
# We have one best checkpoint saver per train mode type
class CheckpointsFactory:
	def __init__(self, sess, global_steps):
		'''
		Func Desc:
		Initializes the class with the arguments

		Input:
		self
		sess 
		global_steps

		Output:

		'''
		self.best_savers = {}
		self.initialize_savers(sess, global_steps)

	def get_best_saver(self, train_mode):
		'''
		Func Desc:
		get the best saved checkpoints

		Input:
		self
		Train_mode - the mode of training

		Output:

		'''
		return self.best_savers[train_mode]

	def initialize_savers(self, sess, global_steps):
		'''
		Func Desc:
		Initialize the required savers

		Input:
		self
		sess
		global_steps

		Output:

		'''
		for mode in train_modes:
			self.init_saver(sess, mode, global_steps)

	def init_saver(self, sess, mode, global_steps):
		'''
		Func Desc:
		Initialize the required savers with the given mode

		Input:
		self
		sess
		mode
		global_steps

		Output:

		'''
		ckpt_dir = checkpoint_dir
		self.best_savers[mode] = BestCheckpoint(ckpt_dir, mode, sess,
				num_checkpoints, tf.global_variables(), global_steps[mode])

