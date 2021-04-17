.. Edurevol documentation master file, created by
   sphinx-quickstart on Mon Mar 22 20:44:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPEAR(DECILE) documentation!
========================================

.. toctree::
   :maxdepth: 5
   :caption: Contents:


Labeling
================================

LF
------
.. automodule:: labeling.lf.core
	:members: LabelingFunction, labeling_function

Continuous scoring
-------------------
.. automodule:: labeling.continuous_scoring.core
	:members: BaseContinuousScorer,continuous_scorer

LFApply
-------------
.. automodule:: labeling.apply.core
	:members: 

LFSet
------
.. automodule:: labeling.lf_set.core
	:members: LFSet

LFAnalysis
-------------
.. automodule:: labeling.analysis.core
	:members: LFAnalysis

Noisy Labels
---------------
.. automodule:: labeling.noisy_labels.core
	:members: NoisyLabels
             

CAGE
==================

.. automodule:: Cage_JL.core_cage
	:members:

Joint Learning(JL)
==================

.. automodule:: Cage_JL.core_jl
	:members:


CAGE, JL utils
===============

Feature-based Models
---------------------

.. automodule:: Cage_JL.models
	:members:

CAGE utils
-----------

   From here on, Graphical model imply CAGE algorithm

   JL uses these utils too

.. automodule:: Cage_JL.utils
	:members:

JL utils
---------

.. automodule:: Cage_JL.utils_jl
	:members:

IMPLYLOSS
============

Implyloss Checkmate
---------------------------
.. automodule:: Implyloss.my_checkmate
	:members:

Implyloss Checkpoints
---------------------------
.. automodule:: Implyloss.my_checkpoints
	:members:

Implyloss Config
---------------------------
.. automodule:: Implyloss.my_config
	:members:

Implyloss Data Feeders
---------------------------
.. automodule:: Implyloss.my_data_feeders
	:members:

Implyloss Data Feeders Utils
----------------------------------
.. automodule:: Implyloss.my_data_feeder_utils
	:members:

Implyloss Gen Cross Entropy Utils
------------------------------------
.. automodule:: Implyloss.my_gen_cross_entropy_utils
	:members:


Implyloss Model
---------------------------
.. automodule:: Implyloss.my_model
	:members:

Implyloss PR Utils
---------------------------
.. automodule:: Implyloss.my_pr_utils
	:members:

Implyloss Test
---------------------------
.. automodule:: Implyloss.my_test
	:members:

Implyloss Train
---------------------------
.. automodule:: Implyloss.my_train
	:members:


Implyloss Utils
---------------------------
.. automodule:: Implyloss.my_utils
	:members: