.. SPEAR(DECILE) documentation master file, created by
   sphinx-quickstart on Sat Apr 17 15:41:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


SPEAR(DECILE) documentation!
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

----

CAGE
==================

:cite:t:`2020:CAGE`

.. automodule:: Cage.core
	:members:

----

Joint Learning(JL)
==================

:cite:t:`2020:JL`

	From here on, Feature model(fm) imply Feature based classification model

.. automodule:: JL.core
	:members:

----

CAGE, JL - UTILS
==================

CAGE and JL utils
------------------

   From here on, Graphical model(gm) imply CAGE algorithm and Feature model(fm) imply Feature based classification model

.. automodule:: utils
	:members:

JL utils
---------

.. automodule:: JL.utils_jl
	:members:

Feature-based Models
---------------------

.. automodule:: JL.models
	:members:

----

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


Bibilography
=============

.. bibliography::