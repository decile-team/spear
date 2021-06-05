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
.. automodule:: spear.labeling.lf.core
	:members: LabelingFunction, labeling_function

Continuous scoring
-------------------
.. automodule:: spear.labeling.continuous_scoring.core
	:members: BaseContinuousScorer,continuous_scorer

LFApply
-------------
.. automodule:: spear.labeling.apply.core
	:members: 

LFSet
------
.. automodule:: spear.labeling.lf_set.core
	:members: LFSet

LFAnalysis
-------------
.. automodule:: spear.labeling.analysis.core
	:members: LFAnalysis

Noisy Labels
---------------
.. automodule:: spear.labeling.prelabels.core
	:members: NoisyLabels

----

CAGE
==================

:cite:t:`2020:CAGE`

.. automodule:: spear.Cage.core
	:members:

----

Joint Learning(JL)
==================

:cite:t:`2020:JL`

	From here on, Feature model(fm) imply Feature based classification model

.. automodule:: spear.JL.core
	:members:

----

CAGE, JL - UTILS
==================

Data loaders
---------------

.. automodule:: spear.utils.data_editer
	:members:

CAGE and JL utils
------------------

   From here on, Graphical model(gm) imply CAGE algorithm and Feature model(fm) imply Feature based classification model

.. automodule:: spear.utils.utils_cage
	:members:

JL utils
---------

.. automodule:: spear.utils.utils_jl
	:members:

Feature-based Models
---------------------

.. automodule:: spear.JL.models.models
	:members:

----

IMPLYLOSS
============

Implyloss Checkmate
---------------------------
.. automodule:: spear.Implyloss.my_checkmate
	:members:

Implyloss Checkpoints
---------------------------
.. automodule:: spear.Implyloss.my_checkpoints
	:members:

Implyloss Config
---------------------------
.. automodule:: spear.Implyloss.my_config
	:members:

Implyloss Data Feeders
---------------------------
.. automodule:: spear.Implyloss.my_data_feeders
	:members:

Implyloss Data Feeders Utils
----------------------------------
.. automodule:: spear.Implyloss.my_data_feeder_utils
	:members:

Implyloss Gen Cross Entropy Utils
------------------------------------
.. automodule:: spear.Implyloss.my_gen_cross_entropy_utils
	:members:


Implyloss Model
---------------------------
.. automodule:: spear.Implyloss.my_model
	:members:

Implyloss PR Utils
---------------------------
.. automodule:: spear.Implyloss.my_pr_utils
	:members:

Implyloss Test
---------------------------
.. automodule:: spear.Implyloss.my_test
	:members:

Implyloss Train
---------------------------
.. automodule:: spear.Implyloss.my_train
	:members:


Implyloss Utils
---------------------------
.. automodule:: spear.Implyloss.my_utils
	:members:


Bibilography
=============

.. bibliography::