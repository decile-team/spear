.. SPEAR(DECILE) documentation master file, created by
   sphinx-quickstart on Sat Apr 17 15:41:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to SPEAR's documentation!
==================================
SPEAR: **S**\ emi-Su\ **pe**\ rvised D\ **a**\ ta Prog\ **r**\ amming

.. toctree::
   :maxdepth: 5
   :caption: Contents:

We present SPEAR, an open-source python library for data programming with semi-supervision.
 The package implements several recent data programming approaches including facility to programmatically 
 label and build training data. SPEAR facilitates weak supervision, either pre-defined, in the form of 
 rules/heuristics and associate 'noisy' labels(or prelabels) to the training dataset. These noisy labels 
 are aggregated to assign labels to the unlabeled data for downstream tasks. Several label aggregation 
 approaches have been proposed that aggregate the noisy labels and then train the 'noisily' labeled set in 
 a cascaded manner, while other approaches 'jointly' aggregates and trains the model. In the python package,
  we integrate several cascade and joint data-programming approaches while providing facility to define rules.
   The code and tutorial notebooks are available `here <https://github.com/decile-team/spear>`_.

Labeling
==========

This module takes inspiration from :cite:t:`snorkel:2020`

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

Pre Labels
---------------
.. automodule:: spear.labeling.prelabels.core
	:members: PreLabels

----

CAGE
==================

:cite:t:`2020:CAGE`

.. automodule:: spear.cage.core
	:members:

----

Joint Learning(JL)
==================

:cite:t:`DBLP:journals/corr/abs-2008-09887`

	From here on, Feature model(fm) imply Feature based classification model

.. automodule:: spear.jl.core
	:members:

----

Subset Selection
==================

Uses facilityLocation from :cite:t:`JMLR:v21:19-467`


.. automodule:: spear.jl.subset_selection
	:members:

----

CAGE, JL - UTILS
==================

Note: The arguments whose shapes are mentioned in '[....]' are torch tensors.

Data loaders
---------------

.. automodule:: spear.utils.data_editor
	:members:

----

CAGE and JL utils
------------------

   From here on, Graphical model(gm) imply CAGE algorithm and Feature model(fm) imply Feature based classification model

.. automodule:: spear.utils.utils_cage
	:members:

----

JL utils
---------

.. automodule:: spear.utils.utils_jl
	:members:

----

Feature-based Models
---------------------

.. automodule:: spear.jl.models.models
	:members:

----

ImplyLoss
============

Implyloss Checkmate
---------------------------
.. automodule:: spear.Implyloss.checkmate
	:members:

Implyloss Checkpoints
---------------------------
.. automodule:: spear.Implyloss.checkpoints
	:members:

.. Implyloss Config
.. ---------------------------
.. .. automodule:: spear.Implyloss.config
.. 	:members:

Implyloss Data Feeders
---------------------------
.. automodule:: spear.Implyloss.data_feeders
	:members:

Implyloss Data Feeders Utils
----------------------------------
.. automodule:: spear.Implyloss.data_feeder_utils
	:members:

Implyloss Gen Cross Entropy Utils
------------------------------------
.. automodule:: spear.Implyloss.gen_cross_entropy_utils
	:members:


Implyloss Model
---------------------------
.. automodule:: spear.Implyloss.model
	:members:

Implyloss PR Utils
---------------------------
.. automodule:: spear.Implyloss.pr_utils
	:members:

Implyloss Test
---------------------------
.. automodule:: spear.Implyloss.test
	:members:

Implyloss Train
---------------------------
.. automodule:: spear.Implyloss.train
	:members:


Implyloss Utils
---------------------------
.. automodule:: spear.Implyloss.utils
	:members:


Bibilography
=============

.. bibliography::
