![PyPI](https://img.shields.io/pypi/v/spear)
[![docs](https://readthedocs.org/projects/spear-decile/badge)](https://spear-decile.readthedocs.io/en/master)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![website](https://img.shields.io/badge/website-online-green)](https://decile.org/)

# SPEAR by DECILE

## Semi-Supervised Data Programming for Data Efficient Machine Learning
SPEAR is a library for data programming with semi-supervision. The package implements several recent data programming approaches including facility to programmatically label and build training data.

### SPEAR provides functionality such as 
* creation of LFs/rules/heuristics
* compare against several data programming approaches
* compare against semi-supervised data programming approaches
* use subset selection to reduce annotation effort

## Installation

SPEAR requires Python 3.6 or later. To install SPEAR, we recommend using `pip`:
```git clone https://github.com/decile-team/spear/```
```pip install -r requirements.txt```

## Labelling Functions (LFs)
* discrete LFs - Users can define LFs that return discrete labels
* continuous LFs - return continuous scores to assign labels

## Approaches Implemented
You can read [this paper](https://arxiv.org/pdf/2008.09887.pdf) to know about below approaches
* Only-L 
* L + Umaj
* Learning to Reweight
* L + USnorkel
* Posterior Regularization
* Imply Loss
* CAGE
* SPEAR

## Quick Links
* [SPEAR tutorials](https://github.com/decile-team/spear/tree/main/notebooks)
* [DECILE website](https://decile.org)
* [SPEAR documentation](https://spear-decile.readthedocs.io/)
* [SubModLib - Summarize massive datasets using submodular optimization](https://github.com/decile-team/submodlib)
* [DISTIL- Deep Diversified Interactive Learning](https://github.com/decile-team/distil)
* [CORDS- COResets and Data Subset Selection](https://github.com/decile-team/cords)

## Acknowledgment
SPEAR takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Snorkel](https://github.com/snorkel-team/snorkel) & [Imply Loss](https://github.com/awasthiabhijeet/Learning-From-Rules). Also, SPEAR uses [Apricot](https://github.com/jmschrei/apricot) for subset selection.

## Team
SPEAR is created and maintained by [Ayush](https://cse.iitb.ac.in/~ayusham), [Abhishek]( https://www.cse.iitb.ac.in/~gsaiabhishek/), [Vineeth](https://www.cse.iitb.ac.in/~vineethdorna/), [Harshad](https://www.cse.iitb.ac.in/~harshadingole/), [Parth](https://www.cse.iitb.ac.in/~parthlaturia/), [Pankaj](https://www.linkedin.com/in/pankaj-singh-b000894a/), [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have SPEAR more community driven. Please use it and contribute to it for your active learning research, and feel free to use it for your commercial projects. We will add the major contributors here.

## Publications

[1] Maheshwari, Ayush, et al. "Data Programming using Semi-Supervision and Subset Selection." arXiv preprint arXiv:2008.09887 (2020).

[2] Chatterjee, Oishik, Ganesh Ramakrishnan, and Sunita Sarawagi. "Data Programming using Continuous and Quality-Guided Labeling Functions." arXiv preprint arXiv:1911.09860 (2019).

[3] Sahay, Atul, et al. "Rule augmented unsupervised constituency parsing." arXiv preprint arXiv:2105.10193 (2021).
