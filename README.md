
[![](https://tokei.rs/b1/github/decile-team/spear?category=code)](https://github.com/decile-team/spear)
![PyPI](https://img.shields.io/pypi/v/spear)
[![docs](https://readthedocs.org/projects/spear-decile/badge)](https://spear-decile.readthedocs.io/)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/decile-team/spear/blob/main/LICENSE.txt)
[![website](https://img.shields.io/badge/website-online-green)](https://decile.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/decile-team/spear)

<p align="center">
    <br>
        <img src="https://github.com/decile-team/spear/blob/main/spear_logo.png" width="540" height="150"/>
    </br>
</p>

## Semi-Supervised Data Programming for Data Efficient Machine Learning
SPEAR is a library for data programming with semi-supervision. The package implements several recent data programming approaches including facility to programmatically label and build training data.

### Pipeline
* Design Labeling functions(LFs)
* generate pickle file containing labels by passing raw data to LFs
* Use one of the Label Aggregators(LA) to get final labels

<p align="center">
    <br>
        <img src="https://github.com/decile-team/spear/blob/main/spear_pipeline.svg" width="1000" height="300" />
    </br>
</p>


### SPEAR provides functionality such as 
* development of LFs/rules/heuristics for quick labeling
* compare against several data programming approaches
* compare against semi-supervised data programming approaches
* use subset selection to make best use of the annotation efforts

#### Labelling Functions (LFs)
* discrete LFs - Users can define LFs that return discrete labels
* continuous LFs - return continuous scores/confidence to the labels assigned

#### Approaches Implemented
You can read [this paper](https://arxiv.org/pdf/2008.09887.pdf) to know about below approaches
* Only-L 
* Learning to Reweight
* Posterior Regularization
* Imply Loss
* CAGE
* Joint Learning

Data folder for SMS & TREC can be found [here](https://drive.google.com/file/d/1CJZ73nNa7Ho0BOSDgGx9CRvXoepVSpet/view?usp=sharing). This folder needs to be placed in the same directory as notebooks folder is in, to run the notebooks or examples.

## Installation
* Install Submodlib library
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib`
#### Method 1
To install latest version of SPEAR package using PyPI:
 ```python 
 pip install decile-spear
 ```
#### Method 2    
SPEAR requires Python 3.6 or later. First install [submodlib](https://github.com/decile-team/submodlib#setup). Then install SPEAR:

```bash
git clone https://github.com/decile-team/spear.git
cd spear
pip install -r requirements/requirements.txt
```

## Citation
```bibtex
@misc{abhishek2021spear,
      title={SPEAR : Semi-supervised Data Programming in Python}, 
      author={Guttu Sai Abhishek and Harshad Ingole and Parth Laturia and Vineeth Dorna and Ayush Maheshwari and Ganesh Ramakrishnan and Rishabh Iyer},
      year={2021},
      eprint={2108.00373},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Quick Links
* [SPEAR tutorials](https://github.com/decile-team/spear/tree/main/notebooks)
* [SPEAR documentation](https://spear-decile.readthedocs.io/)
* [Demonstration of Cage and Joint Learning using SPEAR](https://youtu.be/qdukvO3B8YU)
* [Demonstration of Imply Loss, Learn2Reweight using SPEAR](https://youtu.be/SN9YYK4FlU0)
* SMS SPAM: [CAGE colab](https://colab.research.google.com/drive/1vec-Q-xO9wQtM3p_CZ7237gCq0xIR9b9?usp=sharing), [JL colab](https://colab.research.google.com/drive/1HqkqQ8ytWjP9on3du-vVB07IQvo8Li3W?usp=sharing)
* [DECILE website](https://decile.org)
* [SubModLib - Summarize massive datasets using submodular optimization](https://github.com/decile-team/submodlib)
* [DISTIL- Deep Diversified Interactive Learning](https://github.com/decile-team/distil)
* [CORDS- COResets and Data Subset Selection](https://github.com/decile-team/cords)



### Acknowledgment
SPEAR takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Snorkel](https://github.com/snorkel-team/snorkel), [Snuba](https://github.com/HazyResearch/reef)  & [Imply Loss](https://github.com/awasthiabhijeet/Learning-From-Rules). Also, SPEAR uses [SUBMODLIB](https://github.com/decile-team/submodlib) for subset selection, which is provided by [DECILE](https://decile.org/) too.

### Team
SPEAR is created and maintained by [Ayush](https://www.cse.iitb.ac.in/~ayusham), [Abhishek](https://www.linkedin.com/in/guttu-sai-abhishek/), [Vineeth](https://www.cse.iitb.ac.in/~vineethdorna/), [Harshad](https://www.cse.iitb.ac.in/~harshadingole/), [Parth](https://www.cse.iitb.ac.in/~parthlaturia/), [Pankaj](https://www.linkedin.com/in/pankaj-singh-b000894a/), [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have SPEAR more community driven. Please use it and contribute to it for your research, and feel free to use it for your commercial projects. We will add the major contributors here.


## Publications

[1] Maheshwari, Ayush, et al. [Data Programming using Semi-Supervision and Subset Selection](https://arxiv.org/abs/2008.09887), In Findings of ACL (Long Paper) 2021.

[2] Chatterjee, Oishik, Ganesh Ramakrishnan, and Sunita Sarawagi. [Data Programming using Continuous and Quality-Guided Labeling Functions](https://arxiv.org/abs/1911.09860), In AAAI 2020.

[3] Sahay, Atul, et al. [Rule augmented unsupervised constituency parsing](https://arxiv.org/abs/2105.10193), In Findings of ACL (Short Paper) 2021.
