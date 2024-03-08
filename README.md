
<!-- [![](https://tokei.rs/b1/github/decile-team/spear?category=code)](https://github.com/decile-team/spear) -->
[![Lines of code](https://img.shields.io/tokei/lines/github/decile-team/spear)]()
![visitors](https://visitor-badge.glitch.me/badge?page_id=decile-team/spear)
![PyPI](https://img.shields.io/pypi/v/spear)
[![docs](https://readthedocs.org/projects/spear-decile/badge)](https://spear-decile.readthedocs.io/)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/decile-team/spear/blob/main/LICENSE.txt)
[![website](https://img.shields.io/badge/website-online-green)](https://decile.org/)

<!-- ![GitHub repo size](https://img.shields.io/github/repo-size/decile-team/spear) -->

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
* facility to store and save data in [pickle file](https://spear-decile.readthedocs.io/en/latest/index.html#spear.utils.data_editor.get_data)

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

Direct download of the zip file can be done via wget using `gdown` library . 
```bash
pip install gdown
gdown 1CJZ73nNa7Ho0BOSDgGx9CRvXoepVSpet
```

## Installation
* Install Submodlib library
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib`
In case of installation issues with the submodlib, please consult [Submodlib Github](https://github.com/decile-team/submodlib).
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
@inproceedings{abhishek-etal-2022-spear,
    title = "{SPEAR} : Semi-supervised Data Programming in Python",
    author = "Abhishek, Guttu  and
      Ingole, Harshad  and
      Laturia, Parth  and
      Dorna, Vineeth  and
      Maheshwari, Ayush  and
      Ramakrishnan, Ganesh  and
      Iyer, Rishabh",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-demos.12",
    pages = "121--127",
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


## Related Publications

1. Divya Jyoti Bajpai, Ayush Maheshwari, Manjesh Kumar Hanawal, Ganesh Ramakrishnan (2024). [FAIR: Filtering of Automatically Induced Rules](https://arxiv.org/abs/2402.15472). In EACL, 2024.

1. Akshat Gautam, Anurag Shandilya, Akshit Srivastava, Venkatapathy Subramanian, Ganesh Ramakrishnan, Kshitij Jadhav (2024). [INSITE: labelling medical images using submodular functions and semi-supervised data programming](https://arxiv.org/abs/2402.07173). In ISBI, 2024.

1. Dhruv Kudale, Badri Vishal Kasuba, Venkatapathy Subramanian, Parag Chaudhuri, Ganesh Ramakrishnan. [TEXTRON: Weakly Supervised Multilingual Text Detection through Data Programming](https://arxiv.org/abs/2402.09811). In WACV, 2024.
 

1. Ayush Maheshwari, Piyush Sharma, Preethi Jyothi, Ganesh Ramakrishnan (2023). [DICTDIS: Dictionary Constrained Disambiguation for Improved NMT](https://arxiv.org/abs/2210.06996)


1.  Abhishek Singh, Venkatapathy Subramanian, Ayush Maheshwari, Pradeep Narayan, Devi Prasad Shetty, Ganesh Ramakrishnan (2023). [EIGEN: Expert-Informed Joint Learning Aggregation for High-Fidelity Information Extraction from Document Images](https://proceedings.mlr.press/v225/singh23a.html). In Proceedings of ML4Health Conference, 2023 (co-located with Neurips).

1. Ayush Maheshwari, Ajay Ravindran, Venkatapathy Subramanian, Ganesh Ramakrishnan (2023). [UDAAN - Machine Learning based Post-Editing tool for Document Translation](https://arxiv.org/abs/2203.01644). Best Paper Award In CODS-COMAD 2023.

1. Durga S, Ayush Maheshwari, Pradeep Shenoy, Prathosh AP, Ganesh Ramakrishnan (2022). [Reweighing auxiliary losses in supervised learning](https://arxiv.org/abs/2202.03250). In AAAI 2023.


1. Maheshwari et al. [Learning to Robustly Aggregate Labeling Functions for Semi-supervised Data Programming](https://arxiv.org/abs/2109.11410), In Findings of ACL (Long Paper) 2022.

1. Maheshwari, Ayush, et al. [Data Programming using Semi-Supervision and Subset Selection](https://arxiv.org/abs/2008.09887), In Findings of ACL (Long Paper) 2021.

1. Sahay, Atul, et al. [Rule augmented unsupervised constituency parsing](https://arxiv.org/abs/2105.10193), In Findings of ACL (Short Paper) 2021.

1. Chatterjee, Oishik, Ganesh Ramakrishnan, and Sunita Sarawagi. [Data Programming using Continuous and Quality-Guided Labeling Functions](https://arxiv.org/abs/1911.09860), In AAAI 2020.


