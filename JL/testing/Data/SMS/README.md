# Content Description

* For description of {d,U,validation,test}processed.p see the main README of this repository
* spam.csv is the original datafile from [SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
* rule.txt contains rules defined by observing a few sentences randomly fectched from spam.csv. 
* Format of rule.txt
	- First column represents the label provided by rule
	- Second column represents the regex corresponding to the rule
	- Third column represents the sentence inspecting which rule was designed.
* Remaing sentences in spam.csv were randomly distributed into train.csv, test.csv, valid.csv
* 500 sentences in train and test, while remaining in train.
* generate_data.py creates {d,U,validation,test}processed.p using rules.txt,train.csv,valid.csv,test.csv respectively
	- obtain_embeddings.py is used, which uses [elmo](https://tfhub.dev/google/elmo/) to convert sentences into embeddings

# Note:
* Since sentences in spam.csv were randomly distributed into train test and valid files, the pickle files you may generate might be little different than what is dumped here.
* If you are re-generating the pickles using generate_data.py, then you should re-train the snorkel's saved_label_model using run_snorkel.py (See the main README)
* Pickle files dumped here were used in our experiments.

