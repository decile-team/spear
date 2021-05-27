import numpy as np

def load_data_to_numpy(file_name="dataset/SMSSpamCollection"):
    #SPAM = 1
    #HAM = 0
    #ABSTAIN = -1
    X = []
    Y = []
    with open(file_name, 'r', encoding='latin1') as f:
        for line in f:
            yx = line.split("\t",1)
            if yx[0]=="spam":
                y=1
            else:
                y=0
            x = yx[1]
            X.append(x)
            Y.append(y)
    
    # X = np.array(X).reshape((len(X),1))
    # Y = np.array(Y).reshape((len(Y),1))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y



