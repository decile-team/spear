import numpy as np

def load_data_to_numpy(file_name="../../data/TREC/train.txt"):
    label_map = {"DESC": "DESCRIPTION",
                "ENTY": "ENTITY",
                "HUM": "HUMAN",
                "ABBR": "ABBREVIATION",
                "LOC": "LOCATION",
                "NUM": "NUMERIC"}
    
    X, Y =  [], []
    mode = file_name.split('/')[-1]
    with open(file_name, 'r', encoding='latin1') as f:
        for line in f:
            label = label_map[line.split()[0].split(":")[0]]
            if mode == "test.txt":
                sentence = (" ".join(line.split()[1:]))
                X.append(sentence)
                Y.append(label)
            else:
                sentence = (" ".join(line.split(":")[1:])).lower().strip()
                X.append(sentence)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# def load_data(mode):
#     label_map = {"DESC": "DESCRIPTION", "ENTY": "ENTITY", "HUM": "HUMAN", "ABBR": "ABBREVIATION", "LOC": "LOCATION",
#            "NUM": "NUMERIC"}
#     data = []

#     with open(mode + '.txt', 'r', encoding='latin1') as f:
#         for line in f:
#             label = LABEL_DICT[label_map[line.split()[0].split(":")[0]]]
#             if mode == "test":
#                 sentence = (" ".join(line.split()[1:]))
#             else:
#                 sentence = (" ".join(line.split(":")[1:])).lower().strip()
#             data.append((sentence, label))
#     return data
