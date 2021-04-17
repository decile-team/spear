import collections

f_d = 'f_d'
f_d_U = 'f_d_U'
test_w = 'test_w'

train_modes = [f_d, f_d_U]

F_d_U_Data = collections.namedtuple('GMMDataF_d_U', 'x l m L d r')
F_d_Data = collections.namedtuple('GMMDataF_d', 'x labels')