from labeling.preprocess import *

@preprocessor()
def square(x):
    return x*x

print("Preprocessor function declaration is done")

print(square)

if square(5) == 5*5:
    print("="*10+"Basic preprocessor testing is successfull"+"="*10)
else:
    print("="*10+"Something went wrong"+"="*10)
