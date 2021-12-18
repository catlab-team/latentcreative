import numpy as np

def sample_artbreeder_class(num_samples):
    #assert False # add trunctation
    with open("others/class_info.txt", "r") as file: 
        class_data = np.loadtxt(file)
    ys = np.zeros((num_samples, 1000))
    
    for i in range(1000):
        ys[:,i] = np.random.normal(class_data[i,0], class_data[i,1], num_samples)
    return ys


if __name__ == "__main__":
    a = sample_artbreeder_class(10)
    print(a)
