import numpy as np
import matplotlib.pyplot as plt

def print_mislabeled_images(classes, X, y, p):
    # Plots images where predictions and truth were different.

    plt.figure(2)
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (80.0, 80.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        Frow = round(num_images/2)
        Nrow = num_images-Frow
        if num_images <= Frow:
            plt.subplot(2, Frow, i + 1)
        else:
            plt.subplot(2, Nrow, i + 1)   
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Pred: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))