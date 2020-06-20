import numpy as np
# import matplotlib.pyplot as plt
# from   mpl_toolkits.mplot3d import Axes3D
import csv
# import sys
import sklearn.decomposition as dc

def createGraph(file_path):
    CSVTEXT = np.loadtxt(file_path, delimiter=',', dtype=str)
    LABEL = CSVTEXT[:,0]
    DATATEXT = np.delete(CSVTEXT, 0, 1)

    DATA = DATATEXT.astype(np.float64)
    DATA = DATA * (-1) + 3

    DATANUM = len(DATA)


    pca = dc.PCA(n_components=2)
    pos = pca.fit_transform(DATA)

    # fig = plt.figure()

    # plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')

    # plt.draw()
    # fig.savefig(args[2])
    #print(LABEL[0])
    return [{'x': p[0], 'y': p[1], 'label': l} for p, l in zip(pos, LABEL)], LABEL
