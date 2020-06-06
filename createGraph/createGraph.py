
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import sys
import sklearn.manifold as mf


args = sys.argv
argc = len(args)

if argc < 3 or not (args[1].endswith('.csv')):
   print("Invalid args")
   print("please set args\n ex.",args[0] ,"inputfile(.csv) outputfile(.png)")
   sys.exit()


print("create",args[2], "from", args[1])

CSVTEXT = np.loadtxt(args[1], delimiter=',', dtype=str)
#print(CSVTEXT)

LABEL = CSVTEXT[:,0]
#print(LABEL)

DATATEXT = np.delete(CSVTEXT, 0, 1)
#print(DATATEXT)

DATA = DATATEXT.astype(np.float64)
DATA = DATA * (-1) + 3
#print(DATA)

DATANUM = len(DATA)

mds = mf.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
pos = mds.fit_transform(DATA)
#print(pos)
#plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')
fig = plt.figure()

plt.scatter(pos[:, 0], pos[:, 1], marker = 'o')
#for i in range(DATANUM):
#    plt.plot(pos[i,0], pos[i,1], 'b.')

"""
#for label

for label, x, y in zip(LABEL, pos[:, 0], pos[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (70, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )
"""
plt.draw()
fig.savefig(args[2])


"""
#for 3d

mds3d = mf.MDS(n_components=3, dissimilarity="precomputed", random_state=6)
pos3d = mds3d.fit_transform(DATA)
print(pos3d)

fig3d = plt.figure()
ax1 = Axes3D(fig3d)

ax1.scatter3D(pos3d[:,0], pos3d[:,1], pos3d[:,2], label='Dataset')
fig3d.savefig("img3d.png")
"""
