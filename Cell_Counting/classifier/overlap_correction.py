# Correct oversampling by searching for cells which lie within a radius of each other

from readmarkerxml import readmarkerxml
import numpy as np
import matplotlib.pyplot as plt

marker = readmarkerxml(None)

radius = 12
num_overlap = 0
overlap_marker = np.empty((0,3), int)

for cell_1 in marker:
    for cell_2 in marker:
        if cell_1[2] == cell_2[2]-1:
            if np.sqrt( (cell_1[0]-cell_2[0])**2 + (cell_1[1]-cell_2[1])**2 ) < radius:
                num_overlap += 1
                overlap_marker = np.vstack((overlap_marker, cell_2))

print num_overlap
remain_marker = marker
for cell in overlap_marker:
    remain_marker = np.delete(remain_marker, np.where(np.all(remain_marker==cell, axis=1)), axis=0)

#fig, ax = plt.subplots()
#ax.scatter(marker[:,2], marker[:,1], color='b')
#ax.scatter(overlap_marker[:,2], overlap_marker[:,1], color='r')
#ax.scatter(remain_marker[:,2], remain_marker[:,1], color='g')
#plt.show(block=False)

# numpy.savetxt("foo.csv", a, delimiter=",")
