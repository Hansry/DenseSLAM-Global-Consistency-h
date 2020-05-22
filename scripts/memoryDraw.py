import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy.io
import numpy as np


#################################################
def floatrange(start,stop,steps):
    return [start+float(i)*(stop-start)/(float(steps)-1) for i in range(steps)]


txt_set = ["memory_decay_slide_window","memory_slide_window", "memory_decay","memory"]
label_set = ["Slidewindow + Map regularization", "Slidewindow", "Map regularization", "Origin"]
color_set = ['r', 'y', 'b', 'g']
data_list = []
for i in txt_set:
    fileName = "../"+i+".txt"
    print(fileName)
    data = []
    with open(fileName,"r") as f:
        data_x = []
        data_y = []
        for line in f.readlines():
            currentLine = line.strip('\n').split(" ")
            data_x.append(float(currentLine[-2]))
            data_y.append(float(currentLine[-1]))
        data.append(data_x)
        data.append(data_y)
    data_list.append(data)

for j in range(len(data_list)): #4组
    plt.plot(data_list[j][0], data_list[j][1], color=color_set[j], label=label_set[j])

plt.legend(loc="lower right")
fig1 = plt.figure(1)
axes = plt.subplot(111)
axes = plt.gca()
#axes.set_yticks([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])
#axes.grid(True)  # add grid

plt.ylabel('Memory Usage / 100MB')   # set ystick label
plt.xlabel('Frame Number')  # set xstck label
plt.grid(linestyle=':',alpha=1,linewidth=0.8)
plt.savefig('/home/hansry/Abs_Rel_1.png',dpi = 400,bbox_inches='tight')
plt.show()


