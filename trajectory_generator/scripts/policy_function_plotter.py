from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm                                                                    

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import json
import ast

def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def my_z_function(x_value, y_value):
    global x
    global y
    global z

    values = zip(x, y, z)
    for val in values:
        if val[0] == x_value and val[1] == y_value:
            return val[2]

epoch = -1
dim = "dim_3"


with open("/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/latest/policy/policy_function_1e.txt", 'r') as f:
    data = f.read()               
policy_function = json.loads(data)
policy_function = ast.literal_eval(json.dumps(policy_function))


data_len = len(policy_function[epoch][dim])
x = policy_function[epoch]["x"][-data_len::1]
y = policy_function[epoch]["y"][-data_len::1]
z = policy_function[epoch][dim][::1]

# X, Y = np.meshgrid(x, y)
# Z = z_function(X, Y)
# Z = my_z_function(X, Y)


# fig_surf = plt.figure(dim)
# ax = Axes3D(fig_surf)
# ax_surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)

# ax_surf.set_xlabel('X state value')
# ax_surf.set_ylabel('Y state value')
# ax_surf.set_zlabel(dim)
# ax_surf.set_title(dim);


fig_3d = plt.figure(dim)                                                                                       
ax_3d = plt.axes(projection="3d")

# ax_3d.scatter3D(x, y, z, c=z, cmap="hsv")
ax_3d.scatter3D(x, y, z, c=z, cmap=cm.jet)
# ax_3d.plot_wireframe(X, Y, Z, color='green')
# ax_3d.plot_surface(x, y, z, rstride=1, cstride=1, cmap='winter', edgecolor='none')

ax_3d.set_xlabel('X state value')
ax_3d.set_ylabel('Y state value')
ax_3d.set_zlabel(dim)
ax_3d.set_title(dim);


for epoch, policy in enumerate(policy_function):
    fig_2d = plt.figure(dim+"_"+str(epoch)+"e")    
    fig_2d.suptitle(dim+" for epoch: "+str(epoch))
    ax_2d = plt.subplot(1, 1, 1)
    x_values = policy_function[epoch]["x"][-data_len::]
    y_values = policy_function[epoch]["y"][-data_len::]
    z_values = policy_function[epoch][dim][::]
    x = []
    z = []
    for index, y_value in enumerate(y_values):
        if y_value == 0:
            x.append(x_values[index])
            z.append(z_values[index])
    sns.scatterplot(x, z, ax=ax_2d)


plt.show()

