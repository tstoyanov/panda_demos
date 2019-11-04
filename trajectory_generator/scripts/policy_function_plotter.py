from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.colors as mcol                                                            

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
dim = "dim_1"
y_target = None
x_target = 0.8


with open("/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/curved_centered/policy/policy_function.txt", 'r') as f:
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


steps = len(policy_function)
# initial_color = (0, 0, 1)
# final_color = (1, 0, 0)
# new_color_delta = tuple((item_final - item_initial) / steps for item_initial, item_final in zip(initial_color, final_color))
# new_color = initial_color


# # Make a user-defined colormap.
# cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])

# # Make a normalizer that will map the time values from
# # [start_time,end_time+1] -> [0,1].
# cnorm = mcol.Normalize(vmin=0,vmax=steps)

# # Turn these into an object that can be used to map time values to colors and
# # can be passed to plt.colorbar().
# # cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
# cpick = cm.ScalarMappable(norm=cnorm,cmap=plt.get_cmap("jet"))
# cpick.set_array([])

# fig_2d_stack = plt.figure(dim+"_"+str(epoch)+"e_stack")    
# fig_2d_stack.suptitle(dim+" - y={}".format(y_target))
# ax_2d_stack = plt.subplot(1, 1, 1)

for epoch, policy in enumerate(policy_function):
    fig_2d = plt.figure(dim+"_"+str(epoch)+"e")    
    ax_2d = plt.subplot(1, 1, 1)
    x_values = policy_function[epoch]["x"][-data_len::]
    y_values = policy_function[epoch]["y"][-data_len::]
    z_values = policy_function[epoch][dim][::]
    x = []
    z = []
    if x_target is None:
        fig_2d.suptitle(dim+" - y={}\nepoch: {}".format(y_target, epoch))
        for index, y_value in enumerate(y_values):
            if y_value == y_target:
                x.append(x_values[index])
                z.append(z_values[index])
        g = sns.scatterplot(x, z, ax=ax_2d, color="red")
        plt.xlabel("x_state")
    elif y_target is None:
        fig_2d.suptitle(dim+" - x={}\nepoch: {}".format(x_target, epoch))
        for index, x_value in enumerate(x_values):
            if x_value == x_target:
                x.append(y_values[index])
                z.append(z_values[index])
        g = sns.scatterplot(x, z, ax=ax_2d, color="red")
        plt.xlabel("y_state")
    # new_color = cpick.to_rgba(epoch)
    # g_stack = sns.lineplot(x, z, ax=ax_2d_stack, color=new_color, lw=3.0)
    # # g_stack = sns.scatterplot(x, z, ax=ax_2d_stack, color=new_color, kwargs="linewidth=2.0")
    # new_color = tuple(current_item + delta_item for current_item, delta_item in zip(new_color, new_color_delta))
    plt.ylabel(dim)

    # fig_2d_stack.subplots_adjust(right=0.8)
    # cbar_ax = fig_2d_stack.add_axes([0.85, 0.11, 0.02, 0.775])
    # fig_2d_stack.colorbar(cpick, cax=cbar_ax, label="initial policy                                                                                                                                                final policy", ticks=[])


plt.show()

