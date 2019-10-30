from mpl_toolkits import mplot3d                                                                          

import numpy as np
import matplotlib.pyplot as plt

import json
import ast

def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()                                                                                        
ax = plt.axes(projection="3d")

with open("/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/latest/policy/ls_subsample", 'r') as f:
       data = f.read()               
ls_subsample = json.loads(data)
ls_subsample = ast.literal_eval(json.dumps(ls_subsample))

x = ls_subsample["x"][::5]
y = ls_subsample["y"][::5]
z = ls_subsample["ls1"][::5]
#X, Y = np.meshgrid(x, y)
#Z = z_function(X, Y)

ax.scatter3D(x, y, z, c=z, cmap="hsv")
#ax.plot_wireframe(x, y, z, color='green')
#ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='winter', edgecolor='none')

ax.set_xlabel('X state value')
ax.set_ylabel('Y state value')
ax.set_zlabel('ls1')
ax.set_title('ls1');

plt.show()

