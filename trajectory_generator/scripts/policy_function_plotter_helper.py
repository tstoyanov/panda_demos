algorithm.load_checkpoint("/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/10_perfect/checkpoint/2019-10-31_15:39:28_r3.576.tar")
for state_x in x_dist:
  for state_y in y_dist:
    policy_function["x"].append(state_x)
    policy_function["y"].append(state_y)
    policy_ret = algorithm.policy(torch.tensor([state_x, state_y]))
    for dim_index, _ in enumerate(policy_ret[0]):
      policy_function["dim_"+str(dim_index+1)].append(round(policy_ret[0][dim_index], 4))
policy_function_list.append(copy.deepcopy(policy_function))

checkpoint_folder = "/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/10_perfect_add_30/checkpoint/"
policy_folder = "/home/aass/workspace_shuffle/src/panda_demos/trajectory_generator/saved_models/policy_network/a200_b1e-1_80000e_smoother_batch/10_perfect_add_30/policy/"
checkpoint_files = [f for f in listdir(checkpoint_folder) if isfile(join(checkpoint_folder, f))]
checkpoint_files.sort()
for checkpoint_file in checkpoint_files:
  del policy_function["x"][:]
  del policy_function["y"][:]
  for dim_index, _ in enumerate(latent_space_data["mean"]):
    del policy_function["dim_"+str(dim_index+1)][:]

  algorithm.load_checkpoint(input_folder+checkpoint_file)

  for state_x in x_dist:
    for state_y in y_dist:
      policy_function["x"].append(state_x)
      policy_function["y"].append(state_y)
      policy_ret = algorithm.policy(torch.tensor([state_x, state_y]))
      for dim_index, _ in enumerate(policy_ret[0]):
        policy_function["dim_"+str(dim_index+1)].append(round(policy_ret[0][dim_index], 4))
  policy_function_list.append(copy.deepcopy(policy_function))

with open(policy_save_dir + "policy_function.txt", "w") as f:
  json.dump(policy_function_list, f)
