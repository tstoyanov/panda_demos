import torch
import numpy as np
import torch.nn.utils as utils

# Implementation of the reinforce algorithm for a continuous action space
class ALGORITHM:
    def __init__(self, policy_model, optimizer, gamma):
        self.policy_model = policy_model
        self.optimizer = optimizer
        self.gamma = gamma

    def update_policy(self, log_probs, rewards, entropies):
        R = 0
        loss = 0

        for i in reversed(range(len(rewards))):
            # R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*rewards[i]).sum()
        loss = loss / len(rewards)

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(self.policy_model.parameters(), 40)
        self.optimizer.step()
        return loss