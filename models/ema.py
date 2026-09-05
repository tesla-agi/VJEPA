import torch
import copy

def build_target(online):
    target=copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad=False
    return target

def update_target(target,online,tau=0.998):
    with torch.no_grad():
        for t,o in zip(target.parameters(),online.parameters()):
            t.mul_(tau).add_(o,alpha=(1-tau))

def tau_schedule(step,total_steps,tau_start=0.998):
    return 1.0-(1.0-tau_start)*(1-step/total_steps)

