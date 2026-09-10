import copy
import torch


def build_target(online):
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def update_target(target, online, tau):
    for t, o in zip(target.parameters(), online.parameters()):
        t.mul_(tau).add_(o.detach(), alpha=1.0 - tau)
    for t, o in zip(target.buffers(), online.buffers()):
        t.copy_(o)


def tau_schedule(step, total_steps, tau_start=0.996, tau_end=1.0):
    if total_steps <= 1:
        return tau_end
    p = min(max(step / (total_steps - 1), 0.0), 1.0)
    return tau_start + (tau_end - tau_start) * p
