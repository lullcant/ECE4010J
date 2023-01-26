import argparse

import torch
from torch.autograd import Variable
from torch import optim
#from experiment import Experiment

from visual import plot_density, scatter_points
from finite_flows import Normalizing_Flow
from free_energy import FreeEnergyBound

def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)

def p_z(z):

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)

    return torch.exp(-u)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--log_interval", type=int, default=300,
    help="How frequenlty to print the training stats."
)
parser.add_argument(
    "--plot_interval", type=int, default=300,
    help="How frequenlty to plot samples from current distribution."
)
parser.add_argument(
    "--plot_points", type=int, default=1000,
    help="How many to points to generate for one plot."
)

args = parser.parse_args()

torch.manual_seed(42)


# with Experiment({
#     "batch_size": 40,
#     "iterations": 10000,
#     "initial_lr": 0.01,
#     "lr_decay": 0.999,
#     "flow_length": 16,
#     "name": "planar"
# }) as experiment:
batch_size= 40
iterations = 10000
lr = 0.01
lr_decay = 0.999
flow_length = 16
name = 'planar'


flow = Normalizing_Flow(dimension=2, flow_length=flow_length)
bound = FreeEnergyBound(density=p_z)
optimizer = optim.RMSprop(flow.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

plot_density(p_z, directory='./distribution')

def should_log(iteration):
    return iteration % args.log_interval == 0

def should_plot(iteration):
    return iteration % args.plot_interval == 0

for iteration in range(1, iterations + 1):

    scheduler.step()

    samples = Variable(random_normal_samples(batch_size))
    zk, log_jacobians = flow(samples)

    optimizer.zero_grad()
    loss = bound(zk, log_jacobians)
    loss.backward()
    optimizer.step()

    # if should_log(iteration):
    #     print("Loss on iteration {}: {}".format(iteration , loss.data[0]))

    if should_plot(iteration):
            samples = Variable(random_normal_samples(args.plot_points))
            zk, det_grads = flow(samples)
            scatter_points(
                zk.data.numpy(),
                directory='./samples',
                iteration=iteration,
                flow_length=flow_length
            )