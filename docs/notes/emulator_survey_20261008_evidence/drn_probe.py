# DRN-XS probe: step time on this GPU, init zero fraction, floating-node NaN, scale gauge, clamp revival
import sys, time
EBL = 'C:/Users/KDI/AppData/Local/Temp/claude/C--Users-KDI-Downloads/bd065293-d22b-401b-83e9-b25b1031c6aa/scratchpad/ebl'
DATA = 'C:/Users/KDI/Downloads/Projects/subtractive-intelligence/data'
sys.path.insert(0, EBL)
import torch, torchvision
from model.resistive.network import DeepResistiveEnergy
from model.function.network import Network
from model.function.cost import SquaredError
from model.resistive.minimizer import QuadraticMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.monitor import Optimizer

torch.manual_seed(0)
dev = 'cuda'
E = DeepResistiveEnergy([(2, 28, 28), (100,), (10,)], [1.0, 1.0], 100.)
E.set_device(dev)
C = SquaredError(E.layers()[-1])
net = Network(E)
params = E.params(); layers = E.layers(); free = net.free_layers()
aug = AugmentedFunction(E, C)
mt = QuadraticMinimizer(aug, free); mt.num_iterations = 4; mt.mode = 'asynchronous'
est = EquilibriumProp(params, layers, aug, C, mt); est.nudging = 1.0; est.variant = 'centered'
opt = Optimizer(E, C, [0.006, 0.006, 0.006, 0.006], 0., 0.)
mi = QuadraticMinimizer(E, free); mi.num_iterations = 4; mi.mode = 'asynchronous'
allp = params + C.params()
W = [p for p in params if p.name.startswith('DenseWeight')]
print('params:', [p.name for p in params])
print('init zero fraction:', [round((w.get() == 0).float().mean().item(), 3) for w in W])

tr = torchvision.datasets.MNIST(root=DATA, train=True, download=False)
te = torchvision.datasets.MNIST(root=DATA, train=False, download=False)
Xtr = tr.data.float().div(255).unsqueeze(1).to(dev); Ytr = tr.targets.to(dev)
Xte = te.data.float().div(255).unsqueeze(1).to(dev); Yte = te.targets.to(dev)
perm = torch.randperm(Xtr.shape[0], device=dev)
Xtr = Xtr[perm]; Ytr = Ytr[perm]

def train_steps(n, start, keep=None):
    for k in range(n):
        i = ((start + k) * 4) % 60000
        net.set_input(Xtr[i:i + 4], reset=False)
        mi.compute_equilibrium()
        C.set_target(Ytr[i:i + 4])
        grads = est.compute_gradient()
        for p, g in zip(allp, grads): p.state.grad = g
        opt.step()
        for p in params: p.clamp_()
        if keep is not None:
            W[0].state.mul_(keep)

def evaluate(M=2000, bs=100):
    err = 0
    for i in range(0, M, bs):
        net.set_input(Xte[i:i + bs], reset=True)
        mi.compute_equilibrium()
        err += (layers[-1].state.argmax(1) != Yte[i:i + bs]).sum().item()
    return err / M

def pw(Va, Vb, G):
    return Va.pow(2) @ G.sum(1) + Vb.pow(2) @ G.sum(0) - 2 * ((Va @ G) * Vb).sum(1)

def free_power(x):
    net.set_input(x, reset=True); mi.compute_equilibrium()
    B = x.shape[0]
    V0 = layers[0].state.reshape(B, -1); V1 = layers[1].state; V2 = layers[2].state
    P = pw(V0, V1, W[0].get().reshape(-1, 100)) + pw(V1, V2, W[1].get())
    return P.mean().item(), layers[-1].state.clone()

# 1) step time
train_steps(50, 0)
torch.cuda.synchronize(); t0 = time.perf_counter()
train_steps(1000, 50)
torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 1000
print('ms per EP step (batch 4): %.2f -> est. min per epoch (15000 steps): %.1f' % (dt * 1000, dt * 15000 / 60))
train_steps(2000, 1050)
print('test error after 3050 steps (0.2 epoch), 2000 test imgs: %.3f' % evaluate())
print('zero fraction after 0.2 epoch:', [round((w.get() == 0).float().mean().item(), 3) for w in W])

# 2) scale gauge: scale all conductances and bias currents by 0.1
x = Xte[:200]
P1, o1 = free_power(x)
for p in params: p.state.mul_(0.1)
P2, o2 = free_power(x)
for p in params: p.state.mul_(10.0)
print('scale x0.1: output max abs diff %.2e (rel %.2e), power ratio %.4f' % ((o1 - o2).abs().max().item(), ((o1 - o2).abs().max() / o1.abs().max()).item(), P2 / P1))

# 3) floating node: remove all edges of hidden node 0
s0 = W[0].state.clone(); s1 = W[1].state.clone()
W[0].state[..., 0] = 0.; W[1].state[0, :] = 0.
_, o3 = free_power(x)
print('floating hidden node -> output has NaN: %s, inf: %s, hidden state node0 sample: %s' % (torch.isnan(o3).any().item(), torch.isinf(o3).any().item(), layers[1].state[:3, 0].tolist()))
W[0].state.copy_(s0); W[1].state.copy_(s1)

# 4) clamp revival: zero 90% of layer-1 edges once, train 300 steps without a mask
g = torch.Generator(device='cpu').manual_seed(1)
keep = (torch.rand(W[0].state.shape, generator=g) < 0.1).float().to(dev)
was_pos = (W[0].state > 0)
W[0].state.mul_(keep)
removed = (keep == 0) & was_pos
train_steps(300, 3050)
revived = ((W[0].state > 0) & (keep == 0)).float().sum().item()
print('pruned edges revived after 300 unmasked steps: %d of %d zeroed-out positions (%.1f%%)' % (revived, int((keep == 0).sum().item()), 100 * revived / (keep == 0).sum().item()))
print('peak GPU MB: %.0f' % (torch.cuda.max_memory_allocated() / 2**20))
