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
E = DeepResistiveEnergy([(2, 28, 28), (1024,), (10,)], [1.0, 1.0], 480.)
E.set_device(dev)
C = SquaredError(E.layers()[-1]); net = Network(E)
params = E.params(); layers = E.layers(); free = net.free_layers()
aug = AugmentedFunction(E, C)
mt = QuadraticMinimizer(aug, free); mt.num_iterations = 4; mt.mode = 'asynchronous'
est = EquilibriumProp(params, layers, aug, C, mt); est.nudging = 1.0; est.variant = 'centered'
opt = Optimizer(E, C, [0.006, 0.006, 0.006, 0.006], 0., 0.)
mi = QuadraticMinimizer(E, free); mi.num_iterations = 4; mi.mode = 'asynchronous'
allp = params + C.params()
tr = torchvision.datasets.MNIST(root=DATA, train=True, download=False)
Xtr = tr.data.float().div(255).unsqueeze(1).to(dev); Ytr = tr.targets.to(dev)
def steps(n, s):
    for k in range(n):
        i = ((s + k) * 4) % 60000
        net.set_input(Xtr[i:i + 4], reset=False); mi.compute_equilibrium(); C.set_target(Ytr[i:i + 4])
        grads = est.compute_gradient()
        for p, g in zip(allp, grads): p.state.grad = g
        opt.step()
        for p in params: p.clamp_()
steps(30, 0)
torch.cuda.synchronize(); t0 = time.perf_counter(); steps(300, 30); torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 300
print('DRN-1H ms per EP step (batch 4): %.2f -> min per epoch: %.1f, 10 epochs: %.0f min' % (dt * 1000, dt * 15000 / 60, dt * 150000 / 60))
print('peak GPU MB: %.0f' % (torch.cuda.max_memory_allocated() / 2**20))
