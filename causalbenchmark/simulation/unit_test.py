from ..imports import *

from .models import Prior, Conditional, Network, Bernoulli, ConditionalBernoulli, BernoulliNetwork
from .solvers import ATE_Sign



def test_vars():
	b = Bernoulli(0.5)
	assert b.p == 0.5

	mu = b.sample(1000).view(-1).float().mean().item()
	assert abs(mu - 0.5) < 0.05

	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.5, .5], name='Y')

	xs = X.sample(1000)
	mu1 = xs.float().mean().item()
	assert abs(mu1 - 0.6) < 0.05
	ys = Y.sample(1000, xs.unsqueeze(1))
	mu2 = ys.float().mean().item()
	assert abs(mu2 - 0.5) < 0.05



def test_net():
	net = BernoulliNetwork({'Y': ['X'], 'X': []}, {'X': 0.6, 'Y': [0.5, 0.5]})

	xs = net.sample(1000).float().mean(0)

	assert (xs-torch.tensor([0.6, 0.5])).abs().max().item() < 0.05



def test_marginals():
	net = BernoulliNetwork({'Y': ['X'], 'X': []}, {'X': 0.6, 'Y': [0.3, 0.8]})

	marginals = net.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor(0.6))
			and torch.allclose(marginals['Y'], torch.tensor(0.6)))

	marginals = net.marginals(X=1)

	assert (torch.allclose(marginals['X'], torch.tensor(1.0))
			and torch.allclose(marginals['Y'], torch.tensor(0.8)))

	marginals = net.marginals(Y=1)

	assert (torch.allclose(marginals['X'], torch.tensor(0.8))
			and torch.allclose(marginals['Y'], torch.tensor(1.0)))



def test_interventions():
	net = BernoulliNetwork({'Y': ['X'], 'X': []}, {'X': 0.6, 'Y': [0.3, 0.8]})

	intervened = net.intervene(X=1)
	marginals = intervened.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor(1.0))
			and torch.allclose(marginals['Y'], torch.tensor(0.8)))

	intervened = net.intervene(Y=1)
	marginals = intervened.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor(0.6))
			and torch.allclose(marginals['Y'], torch.tensor(1.0)))



def test_categorical():
	yvals = torch.tensor([.3, .1, .8])
	net = Network({'X': [], 'Y': ['X']},
				  {'X': [0.1, 0.6, 0.3], 'Y': torch.stack([1-yvals, yvals],-1)})

	marginals = net.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor([0.1, 0.6, 0.3]))
			and torch.allclose(marginals['Y'], torch.tensor([0.67, 0.33])))



def test_confounding():
	net = BernoulliNetwork({'U': [], 'X': ['U'], 'Y': ['U', 'X']},
						   {'U': 0.1, 'X': [0.4, 0.6], 'Y': [[0.3, 0.7], [0.8, 0.2]]})

	marginals = net.marginals()

	assert abs(marginals['U'].item() - 0.1) < 0.01

	print()
	print(net)



def test_ate():
	net = BernoulliNetwork({'X': [], 'Y': ['X']}, {'X': 0.6, 'Y': [0.3, 0.8]})

	ate = net.ate('X', 'Y')

	assert abs(ate - 0.5) < 0.01



def test_trivial_ate_solver():
	net = BernoulliNetwork({'X': [], 'Y': ['X']}, {'X': 0.6, 'Y': [0.3, 0.8]})

	ate = net.ate('X', 'Y')

	solver = ATE_Sign(net)

	sol = Context().include(solver)
	sol.update({'treatment': 'X', 'outcome': 'Y'})

	estimate = sol['estimate']

	assert abs(ate - estimate) < 0.01



def test_ate_solver():
	net = BernoulliNetwork({'Z': [], 'X': ['Z'], 'Y': ['X', 'Z']},
						   {'Z': 0.4996, 'X': [0.4299, 0.5579], 'Y': [[0.7305, 0.8980], [0.0884, 0.2453]]},)

	ate = net.ate('X', 'Y')

	solver = ATE_Sign(net)

	sol = Context().include(solver)
	sol.update({'treatment': 'X', 'outcome': 'Y'})

	estimate = sol['estimate']

	assert abs(ate - estimate) < 0.01











