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
	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.5, .5], name='Y')

	net = Network([X, Y])

	xs = net.sample(1000).float().mean(0)

	assert (xs-torch.tensor([0.6, 0.5])).abs().max().item() < 0.05



def test_marginals():
	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.3, .8], name='Y')

	net = BernoulliNetwork([X, Y])

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
	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.3, .8], name='Y')

	net = BernoulliNetwork([X, Y])

	intervened = net.intervene(X=1)
	marginals = intervened.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor(1.0))
			and torch.allclose(marginals['Y'], torch.tensor(0.8)))

	intervened = net.intervene(Y=1)
	marginals = intervened.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor(0.6))
			and torch.allclose(marginals['Y'], torch.tensor(1.0)))



def test_categorical():

	X = Prior([0.1, 0.6, 0.3], name='X')

	yvals = torch.tensor([.3, .1, .8])
	Y = Conditional([X], torch.stack([1-yvals, yvals],-1), name='Y')

	net = Network([X, Y])

	marginals = net.marginals()

	assert (torch.allclose(marginals['X'], torch.tensor([0.1, 0.6, 0.3]))
			and torch.allclose(marginals['Y'], torch.tensor([0.67, 0.33])))


def test_confounding():

	U = Bernoulli(0.1)
	X = ConditionalBernoulli([U], [0.4, 0.6])
	Y = ConditionalBernoulli([U, X], [[0.3, 0.7], [0.8, 0.2]])
	net = BernoulliNetwork({'U': U, 'X': X, 'Y': Y})

	print(net)



def test_ate():
	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.3, .8], name='Y')

	net = BernoulliNetwork([X, Y])

	ate = net.ate('X')['Y'].item()



def test_trivial_ate_solver():

	X = Bernoulli(.6, name='X')
	Y = ConditionalBernoulli([X], [.3, .8], name='Y')
	net = BernoulliNetwork([X, Y])

	ate = net.ate('X')['Y'].item()

	solver = ATE_Sign(net)

	sol = Context().include(solver)
	sol.update({'treatment': 'X', 'outcome': 'Y'})

	ate = sol['ate']
	estimate = sol['estimate']

	assert abs(ate - estimate) < 0.01



def test_ate_solver():
	# Z = Bernoulli(.6, name='Z')
	# X = ConditionalBernoulli([Z], [.4, .2], name='X')
	# Y = ConditionalBernoulli([X, Z], [[.3, .8], [.2, .7]], name='Y')

	# Z1 = Bernoulli(name='Z1')
	# Z2 = Bernoulli(name='Z2')
	# X = ConditionalBernoulli([Z1, Z2], name='X')
	# Y = ConditionalBernoulli([X, Z1, Z2], name='Y')
	# net = BernoulliNetwork([Z1, Z2, X, Y])

	# Z = Bernoulli(name='Z')
	# X = ConditionalBernoulli([Z], name='X')
	# Y = ConditionalBernoulli([X, Z], name='Y')


	net = BernoulliNetwork({'Z': [], 'X': ['Z'], 'Y': ['X', 'Z']},
						   {'Z': 0.4996, 'X': [0.4299, 0.5579], 'Y': [[0.7305, 0.8980], [0.0884, 0.2453]]},)


	# Z = Bernoulli(.4996, name='Z')
	# X = ConditionalBernoulli([Z], [.4299, .5579], name='X')
	# Y = ConditionalBernoulli([X, Z], [[.7305, .8980], [.0884, .2453]], name='Y')
	# net = BernoulliNetwork([Z, X, Y])

	solver = ATE_Sign(net)

	sol = Context().include(solver)
	sol.update({'treatment': 'X', 'outcome': 'Y'})

	ate = sol['ate']
	estimate = sol['estimate']

	assert abs(ate - estimate) < 0.01











