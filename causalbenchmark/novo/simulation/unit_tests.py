from ..imports import *

from .models import Prior, Conditional, Network, Bernoulli, ConditionalBernoulli


	
def test_vars():
	b = Bernoulli(0.5)
	assert b.p == 0.5

	mu = b.sample(1000).view(-1).float().mean().item()
	assert abs(mu - 0.5) < 0.05

	p = Bernoulli(.6, varname='p')
	c = ConditionalBernoulli([p], [[.5, .5]], varname='c')

	xs = p.sample(1000)
	mu1 = xs.float().mean().item()
	assert abs(mu1 - 0.6) < 0.05
	ys = c.sample(1000, xs.unsqueeze(1))
	mu2 = ys.float().mean().item()
	assert abs(mu2 - 0.5) < 0.05



def test_net():
	p = Bernoulli(.6, varname='p')
	c = ConditionalBernoulli([p], [[.5, .5]], varname='c')

	net = Network([p, c])

	xs = net.sample(1000).float().mean(0)

	assert (xs-torch.tensor([0.6, 0.5])).abs().max().item() < 0.05






















