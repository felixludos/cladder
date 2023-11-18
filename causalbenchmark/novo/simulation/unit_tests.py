from ..imports import *

from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork


def test_pom():
	d1 = Categorical([[0.1, 0.9]])
	d2 = ConditionalCategorical([[[0.4, 0.6], [0.3, 0.7]]])

	model = BayesianNetwork([d1, d2], [(d1, d2)])


	























