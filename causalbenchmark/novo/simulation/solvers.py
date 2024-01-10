from itertools import product
from omniply import ToolKit, tool



class Solver(ToolKit):
	@tool('ground_truth')
	def get_ground_truth(self, leftestimate, rightestimate, *, criterion='>'):
		yes = leftestimate > rightestimate if criterion == '>' else leftestimate < rightestimate
		return 1. if yes else 0.



class Simple_Solver(Solver):
	'''Uses a single instance of the model to solve the problem'''
	def __init__(self, net, **kwargs):
		super().__init__(**kwargs)
		self.net = net



class ATE_Sign(Simple_Solver):
	@tool('backdoors')
	def get_estimand(self, treatment, outcome):
		t, o, backdoors = self.net.backdoor_estimand(treatment, outcome)
		assert t == treatment
		assert o == outcome
		return backdoors


	@tool('terms')
	def get_terms(self, treatment, outcome):
		return self.net.ate_terms(treatment, outcome)


	@tool('values')
	def get_term_values(self, terms):
		vals = []
		for term in terms:
			var, conds = term
			val = self.net.marginals(**conds)[var].item()
			vals.append(val)
		return vals


	@tool('estimate')
	def get_ate(self, treatment, outcome):
		return self.net.ate(treatment, outcome).item()


	@tool('leftestimate')
	def get_estimate(self, estimate):
		return estimate
	@tool('rightestimate')
	def _get_zero(self):
		return 0.
















