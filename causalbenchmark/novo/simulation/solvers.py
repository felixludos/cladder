from itertools import product
from omniply import ToolKit, tool



class Solver(ToolKit):
	def __init__(self, net, **kwargs):
		super().__init__(**kwargs)
		self.net = net



class ATE_Sign(Solver):
	@tool('estimand')
	def get_estimand(self, treatment, outcome):
		t, o, backdoors = self.net.backdoor_estimand(treatment, outcome)
		assert t == treatment
		assert o == outcome
		return treatment, outcome, backdoors


	@tool('backdoors')
	def get_conditions(self, estimand):
		return estimand[2]


	@tool('terms')
	def get_terms(self, treatment, outcome, backdoors):
		terms = []
		for bd in backdoors:
			terms.append((bd, {}))
		for tval in [0, 1]:
			if backdoors:
				for cvals in product([0, 1], repeat=len(backdoors)):
					terms.append((outcome, {treatment: tval, **dict(zip(backdoors, cvals))}))
			else:
				terms.append((outcome, {treatment: tval}))
		return terms


	@tool('values')
	def get_term_values(self, terms):
		vals = []
		for term in terms:
			var, conds = term
			val = self.net.marginals(**conds)[var].item()
			vals.append(val)
		return vals


	@tool('estimate')
	def get_estimate(self, treatment, outcome, terms, values):
		conds = {term[0]: val for term, val in zip(terms, values) if term[0] != outcome}
		conds = {(cond, cval): val if cval == 1 else 1 - val for cond, val in conds.items() for cval in [0, 1]}
		vals = []
		for term, val in zip(terms, values):
			if term[0] == outcome:
				sign = 1 if term[1][treatment] == 1 else -1
				gates = [conds[cond,cval] for cond, cval in term[1].items() if cond != treatment]
				vals.append(sign * val)
				for gate in gates:
					vals[-1] *= gate
		return sum(vals)


	@tool('ate')
	def get_ate(self, treatment, outcome):
		# return self.net.ate(treatment)[outcome].item()
		return self.net.ate(treatment, outcome).item()


	@tool('ground_truth')
	def get_ground_truth(self, estimate, criterion):
		yes = estimate > 0 if criterion == '>' else estimate < 0
		return 'yes' if yes else 'no'













