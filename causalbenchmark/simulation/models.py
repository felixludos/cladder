from omnibelt import toposort
import omnifig as fig

import math
from itertools import product
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import torch
from dowhy import CausalModel
from torch import nn
from pomegranate.distributions import Categorical as _CategoricalBase
from pomegranate.distributions import ConditionalCategorical as _ConditionalCategoricalBase
from pomegranate.bayesian_network import BayesianNetwork as _BayesianNetworkBase



class Variable(fig.Configurable):
	'''
	Generally created automatically by a Network, not directly.

	Explicitly keeps track of variable specific properties, such as name or parameters,
	and implicitly tracks network dependent features like parents.
	'''
	def __init__(self, *, name: str = None, rng = None, net: 'Network' = None, **kwargs):
		super().__init__(**kwargs)
		self._net = net
		self._name = name
		self._rng = rng


	@staticmethod
	def process_raw_params(probs=None, *, logits=None, n=2, parentshapes=()):
		shape = parentshapes + (n,)
		if probs is None:
			logits = torch.randn(shape) if logits is None else torch.tensor(logits).float()
			probs = logits.softmax(dim=-1)
		else:
			probs = torch.tensor(probs).float() if isinstance(probs, (list, tuple, float, int)) else probs.clone().float()
			assert probs.min() >= 0
			probs = probs / probs.sum(dim=-1, keepdim=True)
		return probs


	@property
	def name(self) -> str:
		return self._name
	@name.setter
	def name(self, value: str):
		self._name = value


	@property
	def n(self) -> int:
		raise NotImplementedError


	@property
	def parents(self) -> tuple:
		return ()


	@property
	def param(self):
		raise NotImplementedError


	def __repr__(self):
		name = getattr(self, 'name')
		if name is None:
			name = self.__class__.__name__

		suffix = ''
		if len(self.parents):
			suffix = f' | {", ".join(p.name for p in self.parents)}'

		return f'{name}(n={self.n}{suffix})'


	# def as_hard_intervention(self, value: int):
	# 	probs = torch.zeros(self.n)
	# 	probs[value] = 1
	# 	return Prior(probs=probs, name=self.name)



class BernoulliVariable(Variable):
	@property
	def param(self):
		return self.probs[0][...,1]


	# def as_intervention(self, value: float):
	# 	assert 0 <= value <= 1, f'Value must be in [0, 1], got {value}'
	# 	return Bernoulli(prob=value, name=self.name)
	#
	#
	# def as_hard_intervention(self, value: int):
	# 	return self.as_intervention(value=value)


	def __repr__(self):
		name = getattr(self, 'name')
		if name is None: name = ''
		if len(self.parents):
			parent_details = ", ".join(p.name for p in self.parents) if all(p.name is not None for p in self.parents) \
				else f'{len(self.parents)} parent{"s" if len(self.parents) > 1 else ""}'
			suffix = f'{"" if len(name) else " ·"} | {parent_details}'
		else:
			suffix = f'{"=" if len(name) else ""}{self.p:.2f}'.rstrip('0').rstrip('.')
		return f'p({name}{suffix})'


	def set_params(self, params: torch.FloatTensor):
		val = torch.stack([1 - params, params], dim=-1).view(-1)
		self._net._factor_mapping[self].probs.data.view(-1).copy_(val)
		self.probs[0].data.view(-1).copy_(val)


	def get_params(self) -> torch.FloatTensor:
		return self.probs[0].data[...,1].view(-1)


	def num_params(self):
		return 2 ** len(self.parents)



@fig.component('cat')
class Prior(Variable, _CategoricalBase):
	def __init__(self, probs=None, logits=None, n=2, **kwargs):
		probs = self.process_raw_params(probs, logits=logits, n=n)
		super().__init__(probs=[probs.tolist()], **kwargs)


	@property
	def n(self) -> int:
		return self.probs.shape[-1]



@fig.component('bern')
class Bernoulli(Prior, BernoulliVariable):
	def __init__(self, probs=None, *, logits=None, **kwargs):
		if probs is not None:
			probs = [1-probs, probs]
		elif logits is not None:
			logits = [-logits/2, logits/2]
		super().__init__(probs=probs, logits=logits, n=2, **kwargs)


	@property
	def p(self) -> float:
		return self.probs[0][1]



# @fig.component('concat')
# class Conditional(Variable, _ConditionalCategoricalBase):
# 	def __init__(self, parents, probs=None, *, logits=None, n=2, **kwargs):
# 		# IMPORTANT: don't rely on parents for sampling or inference, only for structure
# 		probs = self.process_raw_params(probs, logits=logits, n=n, parentshapes=tuple(p.n for p in parents))
# 		super().__init__(probs=[probs.tolist()], **kwargs)
# 		self._parents = tuple(parents)
#
#
# 	@property
# 	def parents(self) -> tuple:
# 		return self._parents
#
#
# 	@property
# 	def n(self) -> int:
# 		return self.n_categories[0][-1]



@fig.component('concat')
class Conditional(Variable, _ConditionalCategoricalBase):
	def __init__(self, parents: list[Variable], probs=None, logits=None, n=2, **kwargs):
		probs = self.process_raw_params(probs, logits=logits, n=n, parentshapes=tuple(p.n for p in parents))
		super().__init__(probs=[probs.tolist()], **kwargs)
		self._parents = tuple(parents)


	@property
	def parents(self) -> tuple:
		return self._net.parents_of(self)


	@property
	def n(self) -> int:
		return self.n_categories[0][-1]


@fig.component('conbern')
class ConditionalBernoulli(Conditional, BernoulliVariable):
	def __init__(self, parents: list[Variable], probs=None, *, logits=None, **kwargs):
		if probs is not None:
			probs = torch.tensor(probs).float()
			probs = torch.stack([1-probs, probs], dim=-1)
		elif logits is not None:
			logits = torch.tensor(logits).float()
			logits = torch.stack([-logits, logits], dim=-1) / 2
		super().__init__(parents=parents, probs=probs, logits=logits, **kwargs)





# @fig.component('net')
# class Network(fig.Configurable, _BayesianNetworkBase):
# 	def __init__(self, variables: list['Variable'] | dict[str, 'Variable'], *, rng=None, **kwargs):
# 		if isinstance(variables, (list, tuple)):
# 			variables = {v.name: v for v in variables}
# 		for name, v in variables.items():
# 			v.name = name
# 		nodes = [v for v in variables.values()]
# 		edges = [(variables[s], variables[e]) for v in variables.values() for s, e in v.implied_edges()]
# 		super().__init__(distributions=nodes, edges=edges, **kwargs)
# 		self._variables = variables
# 		self._rng = rng
# 		order = toposort({v.name: [p.name for p in v.parents] for v in self._variables.values()})
# 		self.vars = [self._variables[name] for name in order]



@fig.component('net')
class Network(fig.Configurable, _BayesianNetworkBase):
	_prior_type = Prior
	_conditional_type = Conditional

	def __init__(self, connectivity: dict[str, list[str]], probs: dict[str, list[float]] = None, *, rng=None, **kwargs):
		probs = probs or {}
		order = toposort(connectivity)

		variables = {}
		nodes = []
		edges = []

		for name in order:
			parent_names = connectivity[name]
			if len(parent_names):
				parents = [variables[p] for p in parent_names]
				variables[name] = self._conditional_type(parents=parents, name=name, rng=rng,
														 probs=probs.get(name, None), net=self)
			else:
				variables[name] = self._prior_type(name=name, rng=rng, probs=probs.get(name, None), net=self)
			nodes.append(variables[name])
			edges.extend((variables[p], variables[name]) for p in parent_names)

		super().__init__(distributions=nodes, edges=edges, **kwargs)
		self._connectivity = connectivity
		self._variables = variables
		self._rng = rng
		self.vars = nodes


	def __getitem__(self, item):
		return self._variables[item]


	def __len__(self):
		return len(self._variables)


	def parents_of(self, var: Variable | str):
		if isinstance(var, str):
			var = self._variables[var]
		return [self._variables[parent] for parent in self._connectivity[var.name]]


	def marginals(self, **conds: int):
		mask = []
		vals = []

		conds = {k: torch.tensor(v).long() for k, v in conds.items()}
		size = next(iter(conds.values())).size() if len(conds) else ()
		default_size = len(size) == 0
		neutral = torch.zeros(size).long()

		for var in self.vars:
			mask.append(var.name in conds)
			vals.append(conds.get(var.name, neutral))

		mask = torch.tensor(mask).view(-1, len(self.vars))
		vals = torch.stack(vals, dim=-1).view(-1, len(self.vars))
		X = torch.masked.MaskedTensor(vals, mask)

		mars = self.predict_proba(X)
		return {var.name: mar.squeeze(0) if default_size else mar for var, mar in zip(self.vars, mars)}


	def intervene(self, **interventions: int):
		connectivity = self._connectivity.copy()
		probs = {var.name: var.param for var in self.vars}
		for vname, val in interventions.items():
			connectivity[vname] = []
			probs[vname] = val

		return self.__class__(connectivity=connectivity, probs=probs, rng=self._rng)


	def old_ate(self, treatment: str, *, conditions: dict[str, int] = None, treated_val = 1, not_treated_val = 0):
		'''sometimes produces inconsistent results, not sure why, for now use ate() instead'''
		if conditions is None: conditions = {}
		treated = self.intervene(**{treatment: treated_val}).marginals(**conditions)
		not_treated = self.intervene(**{treatment: not_treated_val}).marginals(**conditions)
		return {var.name: treated[var.name] - not_treated[var.name] for var in self.vars}


	def ate_terms(self, treatment: str, outcome: str, *, conditions: dict[str, int] = None,
				  treated_val = 1, not_treated_val = 0):
		estimand = self.backdoor_estimand(treatment, outcome)
		if estimand is None:
			return []
		assert estimand[0] == treatment, f'Expected treatment to be {treatment}, got {estimand[0]}'
		assert estimand[1] == outcome, f'Expected outcome to be {outcome}, got {estimand[1]}'
		conditions = conditions or {}
		backdoors = [b for b in estimand[2] if b not in conditions]

		terms = []
		for b in backdoors:
			terms.append((b, conditions.copy()))

		integration = [dict(zip(backdoors, cvals)) for cvals in product(*[list(range(self[b].n)) for b in backdoors])] \
			if backdoors else [{}]
		for tval in [treated_val, not_treated_val]:
			terms.extend((outcome, {treatment: tval, **conditions, **element}) for element in integration)

		return terms


	def ate(self, treatment: str, outcome: str, *, conditions: dict[str, int] = None,
			treated_val = 1, not_treated_val = 0):
		return self.ate_estimate(treatment, outcome, conditions=conditions,
								 treated_val=treated_val, not_treated_val=not_treated_val).item()


	def ate_estimate(self, treatment: str, outcome: str, conditions: dict[str, int] = None, terms = None,
					 *, treated_val = 1, not_treated_val = 0):
		if terms is None:
			terms = self.ate_terms(treatment, outcome, conditions=conditions,
							   treated_val=treated_val, not_treated_val=not_treated_val)
		if not len(terms):
			return torch.tensor(0.0)
			# return None
		conditions = conditions or {}
		if any(term[0] != outcome for term in terms):
			values = self.marginals(**conditions)
			wts = {var: values[var] for var, _ in terms if var != outcome}
			wts = {(cond, cval): val if cval == 1 else 1 - val for cond, val in wts.items() for cval in [0, 1]}

		ate = []
		for term in terms:
			var, conds = term
			if var == outcome:
				sign = 1 if term[1][treatment] == treated_val else -1
				gates = [wts[cond,cval] for cond, cval in conds.items()
						 if cond != treatment and cond not in conditions]
				ate.append(sign * self.marginals(**conds)[var]
						   * (torch.prod(torch.stack(gates)) if len(gates) else 1))
		return sum(ate)


	def covariance(self, var1: str, var2: str, **conditions: int) -> float:
		'''cov(A, B) = E[AB] - E[A]*E[B] = (E[A|B] - E[A])*E[B]'''
		marginals = self.marginals(**conditions)
		p1, p2 = marginals[var1], marginals[var2]
		p1g2 = self.marginals(**{var2: 1}, **conditions)[var1]
		return (p1g2 - p1) * p2


	def variances(self, **conditions: int) -> dict[str, float]:
		return {vname: p*(1-p) for vname, p in self.marginals(**conditions).items()}


	def correlation(self, var1: str, var2: str, **conditions: int) -> float:
		sigmas = self.variances(**conditions)
		return self.covariance(var1, var2, **conditions) / np.sqrt(sigmas[var1] * sigmas[var2])


	def to_networkx(self):
		G = nx.DiGraph()
		for var in self.vars:
			G.add_node(var.name)
			for parent in var.parents:
				G.add_edge(parent.name, var.name)
		return G


	def to_dowhy(self, treatment: str, outcome: str):
		G = self.to_networkx()
		pydot_graph = to_pydot(G)
		dot_graph_str = pydot_graph.to_string()
		dummy_data = pd.DataFrame({var.name: [0] for var in self.vars})
		return CausalModel(dummy_data, treatment=treatment, outcome=outcome, graph=dot_graph_str)


	def _parse_backdoor_estimand(self, estimand, treatment: str, outcome: str):
		do = estimand.args[1][0].args[0][0].name
		exp = estimand.args[0].args[0].name
		if '|' in exp:
			out, cond = exp.split('|')
			cond = cond.split(',')
		else:
			out = exp
			cond = []

		assert treatment == do, f'Expected treatment to be {do}, got {treatment}'
		assert outcome == out, f'Expected outcome to be {out}, got {outcome}'

		return do, out, cond


	def _backdoor_terms(self, do: str, out: str, cond: list[str] = None):
		if cond is None:
			cond = []

		terms = []
		for c in cond:
			terms.append({'prob': c})
		parents = [do, *cond]
		for cvals in product(*[[0, 1] for c in cond]):
			terms.append({'prob': out, 'cond': {p: v for p, v in zip(parents, cvals)}})
		return terms


	def backdoor_estimand(self, treatment: str, outcome: str):
		model = self.to_dowhy(treatment, outcome)

		estimand_info = model.identify_effect()
		if estimand_info is not None and estimand_info.estimands is not None:
			estimand = estimand_info.estimands[estimand_info.default_backdoor_id]['estimand']
			return self._parse_backdoor_estimand(estimand, treatment, outcome)



@fig.component('bernet')
class BernoulliNetwork(Network):
	_prior_type = Bernoulli
	_conditional_type = ConditionalBernoulli

	vars: list[BernoulliVariable]

	# def __init__(self, variables: list[BernoulliVariable] | dict[str, BernoulliVariable], **kwargs):
	# 	super().__init__(variables, **kwargs)
	# 	assert all(isinstance(v, BernoulliVariable) for v in self.vars)


	def __repr__(self):
		return ' '.join(str(v) for v in reversed(self.vars))


	def marginals(self, **conds: int):
		mars = super().marginals(**conds)
		return {name: mar[..., 1] for name, mar in mars.items()}


	# def intervene(self, **interventions: float):
	# 	newvars = [var.as_intervention(interventions[var.name]) if var.name in interventions else var
	# 			   for var in self.vars]
	# 	return self.__class__(variables=newvars, rng=self._rng)


	def set_params(self, params: torch.FloatTensor):
		params = params.view(-1)
		for var in self.vars:
			N = var.num_params()
			assert len(params) >= N, f'Expected at least {N} parameters, got {len(params)}'
			var.set_params(params[:N])
			params = params[N:]
		assert len(params) == 0


	def get_params(self):
		return torch.cat([v.get_params().view(-1) for v in self.vars], dim=-1)


	def num_params(self):
		return sum(v.num_params() for v in self.vars)






