from omnibelt import toposort
import omnifig as fig

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
	def __init__(self, *, rng=None, name=None, **kwargs):
		super().__init__(**kwargs)
		self._rng = rng
		self._name = name


	@staticmethod
	def process_raw_params(probs=None, *, logits=None, n=2, parentshapes=()):
		shape = parentshapes + (n,)
		if probs is None:
			logits = torch.randn(shape) if logits is None else torch.tensor(logits).float()
			probs = logits.softmax(dim=-1)
		else:
			probs = torch.tensor(probs).float() if isinstance(probs, (list, tuple)) else probs.clone().float()
			assert probs.min() >= 0
			probs = probs / probs.sum(dim=-1, keepdim=True)
		return probs


	@property
	def name(self) -> str:
		return self._name
	@name.setter
	def name(self, value):
		self._name = value


	@property
	def n(self) -> int:
		raise NotImplementedError


	@property
	def parents(self) -> tuple:
		return ()


	def implied_edges(self):
		return [(parent.name, self.name) for parent in self.parents]


	def __repr__(self):
		name = getattr(self, 'name')
		if name is None:
			name = self.__class__.__name__

		suffix = ''
		if len(self.parents):
			suffix = f' | {", ".join(p.name for p in self.parents)}'

		return f'{name}(n={self.n}{suffix})'


	def as_hard_intervention(self, value: int):
		probs = torch.zeros(self.n)
		probs[value] = 1
		return Prior(probs=probs, name=self.name)



class BernoulliVariable(Variable):
	@property
	def param(self):
		return self.probs[0][...,1]


	def as_intervention(self, value: float):
		assert 0 <= value <= 1, f'Value must be in [0, 1], got {value}'
		return Bernoulli(prob=value, name=self.name)


	def as_hard_intervention(self, value: int):
		return self.as_intervention(value=value)


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
		self.probs[0].data.view(-1).copy_(torch.stack([1 - params, params], dim=-1).view(-1))


	def get_params(self) -> torch.FloatTensor:
		return self.probs[0].data.view(-1)


	def num_params(self):
		return 2 ** len(self.parents)



@fig.component('cat')
class Prior(Variable, _CategoricalBase):
	def __init__(self, probs=None, *, logits=None, n=2, **kwargs):
		probs = self.process_raw_params(probs, logits=logits, n=n)
		super().__init__(probs=[probs.tolist()], **kwargs)


	@property
	def n(self) -> int:
		return self.probs.shape[-1]



@fig.component('bern')
class Bernoulli(Prior, BernoulliVariable):
	def __init__(self, prob=None, *, logit=None, probs=None, logits=None, **kwargs):
		if prob is not None:
			probs = [1-prob, prob]
		elif logit is not None:
			logits = [-logit, logit]
		super().__init__(probs=probs, logits=logits, n=2, **kwargs)


	@property
	def p(self) -> float:
		return self.probs[0][1]



@fig.component('concat')
class Conditional(Variable, _ConditionalCategoricalBase):
	def __init__(self, parents, probs=None, *, logits=None, n=2, **kwargs):
		# IMPORTANT: don't rely on parents for sampling or inference, only for structure
		probs = self.process_raw_params(probs, logits=logits, n=n, parentshapes=tuple(p.n for p in parents))
		super().__init__(probs=[probs.tolist()], **kwargs)
		self._parents = tuple(parents)


	@property
	def parents(self) -> tuple:
		return self._parents


	@property
	def n(self) -> int:
		return self.n_categories[0][-1]



@fig.component('conbern')
class ConditionalBernoulli(Conditional, BernoulliVariable):
	def __init__(self, parents, conds=None, *, logit_conds=None, probs=None, logits=None, **kwargs):
		if conds is not None:
			probs = torch.tensor(conds).float()
			probs = torch.stack([1-probs, probs], dim=-1)
		elif logit_conds is not None:
			logits = torch.tensor(logit_conds).float()
			logits = torch.stack([-logits, logits], dim=-1)
		super().__init__(parents, probs=probs, logits=logits, **kwargs)



@fig.component('net')
class Network(fig.Configurable, _BayesianNetworkBase):
	def __init__(self, variables: list['Variable'] | dict[str, 'Variable'], *, rng=None, **kwargs):
		if isinstance(variables, (list, tuple)):
			variables = {v.name: v for v in variables}
		for name, v in variables.items():
			v.name = name
		nodes = [v for v in variables.values()]
		edges = [(variables[s], variables[e]) for v in variables.values() for s, e in v.implied_edges()]
		super().__init__(distributions=nodes, edges=edges, **kwargs)
		self._variables = variables
		self._rng = rng
		order = toposort({v.name: [p.name for p in v.parents] for v in self._variables.values()})
		self.vars = [self._variables[name] for name in order]


	# def __repr__(self):
	# 	return f'{self.__class__.__name__}({", ".join(v.name for v in self.vars)})'


	def __getitem__(self, item):
		return self._variables[item]


	def __len__(self):
		return len(self._variables)


	def parents_of(self, var: Variable | str):
		if isinstance(var, str):
			var = self._variables[var]
		return [self._variables[parent.name] for parent in var.parents]


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
		newvars = [var.as_hard_intervention(interventions[var.name]) if var.name in interventions else var
				   for var in self.vars]
		return self.__class__(variables=newvars, rng=self._rng)


	def ate(self, treatment: str, *, treated_val=1, not_treated_val=0):
		treated = self.intervene(**{treatment: treated_val}).marginals()
		not_treated = self.intervene(**{treatment: not_treated_val}).marginals()
		return {var.name: treated[var.name] - not_treated[var.name] for var in self.vars}


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
		estimand = estimand_info.estimands[estimand_info.default_backdoor_id]['estimand']
		return self._parse_backdoor_estimand(estimand, treatment, outcome)



@fig.component('bernet')
class BernoulliNetwork(Network):
	vars: list[BernoulliVariable]

	def __init__(self, variables: list[BernoulliVariable] | dict[str, BernoulliVariable], **kwargs):
		super().__init__(variables, **kwargs)
		assert all(isinstance(v, BernoulliVariable) for v in self.vars)


	def __repr__(self):
		return ' '.join(str(v) for v in reversed(self.vars))


	def marginals(self, **conds: int):
		mars = super().marginals(**conds)
		return {name: mar[..., 1] for name, mar in mars.items()}


	def intervene(self, **interventions: float):
		newvars = [var.as_intervention(interventions[var.name]) if var.name in interventions else var
				   for var in self.vars]
		return self.__class__(variables=newvars, rng=self._rng)


	def set_params(self, params: torch.FloatTensor):
		params = params.view(-1)
		for var in self.vars:
			var.set_params(params[:var.num_params()])
			params = params[var.num_params():]


	def get_params(self):
		return torch.cat([v.get_params().view(-1) for v in self.vars], dim=-1)


	def num_params(self):
		return sum(v.num_params() for v in self.vars)






