

import omnifig as fig

import numpy as np
import torch
from pomegranate.distributions import Categorical as _CategoricalBase
from pomegranate.distributions import ConditionalCategorical as _ConditionalCategoricalBase
from pomegranate.bayesian_network import BayesianNetwork as _BayesianNetworkBase



@fig.autocomponent('rng')
def get_rng(seed, reset_master=True):
	if reset_master:
		torch.manual_seed(seed)
	return torch.Generator().manual_seed(seed)



class Variable(fig.Configurable):
	def __init__(self, *, rng=None, varname=None, **kwargs):
		super().__init__(**kwargs)
		self._rng = rng
		self._varname = varname


	@staticmethod
	def process_raw_params(probs=None, *, logits=None, n=2, parentshapes=()):
		shape = parentshapes + (n,)
		if probs is None:
			logits = torch.randn(shape) if logits is None else torch.tensor(logits).float()
			probs = logits.softmax(dim=-1)
		else:
			probs = torch.tensor(probs).float()
			assert probs.min() >= 0
			probs = probs / probs.sum(dim=-1, keepdim=True)
		return probs


	@property
	def name(self) -> str:
		return self._varname
	@name.setter
	def name(self, value):
		self._varname = value


	@property
	def n(self) -> int:
		raise NotImplementedError


	@property
	def parents(self) -> tuple:
		return ()


	def implied_edges(self):
		return [(p, self) for p in self.parents]


	def __str__(self):
		name = getattr(self, 'name')
		if name is None:
			name = self.__class__.__name__
		return f'{name}({self.n})'


	def __repr__(self):
		return super().__str__()


	def as_hard_intervention(self, value: int):
		probs = torch.zeros(self.n)
		probs[value] = 1
		return Prior(probs=probs, name=self.name)



@fig.component('cat')
class Prior(Variable, _CategoricalBase):
	def __init__(self, probs=None, *, logits=None, n=2, **kwargs):
		probs = self.process_raw_params(probs, logits=logits, n=n)
		super().__init__(probs=[probs.tolist()], **kwargs)


	@property
	def n(self) -> int:
		return self.probs.shape[-1]



@fig.component('bern')
class Bernoulli(Prior):
	def __init__(self, prob=None, *, logit=None, probs=None, logits=None, **kwargs):
		if prob is not None:
			probs = [1-prob, prob]
		elif logit is not None:
			logits = [-logit, logit]
		super().__init__(probs=probs, logits=logits, n=2, **kwargs)


	@property
	def p(self) -> float:
		return self.probs[0][1]


	def __str__(self):
		name = getattr(self, 'name')
		if name is None:
			name = self.__class__.__name__
		return f'{name}({self.p:.2f})'



@fig.component('concat')
class Conditional(Variable, _ConditionalCategoricalBase):
	def __init__(self, parents, probs=None, *, logits=None, n=2, **kwargs):
		probs = self.process_raw_params(probs, logits=logits, n=n, parentshapes=tuple(p.n for p in parents))
		super().__init__(probs=probs.tolist(), **kwargs)
		self._parents = tuple(parents)


	@property
	def parents(self) -> tuple:
		return self._parents


	@property
	def n(self) -> int:
		return self.n_categories[0][-1]



@fig.component('conbern')
class ConditionalBernoulli(Conditional):
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
		edges = [e for v in variables.values() for e in v.implied_edges()]
		super().__init__(distributions=nodes, edges=edges, **kwargs)
		self._variables = variables
		self._rng = rng


	def __str__(self):
		return f'{self.__class__.__name__}({", ".join(self._variables.keys())})'


	def __repr__(self):
		return super().__str__()


	def __getitem__(self, item):
		return self._variables[item]


	def __len__(self):
		return len(self._variables)



	def marginals(self, **conds):
		pass

	
	# def interventions(self, **conds):
	# 	pass








