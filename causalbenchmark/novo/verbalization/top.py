from typing import Any
import random

import omnifig as fig
from omniply import Context, ToolKit

import torch

from .. import misc
from .base import VerbalizationBase
from .decision import Decision
from ..templating import Template
from .clause import MarginalVerbalization, ConditionalVerbalization


class Verbalizer(Context, VerbalizationBase):
	def __init__(self, rng=None, **kwargs):
		super().__init__(**kwargs)
		self._rng = misc.get_rng(rng)


	def _convert_verbalization_to_tools(self, info: dict[str, str | list[str]], label_fmt='{key}'):
		tools = []
		for key, options in info.items():
			label = label_fmt.format(key=key)
			if isinstance(options, list):
				new = self._DecisionType([self._TemplateType(tmpl, label) for tmpl in options], name=label)
			elif isinstance(options, str):
				new = self._TemplateType(options, label)
			else:
				continue
			tools.append(new)
		return tools


	def populate_variable_info(self, variable: dict[str, str | list[str]],
							   conditions: list[dict[str, str | list[str]]] = None, verbalization=None, **static):
		self.extend(self._convert_verbalization_to_tools(variable))
		if conditions is not None:
			assert len(conditions) > 0, 'Must have at least one condition'
			for i, condition in enumerate(conditions):
				key = '{key}'
				self.extend(self._convert_verbalization_to_tools(condition, label_fmt=f'c{i}_{key}'))
			self['num_conditions'] = len(conditions)

		if verbalization is None:
			verbalization = MarginalVerbalization() if conditions is None else ConditionalVerbalization()
		self.include(verbalization)
		self.update(static)
		return self


	def decisions(self, gizmo: str = None):
		for gadget in self._vendors(gizmo=gizmo):
			if isinstance(gadget, Decision):
				yield gadget


	def identity(self, gizmo: str = None):
		identity = {decision.identity(): None for decision in self.decisions(gizmo=gizmo)}
		identity.update({key: self[key] for key in self.cached() if key in identity})
		if gizmo is None:
			return {k:v for k,v in identity.items() if v is not None}
		return identity.get(gizmo)


	def select(self, choices: list[Any]):
		'''use self.rng (torch.Generator) to choose an element'''
		return choices[torch.randint(0, len(choices), (1,), generator=self._rng).item()]















