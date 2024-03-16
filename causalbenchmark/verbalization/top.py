from typing import Any, Iterator
import random

import omnifig as fig
from omniply import Context, ToolKit, Scope, Selection, tool
from omniply.core.abstract import AbstractGadget, AbstractGig

import torch

from .. import misc
from .base import VerbalizationBase
from .decision import Decision, AbstractDecision
from ..templating import Template
from .variable import VariableVerbalization
from .clause import MarginalVerbalization, ConditionalVerbalization



class VerbalizerBase(AbstractGig, VerbalizationBase):
	def __init__(self, *, rng=None, **kwargs):
		super().__init__(**kwargs)
		self._rng = misc.get_rng(rng)

	def select(self, choices: list[Any]):
		'''use self.rng (torch.Generator) to choose an element'''
		return choices[torch.randint(0, len(choices), (1,), generator=self._rng).item()]


	def decisions(self, gizmo: str = None):
		for gadget in self._vendors(gizmo=gizmo):
			if isinstance(gadget, AbstractDecision):
				yield gadget


	def identity_keys(self, gizmo: str = None):
		for decision in self.decisions(gizmo=gizmo):
			yield from decision.identity_keys()


	def identity(self, gizmo: str = None):
		identity = {}
		for ident in self.identity_keys(gizmo=gizmo):
			if self.is_cached(ident):
				identity[ident] = self[ident]
		return identity



class Verbalizer(Context, VerbalizerBase):
	def __init__(self, variable: VariableVerbalization = None, conditions: list[VariableVerbalization] = (), **kwargs):
		super().__init__(**kwargs)
		self._conditions = None
		if variable is not None:
			self.set_variable(variable, *conditions)
		elif len(conditions):
			raise ValueError(f'Must specify variable if conditions are specified: {conditions}')


	def populate_defaults(self, **kwargs):
		self.include(
			generate_sentence,
			MarginalVerbalization().populate_default(**kwargs),
			ConditionalVerbalization().populate_default(**kwargs),
		)
		return self


	def set_variable(self, variable: VariableVerbalization, *conditions: VariableVerbalization):
		if self._conditions is not None:
			raise ValueError(f'A Variable has already been added to this verbalizer (create a new one)')
		conds = []
		for idx, condition in enumerate(conditions):
			gap = {gizmo: f'cond{idx}_{gizmo}' for gizmo in condition.gizmos()}
			cond = ConditionVerbalizer(condition, gap=gap, rng=self._rng)
			conds.append(cond)
		self._conditions = conds
		self.include(variable, *conds)
		return self


	@property
	def conditions(self):
		return self._conditions


	def identity_keys(self, gizmo: str = None):
		yield from super().identity_keys(gizmo=gizmo)
		if self._conditions is not None:
			for cond in self._conditions:
				yield from cond.identity_keys(gizmo=gizmo)


	# @tool('conditions')
	# def get_conditions(self):
	# 	if len(self.conditions) > 0:
	# 		return self.conditions



	# def _convert_verbalization_to_tools(self, info: dict[str, str | list[str]], label_fmt='{key}'):
	# 	tools = []
	# 	for key, options in info.items():
	# 		label = label_fmt.format(key=key)
	# 		if isinstance(options, list):
	# 			new = self._DecisionType([self._TemplateType(tmpl, label) for tmpl in options], name=label)
	# 		elif isinstance(options, str):
	# 			new = self._TemplateType(options, label)
	# 		else:
	# 			continue
	# 		tools.append(new)
	# 	return tools
	#
	#
	# def populate_variable_info(self, variable: dict[str, str | list[str]],
	# 						   conditions: list[dict[str, str | list[str]]] = None, verbalization=None, **static):
	# 	self.extend(self._convert_verbalization_to_tools(variable))
	# 	if conditions is not None:
	# 		assert len(conditions) > 0, 'Must have at least one condition'
	# 		for i, condition in enumerate(conditions):
	# 			key = '{key}'
	# 			self.extend(self._convert_verbalization_to_tools(condition, label_fmt=f'c{i}_{key}'))
	# 		self['num_conditions'] = len(conditions)
	#
	# 	if verbalization is None:
	# 		verbalization = MarginalVerbalization() if conditions is None else ConditionalVerbalization()
	# 	self.include(verbalization)
	# 	self.update(static)
	# 	return self



@tool.from_context('sentence')
def generate_sentence(ctx: Verbalizer):
	return Template.detok(ctx['conditional' if len(ctx.conditions) else 'marginal'],
						  capitalize=True, sentence=True)



class ConditionVerbalizer(Scope, VerbalizerBase):
	def identity_keys(self, gizmo=None):
		for ident in super().identity_keys(gizmo=gizmo):
			yield self.gizmo_to(ident)


	# def decisions(self):
	# 	for gadget in self._vendors():
	# 		if isinstance(gadget, AbstractDecision):
	# 			yield gadget

	# def identity_keys(self, gizmo: str = None):
	# 	for key in super().identity_keys(gizmo=gizmo):
	# 		yield self.gizmo_to(key)
	#
	#
	# def identity(self, gizmo: str = None):
	# 	identity = {key for key in self.identity_keys(gizmo=gizmo)}
	#
	# 	for key in self.identity_keys(gizmo=gizmo):
	#
	#
	# 	pass







