from typing import Any, Optional, Iterator

from omnibelt import filter_duplicates

from omniply import tool, ToolKit
from omniply.core.abstract import AbstractGadget, AbstractGaggle, AbstractGig

from omniply.core.gadgets import SingleGadgetBase

from .base import VerbalizationBase
from .decision import Decision, AbstractDecision
from ..templating import Template



class Question_Verbalizer(VerbalizationBase, ToolKit):
	def populate_verbalizers(self, varverbs: dict[str, 'VariableVerbalizer']):
		return self


	def add_verbalizers(self, treatment, outcome):
		gap = {gizmo: f'treatment_{gizmo}' for gizmo in treatment.gizmos()}
		self._treatment = treatment
		self._outcome = outcome
		self.include(treatment, outcome)
		return self


	@tool('answer')
	def get_answer(self, ground_truth: float):
		assert ground_truth != 0.5, f'Ill-defined ground truth: {ground_truth}'
		return 'yes' if ground_truth > 0.5 else 'no'



class InvertableSolver(Question_Verbalizer):
	@tool.from_context('invert')
	def get_invert(self, ctx):
		return ctx.select([False, True])


	@tool('answer')
	def get_adjusted_answer(self, answer: str, invert: bool):
		if invert:
			return 'yes' if answer == 'no' else 'no'
		return answer



class ATE_Sign_Verbalizer(Question_Verbalizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._treatment = None
		self._outcome = None


	@property
	def treatment(self):
		return self._treatment
	@property
	def outcome(self):
		return self._outcome


	def populate_verbalizers(self, varverbs: dict[str, 'VariableVerbalizer']):



		return self

	_influence_verbs = ['lead to', 'positively influence', 'cause', 'result in',]
	_invert = [0, 1]
	_question_templates = [
		['does', '{treatment_descriptor}', '{influence_verb}', '{outcome_descriptor}'],
		['if', '{action}', ',', 'does that {influence_verb}', '{outcome_descriptor}'],
	]
	def populate_defaults(self, question_templates=None, **kwargs):
		if question_templates is None:
			question_templates = self._question_templates
		self.include(
			Decision([Template(tmpl, 'question') for tmpl in question_templates], 'question'),
			Decision(self._influence_verbs, 'influence_verb'),
		)
		return self

	@tool.from_context('action')
	def get_action(self, ctx):
		if ctx['invert']:
			action = ctx['treatment_action0']
		else:
			action = ctx['treatment_action1']

		if action.startswith('if ') or action.startswith('when '):
			action = action.split(' ', 1)[1]
		return action












