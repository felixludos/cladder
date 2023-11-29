
from omniply import tool, ToolKit
from omniply.core.gadgets import SingleGadgetBase

from .base import VerbalizationBase
from .decision import Decision
from ..templating import Template



class ConditionDecision(Decision):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._conditions = []
		self._num_conditions = 0



class AntecedentOption(SingleGadgetBase):
	def __init__(self, gizmo=None, **kwargs):
		if gizmo is None:
			gizmo = 'antecedent'
		super().__init__(gizmo=gizmo, **kwargs)


	def grab_from(self, ctx: 'Verbalizer', gizmo=None):
		terms = []
		for i in range(ctx['num_conditions']):
			terms.append(ctx[f'c{i}_condition'])
		return terms



class AutoAntecedentOption(AntecedentOption):
	_conditional_word_options = ['if', 'when', 'given that', 'because', 'provided that']

	_fixed_template = ['{conditional_word}', '{position}']

	def __init__(self, gizmo=None, **kwargs):
		super().__init__(gizmo=gizmo, **kwargs)










