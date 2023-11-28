
from omniply import tool, ToolKit

from .base import Verbalization
from .decision import Decision
from ..templating import Template


class NumberFormatting(ToolKit):
	@tool('quantity')
	def format_probability(self, mean: float):
		return f'{mean * 100:.0f}%'



class ClauseVerbalization(Verbalization):
	_DecisionType = Decision
	_TemplateType = Template


	def populate_variable_info(self, variable: dict[str, str | list[str]], **static):
		tools = []
		for key, options in variable.items():
			if isinstance(options, list):
				new = self._DecisionType([self._TemplateType(tmpl, key) for tmpl in options], name=key)
			elif isinstance(options, str):
				new = self._TemplateType(options, key)
			else:
				continue
			tools.append(new)
		self.extend(tools)
		self.update(static)
		return self


	_precise_templates = [
		['there', 'is', 'a', '{quantity}', '{prob_word}', 'that', '{position}'],
		['the', '{chance_word}', 'that', '{position}', '{"are" if chance_word == "odds" else "is"}', '{quantity}'],
		['{quantity}', 'of', 'the', 'time', '{position}'],
		['{quantity}', 'of', '{domain}', '{descriptor}'],
		['with a', '{confidence_word}', 'of {quantity}, {position}']
	]
	_position_templates = [
		['{subject}', '{verb}' ,],
		['{domain}', '{descriptor}'],
		['{phrase}' ,],
		['{event}' ,],
	]
	_prob_words = ['probability', 'chance', 'likelihood']
	_chance_words = ['probability', 'chance', 'likelihood', 'odds']
	_confidence_words = ['confidence', 'certainty']


	def populate_default(self, precise_templates=None, position_templates=None, **kwargs):
		if precise_templates is None:
			precise_templates = self._precise_templates
		if position_templates is None:
			position_templates = self._position_templates

		self.include(
			Decision([self._TemplateType(tmpl, 'clause') for tmpl in precise_templates], 'clause'),
			Decision([self._TemplateType(tmpl, 'position') for tmpl in position_templates], 'position'),
			Decision(self._prob_words, 'prob_word'),
			Decision(self._chance_words, 'chance_word'),
			Decision(self._confidence_words, 'confidence_word'),
			NumberFormatting(),
		)

		return self









