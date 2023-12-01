
from omniply import tool, ToolKit

from omniply.core.gadgets import SingleGadgetBase

from .base import VerbalizationBase
from .decision import Decision
from ..templating import Template


class NumberFormatting(ToolKit):
	@tool('quantity')
	def format_probability(self, mean: float):
		return f'{mean * 100:.0f}%'



class MarginalVerbalization(ToolKit, VerbalizationBase):
	_precise_templates = [
		['there', 'is', 'a', '{quantity}', '{prob_word}', 'that', '{position}'],
		['the', '{chance_word}', 'that', '{position}', '{"are" if chance_word == "odds" else "is"}', '{quantity}'],
		['{quantity}', 'of', 'the', 'time', '{position}'],
		['{preposition}', '{quantity}', 'of', '{domain},', '{subclause}'],
		['with a', '{confidence_word}', 'of {quantity}, {position}']
	]
	_position_templates = [
		['{subject}', '{predicate}',],
		['{nounclause}',],
		['{subclause}',],
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
			self._DecisionType([self._TemplateType(tmpl, 'clause') for tmpl in precise_templates], 'clause'),
			self._DecisionType([self._TemplateType(tmpl, 'position') for tmpl in position_templates], 'position'),
			self._DecisionType(self._prob_words, 'prob_word'),
			self._DecisionType(self._chance_words, 'chance_word'),
			self._DecisionType(self._confidence_words, 'confidence_word'),
			NumberFormatting(),
		)

		return self


	@tool('sentence')
	def verbalize(self, clause: str):
		return Template.detok(clause, capitalize=True, sentence=True)



class ConditionalVerbalization(MarginalVerbalization):
	_cond_templates = [
		['{antecedent},', '{consequent}'],
		['{consequent},', '{antecedent}'],
	]

	def populate_default(self, cond_templates=None, **kwargs):
		if cond_templates is None:
			cond_templates = self._cond_templates
		super().populate_default(**kwargs)
		self.include(
			self._DecisionType([self._TemplateType(tmpl, 'conditional') for tmpl in cond_templates], 'conditional'),
		)
		return self

	@tool.from_context('antecedent')
	def get_antecedent(self, ctx):
		terms = []

		# maybe cleverly sort conditions or check pronouns or merge subjects etc.

		for i in range(ctx['num_conditions']):
			terms.append(ctx[f'c{i}_condition'])
		return ' and '.join(terms)


	@tool('consequent')
	def get_consequent(self, clause):
		return clause


	@tool('sentence')
	def verbalize(self, conditional: str):
		return Template.detok(conditional, capitalize=True, sentence=True)







