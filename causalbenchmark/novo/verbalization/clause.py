from omnibelt import pformat
from omniply import tool, ToolKit

from omniply.core.gadgets import SingleGadgetBase

from .base import VerbalizationBase
from .decision import Decision, choice
from ..templating import Template



class MarginalVerbalization(ToolKit, VerbalizationBase):
	_marginal_templates = [
		['there', 'is', 'a', '{quantity}', '{chance_word}', 'that', '{clause}'],
		['the', '{chance_word}', 'that', '{clause}', '{"are" if chance_word == "odds" else "is"}', '{quantity}'],
		['{quantity}', 'of', 'the', 'time', '{clause}'],
		['{preposition}', '{quantity}', 'of', '{domain},', '{subclause}'],
		['with a', '{confidence_word}', 'of {quantity}, {clause}']
	]
	_chance_words = ['probability', 'chance', 'likelihood', 'odds']
	_confidence_words = ['confidence', 'certainty']


	def populate_default(self, marginal_templates=None, clause_templates=None, **kwargs):
		if marginal_templates is None:
			marginal_templates = self._marginal_templates

		self.include(
			Decision([Template(tmpl, 'statement') for tmpl in marginal_templates],
					 'marginal', 'marginal_id'),

			Decision(self._chance_words, 'chance_word',
					 choice_validator=lambda ctx, option: option not in {'odds', 'likelihood'}
														  or ctx['marginal_id'] != 0),
			Decision(self._confidence_words, 'confidence_word'),
		)

		return self



class AntecedentVerbalization(Decision):
	_name = 'antecedent'

	# @choice.from_context
	def join_conditionals(self, ctx):
		terms = []
		for cond in ctx.conditions:
			terms.append(cond.grab_from(ctx, cond.gizmo_from('condition')))
			# terms.append(cond['condition'])
		return ' and '.join(terms)


	_clause_template = ['{cond_conjunction}', '{clauses}']

	@choice.from_context
	def join_clauses(self, ctx):
		terms = []
		for cond in ctx.conditions:
			terms.append(cond.grab_from(ctx, cond.gizmo_from('clause')))
			# terms.append(cond['clause'])
		clauses = ' and '.join(terms)
		return Template(self._clause_template).fill_in({'clauses':clauses}, ctx)


	# add other (smarter) options - check pronouns, order conditions, etc.
	pass



class ConditionalVerbalization(ToolKit):
	_cond_templates = [
		['{antecedent},', '{consequent}'],
		['{consequent},', '{antecedent}'],
	]

	def populate_default(self, cond_templates=None, antecedent_choices=None, **kwargs):
		cond_templates = cond_templates or self._cond_templates
		self.include(
			AntecedentVerbalization(antecedent_choices),
			Decision(['if', 'when', 'given that', 'because', 'provided that'], 'cond_conjunction'),
			Decision([Template(tmpl, 'conditional') for tmpl in cond_templates],'conditional'),
		)
		return self


	@tool('consequent')
	def get_consequent(self, marginal):
		return marginal







