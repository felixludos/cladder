from typing import Any, Optional, Iterator

from omnibelt import filter_duplicates

from omniply import tool, ToolKit
from omniply.core.abstract import AbstractGadget, AbstractGaggle, AbstractGig

from omniply.core.gadgets import SingleGadgetBase

from .base import VerbalizationBase
from .decision import Decision, AbstractDecision
from ..templating import Template



class VariableVerbalization(ToolKit, VerbalizationBase):
	def __init__(self, atoms: dict[str, str | list[str]] = None, static: dict[str, Any] =None, **kwargs):
		'''
		`static` should be the node information (name, type, values, parents, etc.)
		`atoms` is the verbalization information (including choices for decisions).
		'''
		if static is None:
			static = {}
		super().__init__(**kwargs)
		self.static = static
		if atoms is not None:
			self.populate_verbalization_atoms(atoms)


	def gadgets(self, gizmo: Optional[str] = None) -> Iterator[AbstractGadget]:
		if gizmo is not None and gizmo in self.static:
			yield self
		else:
			yield from super().gadgets(gizmo)


	def gizmos(self):
		yield from filter_duplicates(super().gizmos(), self.static.keys())


	def grab_from(self, ctx: 'AbstractGig', gizmo: str) -> Any:
		if gizmo in self.static:
			return self.static[gizmo]
		return super().grab_from(ctx, gizmo)



	_clause_templates = [
		['{subject}', '{predicate}',],
		['{nounclause}',],
		['{subclause}',],
	]
	def populate_defaults(self, clause_templates=None, **kwargs):
		if clause_templates is None:
			clause_templates = self._clause_templates
		self.include(
			Decision([Template(tmpl, 'clause') for tmpl in clause_templates], 'clause'),
		)
		return self


	def populate_verbalization_atoms(self, atoms: dict[str, str | list[str]], **static: Any):
		tools = []
		for label, options in atoms.items():
			if isinstance(options, list):
				new = Decision([self._TemplateType(tmpl, label) for tmpl in options], name=label)
			elif isinstance(options, str):
				new = Template(options, label)
			else:
				continue
			tools.append(new)

		self.extend(tools)
		self.static.update(static)
		return self


	def populate_variable(self, node_info: dict, node_verbs: dict, variable_value=1):
		atoms = {**node_verbs, **node_verbs['values'][variable_value]}
		del atoms['values']
		return self.populate_verbalization_atoms(atoms, **node_info)

	@staticmethod
	def process_raw_verbalizations(story):
		base_keys = ['descriptor', 'subject', 'pronoun', 'preposition', 'domain']
		value_keys = ['predicate', 'nounclause', 'subclause', 'condition', 'action']
		verbs = []
		for node in story['nodes']:
			raw = story['verbs'][node['name']]
			info = {'values': {0: {}, 1: {}}}
			info.update({key: raw[key] for key in base_keys})
			for i in [0, 1]:
				info_val = info['values'][i]
				info_val.update({key: raw[f'{key}{i}'] for key in value_keys})
			# node['verbs'] = info
			verbs.append(info)
		story['verbs'] = verbs

# class ConditionVerbalization(ToolKit, AbstractDecision, VerbalizationBase):
# 	def __init__(self, conditions = None, identity_fmt='p{idx}_{key}', **kwargs):
# 		super().__init__(**kwargs)
# 		self.identity_fmt = identity_fmt
# 		self.conditions = []
# 		if conditions is not None:
# 			self.add_conditions(conditions)
#
#
# 	def identity(self):
# 		return self.identity_fmt.format(idx=self['idx'], key=self['key'])
#
#
# 	def add_conditions(self, *conditions: 'Verbalizer'):
# 		self.conditions.extend(conditions)
#
#
# 	_conditional_word_options = ['if', 'when', 'given that', 'because', 'provided that']
#
# 	_fixed_template = ['{conditional_word}', '{position}']
#
# 	def __init__(self, **kwargs):
# 		super().__init__(**kwargs)












