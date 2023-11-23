from pathlib import Path
from itertools import product

from omnibelt import load_json, save_json
import omnifig as fig
from omniply import tool, ToolKit, Context, Scope, Selection, MissingGadget

import torch

from ..templating import FixedTemplate
from .. import misc
from .prompt_templates import setting_generation_prompt_template, graph_selection_prompt_template, stats_prompt_template



def build_prompter(*additional_kits):

	ctx = Context()



	ctx.include(*additional_kits)

	return ctx



def build_model(nodes, probs):
	variables = {}
	for node in nodes:
		if len(node['parents']):
			variables[node['name']] = ConditionalBernoulli([variables[parent] for parent in node['parents']])
		else:
			variables[node['name']] = Bernoulli(0.5)
	net = BernoulliNetwork(variables)
	return net



@fig.component('story')
class Story(Context, fig.Configurable):
	def __init__(self, story_root: Path = None, story_id: str = None, **kwargs):
		if story_root is None:
			story_root = Path('stories')
		super().__init__(**kwargs)
		self._root = story_root
		self.story_id = story_id
		if story_id is not None:
			self.load(story_id)


	def populate_defaults(self):
		story_template = FixedTemplate('setting', setting_generation_prompt_template)
		graph_template = FixedTemplate('graph', graph_selection_prompt_template)
		prob_template = StatisticsPrompting()
		self.include(story_template, graph_template, prob_template)
		return self


	def load(self, story_id: str):
		info = load_json(self._root / f'{story_id}.json')
		self.clear()
		self.update({name: val for name, val in info.items() if val is not None})


	def save(self, story_id: str = None, *, overwrite=False):
		if story_id is None and self.story_id is None:
			raise ValueError('No story ID provided')
		if story_id is None:
			story_id = self.story_id
		path = self._root / f'{story_id}.json'
		if path.exists() and not overwrite:
			raise FileExistsError(f'File {path} already exists (use overwrite=True to overwrite)')
		self._root.mkdir(exist_ok=True, parents=True)

		info = {name: val for name, val in self.items() if val is not None}

		# info = {
		# 	'seed': self['seed'],
		# 	'spark': self['spark'],
		#
		# 	'nodes': self['nodes'],
		#
		# 	'atoms': self.grab('atoms', None),
		# 	'probs': self.grab('probs', None),
		# }
		save_json(info, path)
		return path



@fig.component('stats-prompt')
class StatisticsPrompting(ToolKit, fig.Configurable):
	def __init__(self, prompt_template: str = None,
				 question_template=None, val_template=None, cond_template=None, parent_template=None,
				 desc_template=None, separator=None, **kwargs):
		if prompt_template is None:
			prompt_template = stats_prompt_template
		if question_template is None:
			question_template = '{index}. {parents}what is the probability that {outcome}?'
		if val_template is None:
			val_template = '"{name}" is "{val}" (rather than "{otherval}")'
		if parent_template is None:
			parent_template = '"{name}" is "{val}"'
		if cond_template is None:
			cond_template = 'when {parents}: '
		if desc_template is None:
			desc_template = '"{name}" means "{description}"'
		if separator is None:
			separator = ' and '
		super().__init__(**kwargs)
		self.include(FixedTemplate('stats', template=prompt_template))
		self.question_template = question_template
		self.val_template = val_template
		self.parent_template = parent_template
		self.cond_template = cond_template
		self.desc_template = desc_template
		self.separator = separator


	def generate_question_terms(self, nodes):
		varvals = {node['name']: node['values'] for node in nodes}

		index = 1
		for node in nodes:
			name = node['name']
			parents = node['parents']
			vals = node['values']

			for parentvals in product(*[varvals[parent] for parent in parents]):
				yield self.question_template.format(
					index=index,
					outcome=self.val_template.format(name=name, val=vals[1], otherval=vals[0]),
					parents=self.cond_template.format(parents=self.separator.join(
						[self.parent_template.format(name=parent, val=parentval) for parent, parentval in
						 zip(parents, parentvals)])) if len(parents) else ''
				)
				index += 1


	@tool('questions')
	def generate_questions(self, nodes):
		return '\n'.join(self.generate_question_terms(nodes))


	@tool('descriptions')
	def get_descriptions(self, nodes):
		return '\n'.join(self.desc_template.format(**node) for node in nodes)















