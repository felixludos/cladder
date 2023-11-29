from pathlib import Path
from itertools import product
import networkx as nx

from omnibelt import load_json, save_json, pformat, pformat_vars
import omnifig as fig
from omniply import tool, ToolKit, Context, Scope, Selection, MissingGadget

import torch

from ..templating import FixedTemplate, FileTemplate, LoadedTemplate
from .. import misc
# from .prompt_templates import (default_story_prompt_template, default_graph_prompt_template,
# 							   default_stats_prompt_template, default_verb_prompt_template,
# 							   default_question_prompt_template)


# def build_model(nodes, probs):
# 	variables = {}
# 	for node in nodes:
# 		if len(node['parents']):
# 			variables[node['name']] = ConditionalBernoulli([variables[parent] for parent in node['parents']])
# 		else:
# 			variables[node['name']] = Bernoulli(0.5)
# 	net = BernoulliNetwork(variables)
# 	return net



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


	def populate_defaults(self, motivation_prompt_template=None, graph_prompt_template=None,
						  stats_prompt_template=None, verb_prompt_template=None, questions_prompt_template=None):
		motivation_template = LoadedTemplate('motivation', 'prompt_motivation') \
			if motivation_prompt_template is None \
			else FixedTemplate(motivation_prompt_template, 'prompt_motivation')
		# story_template = LoadedTemplate('story', 'prompt_story') if story_prompt_tempalte is None \
		# 	else FixedTemplate(story_prompt_tempalte, 'prompt_story')
		graph_template = LoadedTemplate('graph', 'prompt_graph') if graph_prompt_template is None \
			else FixedTemplate(graph_prompt_template, 'prompt_graph')
		structure_template = LoadedTemplate('structure', 'prompt_structure')
		prob_template = StatisticsPrompting(stats_prompt_template)
		verb_template = VerbalizationPrompting(verb_prompt_template, questions_prompt_template)
		self.include(motivation_template, graph_template, prob_template, verb_template, structure_template,
					 GraphInfo())
		return self


	def load(self, story_id: str):
		info = load_json(self._root / f'{story_id}.json')
		self.clear()
		self.update({name: val for name, val in info.items() if val is not None})


	def save(self, story_id: str = None, *, overwrite=False, additional_keys=(), allow_missing=False):
		save_keys = ['seed', 'spark', 'motivation', 'nodes', 'stats', 'structure',
					 'verbs', 'queries', 'questions', *additional_keys]

		if story_id is None and self.story_id is None:
			raise ValueError('No story ID provided')
		if story_id is None:
			story_id = self.story_id
		path = self._root / f'{story_id}.json'
		if path.exists() and not overwrite:
			raise FileExistsError(f'File {path} already exists (use overwrite=True to overwrite)')
		self._root.mkdir(exist_ok=True, parents=True)

		info = {key: self.grab(key, None) for key in save_keys}
		if not allow_missing and any(val is None for val in info.values()):
			raise ValueError(f'Not all keys are present in story: '
							 f'{[key for key, val in info.items() if val is None]}')

		save_json(info, path)
		self.story_id = story_id
		return path



class GraphInfo(ToolKit):
	@tool('node_dict')
	def get_node_dict(self, nodes):
		return {node['name']: node for node in nodes}


	@tool('graph')
	@staticmethod
	def get_graph(nodes):
		G = nx.DiGraph()
		for node in nodes:
			G.add_node(node['name'], type=node['type'], observed=node['observed'])
			for parent in node['parents']:
				G.add_edge(parent, node['name'])
		return G

	@tool('treatments')
	@staticmethod
	def get_treatments(nodes):
		tr = [node for node in nodes if node['type'] == 'treatment']
		assert len(tr) == 2
		return tr

	@tool('outcome')
	@staticmethod
	def get_outcome(nodes):
		o = [node for node in nodes if node['type'] == 'outcome']
		assert len(o) == 1
		return o[0]

	@tool('confounders')
	@staticmethod
	def get_confounders(nodes):
		c = [node for node in nodes if node['type'] == 'confounder']
		return c

	@tool('mediators')
	@staticmethod
	def get_mediators(nodes):
		m = [node for node in nodes if node['type'] == 'mediator']
		return m

	@tool('colliders')
	@staticmethod
	def get_colliders(nodes):
		c = [node for node in nodes if node['type'] == 'collider']
		return c



@fig.component('stats-prompt')
class StatisticsPrompting(ToolKit, fig.Configurable):
	def __init__(self, prompt_template: str = None,
				 question_template=None, val_template=None, cond_template=None, parent_template=None,
				 desc_template=None, separator=None, **kwargs):
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
		self.include(LoadedTemplate('stats', 'prompt_stats') if prompt_template is None
					 else FixedTemplate(prompt_template, 'prompt_stats'))
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


	@tool('prob_questions')
	def generate_questions(self, nodes):
		return '\n'.join(self.generate_question_terms(nodes))


	@tool('descriptions')
	def get_descriptions(self, nodes):
		return '\n'.join(self.desc_template.format(**node) for node in nodes)



@fig.component('verb-prompt')
class VerbalizationPrompting(ToolKit, fig.Configurable):
	def __init__(self, verbalization_prompt_template: str = None, question_prompt_template=None, *,
				 rng=None, **kwargs):
		if rng is None:
			rng = misc.get_rng(rng)
		super().__init__(**kwargs)
		self.rng = rng
		self.include(LoadedTemplate('verbs', 'prompt_verbs')
					 if verbalization_prompt_template is None
					 else FixedTemplate(verbalization_prompt_template, 'prompt_verbs'),

					 LoadedTemplate('questions', 'prompt_questions')
					 if question_prompt_template is None
					 else FixedTemplate(question_prompt_template, 'prompt_questions'))


	# _variable_description_template = 'Variable {name!r} (0={values[0]!r}, 1={values[1]!r}) means {description}'
	_variable_description_template = 'Variable {name!r} (0={values[0]!r}, 1={values[1]!r}) means {description}'
	@tool('variable_description')
	def get_verbalization_info(self, nodes):
		return '\n'.join([pformat(self._variable_description_template, node) for node in nodes])


	def _select(self, choices: list):
		'''use self.rng (torch.Generator) to choose an element'''
		return choices[torch.randint(0, len(choices), (1,), generator=self.rng).item()]


	@tool('queries')
	def generate_queries(self, treatments, confounders):
		treatments = [t['name'] for t in treatments]
		confounders = [c['name'] for c in confounders]

		queries = []

		for treatment in treatments:
			queries.append({'treatment': treatment, 'query': 'ate', 'type': 'ate-sign',
							'criterion': self._select(['>', '<'])})

		t1, t2 = treatments if self._select([False, True]) else treatments[::-1]
		queries.append({'treatment1': t1, 'treatment2': t2, 'query': 'ate', 'type': 'ate-compare',
						'criterion': self._select(['>', '<'])})

		t1, t2 = treatments if self._select([False, True]) else treatments[::-1]
		queries.append({'treatment1': t1, 'treatment2': t2, 'query': 'ate', 'type': 'ate-compare-mag',
						'criterion': self._select(['>', '<'])})

		for confounder in confounders:
			for confounder_value in [0, 1]:
				for treatment in treatments:
					queries.append({'treatment': treatment,
									'confounder': confounder, 'confounder_value': confounder_value,
									'query': 'cate', 'type': 'cate-sign', 'criterion': self._select(['>', '<'])})

				t1, t2 = treatments if self._select([False, True]) else treatments[::-1]
				queries.append({'treatment1': t1, 'treatment2': t2,
								'confounder': confounder, 'confounder_value': confounder_value,
								'query': 'cate', 'type': 'cate-compare-c', 'criterion': self._select(['>', '<'])})

				t1, t2 = treatments if self._select([False, True]) else treatments[::-1]
				queries.append({'treatment1': t1, 'treatment2': t2,
								'confounder': confounder, 'confounder_value': confounder_value,
								'query': 'cate', 'type': 'cate-compare-c-mag', 'criterion': self._select(['>', '<'])})

			for treatment in treatments:
				queries.append({'treatment': treatment, 'confounder': confounder,
								'confounder_order': self._select([[0, 1], [1, 0]]),
								'query': 'cate', 'type': 'cate-compare-t', 'criterion': self._select(['>', '<'])})

				queries.append({'treatment': treatment, 'confounder': confounder,
								'confounder_order': self._select([[0, 1], [1, 0]]),
								'query': 'cate', 'type': 'cate-compare-t-mag', 'criterion': self._select(['>', '<'])})

		return queries


	_query_description_templates = {
		'ate-sign': 'ATE({treatment!r}) {criterion} 0',
		'ate-compare': 'ATE({treatment1!r}) {criterion} ATE({treatment2!r})',
		'ate-compare-mag': '|ATE({treatment1!r})| {criterion} |ATE({treatment2!r})|',

		'cate-sign': 'CATE({treatment!r} | {confounder!r} = {confounder_value}) {criterion} 0',
		'cate-compare-c': 'CATE({treatment1!r} | {confounder!r} = {confounder_value}) {criterion} '
						  'CATE({treatment2!r} | {confounder!r} = {confounder_value})',
		'cate-compare-c-mag': '|CATE({treatment1!r} | {confounder!r} = {confounder_value})| {criterion} '
							  '|CATE({treatment2!r} | {confounder!r} = {confounder_value})|',
		'cate-compare-t': 'CATE({treatment!r} | {confounder!r} = {confounder_order[0]}) {criterion} '
						  'CATE({treatment!r} | {confounder!r} = {confounder_order[1]})',
		'cate-compare-t-mag': '|CATE({treatment!r} | {confounder!r} = {confounder_order[0]})| {criterion} '
							  '|CATE({treatment!r} | {confounder!r} = {confounder_order[1]})|',
	}
	@tool('query_description')
	def get_query_descriptions(self, queries):
		lines = [pformat(self._query_description_templates[query['type']], query) for query in queries]
		desc = '\t' + '\n\t'.join(f'{i + 1}. {q}' for i, q in enumerate(lines))
		return desc


