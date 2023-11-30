from ..imports import *
from .. import misc


from ..seeding import Story

from .variable import VariableVerbalization
from .top import Verbalizer
from .flavors import PrecisePercent


def story_atoms(story):
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



def test_load_story():
	print()
	path = misc.data_root() / 'stories'
	story_names = [p.stem for p in path.glob('*.json')]
	print(story_names)
	story_names = story_names[:1]
	assert len(story_names), f'no stories found in {path}'

	stories = []
	for name in story_names:
		story = Story(story_root=misc.data_root() / 'stories', story_id=name)
		stories.append(story)

	for story in stories:
		story_atoms(story)

	story = random.choice(stories)

	node_idx = random.choice(range(len(story['nodes'])))
	node = story['nodes'][node_idx]
	verbs = story['verbs'][node_idx]
	varval = 1
	print(node['name'], node['values'][varval])

	node['mean'] = 0.24

	varverb = VariableVerbalization().populate_defaults().populate_variable(
		node, verbs, varval)

	parent_idx = random.choice([i for i in range(len(story['nodes'])) if i != node_idx])
	parent_val = 1

	p1 = VariableVerbalization().populate_defaults().populate_variable(
		story['nodes'][parent_idx], story['verbs'][parent_idx], parent_val)

	parent2_idx = random.choice([i for i in range(len(story['nodes'])) if i not in {node_idx, parent_idx}])
	parent2_val = 0

	p2 = VariableVerbalization().populate_defaults().populate_variable(
		story['nodes'][parent2_idx], story['verbs'][parent2_idx], parent2_val)

	print(varverb)

	verbalizer = Verbalizer().populate_defaults().add_variable(varverb, p1, p2)
	verbalizer.include(PrecisePercent())

	print(verbalizer)

	text = verbalizer['sentence']
	print(text)

	identity = verbalizer.identity()
	print(identity)




























