from ..imports import *
from .decisions import StaticTemplater, Decision, Verbalizer, TemplateDecision, FixedTemplate, AsSentence
from .formatting import NumberFormatter

from .. import misc


def _get_template_data():
	path = misc.assets_root() / 'templates.yml'
	assert path.exists(), f'path {path!r} does not exist'
	return load_yaml(path)



def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	defaults = [StaticTemplater(key, val) for key, vals in full['default-structure'].items() for val in vals]
	verb.include(*defaults, NumberFormatter())
	verb.include(Decision('prob_text', full['quantity']['prob_keys']))
	verb.include(TemplateDecision('ratio', full['proportion-structure']))
	verb.include(TemplateDecision('unit_text', full['unit-structure']))

	verb.include(TemplateDecision('claim', []
	                              # + full['frequency']['structure']
	                              # + full['quantity']['structure']
	                              # + full['measure']['structure']
	                              # + full['likelihood']['structure']
	                              # + full['estimation']['structure']
	                              # + full['status']['structure']
	                              # + full['population']['structure']
	                              + full['limits']['structure']
	                              # + full['precise']['structure']
	                              ))

	verb.include(Decision('freq_text', full['frequency']['options']))

	verb.include(Decision('quantity_text', full['quantity']['options']))

	verb.include(Decision('measure_text', full['measure']['options']))

	verb.include(Decision('likelihood_text', full['likelihood']['options']))

	verb.include(Decision('quantifier', full['estimation']['options']))

	verb.include(Decision('status_freq', full['status']['options']))

	verb.include(Decision('population_text', full['population']['options']))

	verb.include(Decision('limit_text', full['limits']['options']))
	verb.include(TemplateDecision('range_text', full['limits']['range_options']))

	verb.include(Decision('precise_text', full['precise']['options']))

	verb.include(FixedTemplate('sentence', '{claim}'))
	verb.include(AsSentence('sentence', capitalize=True, period=True))
	return verb




























