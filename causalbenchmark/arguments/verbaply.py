
from .imports import *
from omniply.novo.test_novo import *



# class Template:
# 	pass


class Decision(AbstractDecision):
	def __init__(self, target: str, choices: Dict[str, Dict[str, Any]] = (), key_gizmo: str = None, **kwargs):
		if key_gizmo is None:
			key_gizmo = f'{target}_id'
		if isinstance(choices, (list, tuple)):
			choices = {str(i): {target: val} for i, val in enumerate(choices)}
		super().__init__(**kwargs)
		self._products = {key for info in choices.values() for key in info.keys()}
		assert key_gizmo not in self._products, f'key {key_gizmo} already in {self._products}'
		assert target in self._products, f'target {target} not in {self._products}'
		self._choices = choices
		self.key = key_gizmo
		self._target = target

	def gizmos(self) -> Iterator[str]:
		yield self.key
		yield from self._products

	def __len__(self):
		return len(self._choices)

	def choices(self, gizmo: str = None):
		if gizmo == self.key:
			yield from self._choices.keys()

	def choose(self, ctx: AbstractCrawler, gizmo: str):
		return ctx.select(self, gizmo)

	def grab_from(self, ctx: AbstractCrawler, gizmo: str) -> Any:
		if gizmo == self.key:
			return self.choose(ctx, gizmo)
		assert gizmo in self._products, f'gizmo {gizmo} not in {self._products}'
		choice = ctx[self.key]
		return self._choices[choice].get(gizmo)



# class TemplateSelector(Template):
# 	'''selects between (tools)'''
# 	pass


class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		keys = set(self._extract_template_keys(template))
		self.keys = keys
		self.template = template

	# def __init__(self, terms: Union[str, Iterable[str]] = None, delimiter=''):
	# 	if terms is None:
	# 		terms = []
	# 	elif isinstance(terms, str):
	# 		terms = terms.split(' ')
	# 		# terms = self._extract_template_keys(terms)
	# 	if isinstance(terms, Iterable):
	# 		terms = (key for term in terms for key in self._extract_template_keys(term))
	#
	# 	reqs = {}
	# 	keys = set()
	# 	pieces = []
	# 	for i, (is_key, val) in enumerate(terms):
	# 		if is_key:
	# 			reqs[i] = val
	# 			keys.add(val)
	# 		pieces.append(val)
	# 	self.reqs = reqs
	# 	self.keys = keys
	# 	self.terms = pieces
	# 	self.delimiter = delimiter


	@staticmethod
	def _extract_template_keys(template: str):
		for match in re.finditer(r'\{([^\}]+)\}', template):
			yield match.group(1)


	def fill_in(self, reqs: Dict[str, str]):
		return self.template.format(**reqs)
		# return self.delimiter.join(reqs[self.reqs[i]] if i in self.reqs else term
		# 						   for i, term in enumerate(self.terms)
		# 						   if (i not in self.reqs and term is not None)
		# 						   or (i in self.reqs and self.reqs[i] in reqs))


class TemplateDecision(Decision):
	_Templater = SimpleTemplater
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		for key, info in self._choices.items():
			raw = info.get(self._target)
			if isinstance(raw, str):
				info[self._target] = self._Templater(raw)

	def grab_from(self, ctx: AbstractCrawler, gizmo: str) -> Any:
		out = super().grab_from(ctx, gizmo)
		if isinstance(out, self._Templater):
			reqs = {key: ctx.grab_from(ctx, key) for key in out.keys}
			result = out.fill_in(reqs)
			return result
		return out



class StaticTemplater(AbstractTool, SimpleTemplater):
	def __init__(self, key: str, template: Union[str, Sequence[str]], **kwargs):
		super().__init__(template, **kwargs)
		self.key = key
		# self.template = template


	def gizmos(self) -> Iterator[str]:
		yield self.key


	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		out = self.fill_in(reqs)
		return out



class AsSentence(AbstractTool):
	def __init__(self, target: str, capitalize=True, period=True, source: str = None, **kwargs):
		if source is None:
			source = target # loopy
		super().__init__(**kwargs)
		self.source = source
		self.target = target
		self.capitalize = capitalize
		self.period = period

	def gizmos(self) -> Iterator[str]:
		yield self.target

	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		out = ctx[self.source]
		if self.capitalize:
			out = out[0].upper() + out[1:]
		if self.period:
			out += '.'
		return out


class Verbalization(SimpleFrame):
	def identity(self):
		# keys = {decision.key for decision in self._owner.decisions()}
		# return {key: self[key] for key in self.cached() if key in keys}
		return self._frame.copy()


class Verbalizer(SimpleCrawler, LoopyKit, MutableKit):
	_SubCrawler = Verbalization

	def __init__(self, *args, seed=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.rng = np.random.RandomState(seed)

	def identity(self):
		return self.current.identity()

	def decisions(self, gizmo: str = None) -> Iterator[Decision]:
		for vendor in self._vendors(gizmo):
			if isinstance(vendor, Decision):
				yield vendor

	def spawn(self, gizmo: Optional[str] = None, *, empty_value=None) -> Iterator[Any]:
		ctx = self.current
		for ctx in chain([ctx], self):
			if gizmo is not None:
				try:
					ctx.grab(gizmo)
				except ToolFailedError as e:
					ctx['error'] = e
					ctx[gizmo] = empty_value
				ctx['identity'] = ctx.identity()
			yield ctx

	def spawn_gizmo(self, gizmo: str, *, empty_value=None) -> Iterator[str]:
		for ctx in self.spawn(gizmo, empty_value=empty_value):
			if ctx[gizmo] is not empty_value:
				yield ctx[gizmo]


class NumberFormatter(CraftyKit):
	def __init__(self, lower_condition=0.01, upper_condition=0.99, **kwargs):
		super().__init__(**kwargs)
		self._process_crafts()
		self.upper_condition = upper_condition
		self.lower_condition = lower_condition

	def gizmos(self) -> Iterator[str]:
		yield from super().gizmos()

	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		return super().grab_from(ctx, gizmo)

	@tool.from_context('bound')
	def compute_bound(self, ctx):
		side = ctx['bound_side']
		other_side = 'upper_bound' if side == 'lower_bound' else 'lower_bound'
		val = ctx[f'{other_side}_value']
		if ((side == 'upper_bound' and val <= self.lower_condition)
				or (side == 'lower_bound' and val >= self.upper_condition)):
			return ctx[side]
		raise MissingGizmoError('bound')

	@tool('mean')
	def format_number(self, value):
		return f'{value:.0%}'

	@tool('lower_bound')
	def format_lower_bound(self, lower_bound_value):
		return f'{lower_bound_value:.0%}'

	@tool('lower_bound_100')
	def format_lower_bound_100(self, lower_bound_value):
		return f'{lower_bound_value*100:.0f}'

	@tool('upper_bound')
	def format_upper_bound(self, upper_bound_value):
		return f'{upper_bound_value:.0%}'

	@tool('implication')
	def infer_implication(self, lower_bound_value, upper_bound_value):
		return [lower_bound_value, upper_bound_value]



def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	defaults = [StaticTemplater(key, val) for key, vals in full['default-structure'].items() for val in vals]
	verb.include(*defaults, NumberFormatter())
	verb.include(Decision('prob_text', full['quantity']['prob_keys']))
	verb.include(TemplateDecision('ratio', full['proportion-structure']))
	verb.include(TemplateDecision('unit_text', full['unit-structure']))

	verb.include(TemplateDecision('claim', []
	                              + full['frequency']['structure']
	                              + full['quantity']['structure']
	                              + full['measure']['structure']
	                              + full['likelihood']['structure']
	                              + full['estimation']['structure']
	                              + full['status']['structure']
	                              + full['population']['structure']
	                              + full['limits']['structure']
	                              + full['precise']['structure']
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

	verb.include(AsSentence('claim', capitalize=True, period=True))
	return verb



def test_templater():
	seed = 0
	gen = default_vocabulary(seed=seed)
	ctx = gen.current

	ctx['subject'] = 'Bob'
	ctx['verb'] = 'eats'

	known = list(ctx.gizmos())
	assert 'freq_text' in known

	# out = ctx['freq_text']
	# impl = ctx['implication']

	claim = ctx['claim']
	assert 'Bob' in claim and 'eats' in claim

	id_info = ctx.identity()
	assert 'freq_text_id' in id_info and 'claim_id' in id_info



class InfoTool(AbstractTool):
	def __init__(self, info, **kwargs):
		super().__init__(**kwargs)
		self.info = info

	def gizmos(self) -> Iterator[str]:
		yield from self.info.keys()

	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		return self.info[gizmo]



def test_spawn_templates():
	seed = 0
	gen = default_vocabulary(seed=seed)

	gen.include(InfoTool({
		'subject': 'Bob',
		'verb': 'eats dinner', # verb + object
		# 'unit': 'days',

		# 'pronoun': 'he',
		# 'event': 'dinner', # it is _
		'value': 0.4,
		'lower_bound_value': 0.3,
		'upper_bound_value': 0.999,

		# 'group': 'people', # 20% of _
		# 'verb': 'eat dinner',
	}))

	entries = list(gen.spawn('claim'))
	ctx = gen.current

	results = [entry['claim'] for entry in entries]

	count = sum(r is not None for r in results)
	fraction = count / len(results)

	print(entries)




################# old


class StaticChoice(Decision):
	def __init__(self, name: str, data: Dict[str, Union[str, Dict[str, Any]]]):
		self._name = name
		self.data = data


	@property
	def name(self):
		return self._name


	def _parse_data(self, data):
		raise NotImplementedError


	@property
	def num_options(self):
		return len(self.data.keys())
	@property
	def options(self):
		yield from self.data.keys()


	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates([self.name], *[val for val in self.data.values() if isinstance(val, dict)])


	def _default_value(self, ctx, ID, gizmo):
		if gizmo in self.data:
			return self.data[gizmo]
		raise NotImplementedError


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		ID = ctx.identify(self)
		info = self.data[ID]
		if isinstance(info, str):
			return info
		if gizmo not in info:
			out = self._default_value(ctx, ID, gizmo)
		else:
			out = info[gizmo] # TODO: should probably be done by the verbalizer automatically
		if 'implication' not in info and 'implication' not in ctx:
			ctx['implication'] = info['implication']
		return out



class TemplateChoice(StaticChoice):
	def __init__(self, name: str, data: Dict[str, Union[str, Dict[str, Any]]]):
		super().__init__(name, data)
		self.templates = {key: StaticTemplater(self.name, val if isinstance(val, str) else val[self.name])
		                  for key, val in self.data.items()}


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo == self.name:
			return self.templates[ctx.identify(self)].grab_from(ctx, gizmo)
		return super().grab_from(ctx, gizmo)



class ConditionBuilder(TemplateChoice):
	def __init__(self, data=None):
		super().__init__('cond', data)


	def _attempt_grab(self, ctx, gizmo: str):
		try:
			return self.grab_from(ctx, gizmo)
		except MissingGizmoError:
			return None


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if not len(ctx.variable.given):
			raise ToolFailedError(f'no parents found: {ctx.variable}')

		assert len(ctx.variable.given), f'no conditioning found: {ctx.variable}'

		base_subject = self._attempt_grab(ctx, 'subject')
		base_pronoun = self._attempt_grab(ctx, 'pronoun')

		candidates = [ctx.condition(parent) for parent in ctx.variable.given]

		cond_subjects = [pctx for pctx in candidates if self._attempt_grab(pctx, 'cond-subject') is not None]
		if len(cond_subjects):# and ctx.gen.rand() < 0.5:
			pick = ctx.gen.choice(cond_subjects)
			ctx['subject'] = pick['cond-subject']
			candidates.remove(pick)

		subjects = {}

		for pctx in candidates:
			subject = self._attempt_grab(pctx, 'subject')
			if base_pronoun is not None and base_subject == subject:
				pctx['subject'] = base_pronoun
			subjects.setdefault(subject, []).append(pctx)

		for subject, terms in subjects.items():
			if len(terms) > 1:
				ctx.gen.shuffle(terms)
			subjects[subject] = [terms[0]['head']] + [term['verb'] for term in terms[1:]]

		order = list(subjects.keys())
		ctx.gen.shuffle(order)

		clauses = [term for key in order for term in subjects[key]]

		return util.verbalize_list(clauses)



class ClaimChoice(Decision):
	def __init__(self, claim: Optional[Iterator['Atom']] = None):
		super().__init__()
		self.claims = {}
		if claim is not None:
			for c in claim:
				self.register_claim(c)


	def decisions(self) -> Iterator['Decision']:
		yield from self.claims.values()


	def gizmos(self) -> Iterator[str]:
		yield 'claim'


	@property
	def name(self):
		return 'claim'


	@property
	def num_options(self):
		return len(self.claims)


	@property
	def options(self):
		yield from self.claims.keys()


	def register_claim(self, template: Union[str, 'Atom'], name: Optional[str] = None):
		if name is None:
			assert isinstance(template, Decision), f'must provide name if template is not a Decision: {template}'
			name = template.name
		if not isinstance(template, 'Atom'):
			assert isinstance(template, str), f'template must be a string or Atom: {template}'
			template = StaticTemplater(name, template)
		if name in self.claims:
			raise ValueError(f'already have a sentence type named {name}')
		self.claims[name] = template
		return template


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		ID = ctx.identify(self)
		return self.claims[ID].grab_from(ctx, ID)



class Sourced:
	def __init__(self, data=None):
		if data is None:
			data = {}
		self.data = data



class Story(Sourced):
	@staticmethod
	def parse_term(term):
		var, *given = term.split('|')
		parents = None
		if len(given):
			parents = {k: str(v) for k, v in [x.split('=') for x in given[0].split(',')]}

		value = None
		if '=' in var:
			var, value = var.split('=')
			value = int(value)

		return var, value, parents


	@classmethod
	def set_term_value(cls, term, value=None, parents=None):
		var, old_value, old_parents = cls.parse_term(term)

		if parents is None:
			parents = old_parents
		parents = ','.join(f'{k}={v}' for k, v in parents.items()) if parents is not None else ''

		head = f'{var}={value}' if value is not None else var
		tail = f'|{parents}' if len(parents) else ''

		return head + tail


	def variable(self, name: str, value=None):
		return Variable(self.data['variables'][name], value=value)


	def term(self, term: str):
		var, value, parents = self.parse_term(term)
		if parents is None:
			parents = {}
		conditions = {k: self.variable(k, value=v) for k, v in parents.items()}
		return StatisticalTerm(self.data['variables'][var], conditions=conditions, value=value)



class Variable(Sourced, AbstractTool):
	def __init__(self, data, value=None):
		super().__init__(data)
		if value is None:
			value = self.data.get('default', '1')
		self.categories = self.data.get('categories', [str(v) for v in self.data.get('values', {}).keys()
		                                               if not str(v).startswith('~')])
		self.value = self.parse_variable_value(value)
		values = {str(k): v for k, v in self.data.get('values', {}).items()}
		self.claim = values.get(self.value, {})


	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates(self.claim.keys(),
									 (key for key in self.data.keys() if key != 'values'))


	def parse_variable_value(self, value):
		if isinstance(value, int):
			value = str(value)
		if value.startswith('~'):
			value = tuple(sorted([c for c in self.categories if c != value[1:]]))
		return value


	@property
	def N(self):
		return len(self.categories)


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo in self.claim:
			return self.claim[gizmo]
		if gizmo in self.data:
			return self.data[gizmo]
		raise ToolFailedError(gizmo)



class StatisticalTerm(Variable):
	marginal_style = None
	conditional_style = None

	def __init__(self, data, *, conditions=None, **kwargs):
		super().__init__(data, **kwargs)
		self.given = conditions or {}



# class SentenceTemplate(CapitalizedTemplater, StaticTemplater):
# 	pass
# class SentenceChoice(CapitalizedTemplater, TemplateChoice):
# 	pass



# def default_vocabulary(seed=None):
# 	verb = Verbalizer(seed=seed)
#
# 	full = _get_template_data()
#
# 	# StatisticalTerm.marginal_style = MarginalStyle()
# 	# StatisticalTerm.conditional_style = ConditionalStyle()
#
# 	verb.include(SentenceTemplate('sentence', '{claim}.'))
# 	verb.include(SentenceChoice('sentence', full['conditional-structure']))
#
# 	term_defaults = full['default-structure']
# 	terms = [StaticTemplater(key, val) for key, val_options in term_defaults.items() for val in val_options]
# 	verb.include(*terms)
#
# 	freq_text = TemplateChoice('freq_text', full['frequency']['options'])
# 	verb.include(freq_text)
#
# 	event = TemplateChoice('event_text', full['event-structure'])
# 	verb.include(event)
#
# 	builders = ClaimChoice()
# 	verb.include(builders)
#
# 	freq = TemplateChoice('freq', {str(i): {'freq': code} for i, code in enumerate(full['frequency']['structure'])})
# 	builders.register_claim(freq)
#
# 	return verb
#
#
def _get_template_data():
	path = util.assets_root() / 'templates.yml'
	assert path.exists(), f'path {path!r} does not exist'
	return load_yaml(path)
#
#
# def test_templater():
# 	seed = 0
# 	ctx = default_vocabulary(seed=seed)
#
# 	known = list(ctx.gizmos())
#
# 	out = ctx['freq_text']
# 	impl = ctx['implication']
# 	id_info = ctx.identity
#
# 	print(out)


def test_story():
	seed = 0
	ctx = default_vocabulary(seed=seed)

	story_data = load_yaml(util.assets_root() / 'arguments' / 'contagion.yml')
	story = Story(story_data)

	term = 'Y|U=0,X=0'
	term = 'Y'
	verb = story.term(term)
	ctx.include(verb)

	picks = list(ctx.spawn_identities())

	solutions = list(ctx.spawn('sentence'))

	# out = ctx['freq_text']

	text = ctx['sentence']

	_verb_str = str(ctx)


	print(out)

	pass

