
from .imports import *
from omniply.novo.test_novo import *



class Template:
	pass


class Decision(AbstractDecision):
	def __init__(self, gizmo: str, choices: Dict[str, Dict[str, Any]] = (),
	             default_value_key: str = 'value', **kwargs):
		if isinstance(choices, (list, tuple)):
			choices = {str(i): {default_value_key: val} for i, val in enumerate(choices)}
		super().__init__(**kwargs)
		self._products = {key for info in choices.values() for key in info.keys()}
		assert gizmo not in self._products, f'gizmo {gizmo} already in {self._products}'
		self._choices = choices
		self._gizmo = gizmo

	def gizmos(self) -> Iterator[str]:
		yield self._gizmo
		yield from self._products

	def __len__(self):
		return len(self._choices)

	def choices(self, gizmo: str = None):
		if gizmo == self._gizmo:
			yield from self._choices.keys()

	def choose(self, ctx: AbstractCrawler, gizmo: str):
		return ctx.select(self, gizmo)

	def grab_from(self, ctx: AbstractCrawler, gizmo: str) -> Any:
		if gizmo == self._gizmo:
			return self.choose(ctx, gizmo)
		assert gizmo in self._products, f'gizmo {gizmo} not in {self._products}'
		return self._choices[ctx[self._gizmo]].get(gizmo)



class TemplateSelector(Template):
	'''selects between (tools)'''
	pass


class SimpleTemplater:
	def __init__(self, terms: Union[str, Iterable[str]] = None, delimiter=''):
		if terms is None:
			terms = []
		elif isinstance(terms, str):
			terms = terms.split(' ')
			# terms = self._extract_template_keys(terms)
		if isinstance(terms, Iterable):
			terms = (key for term in terms for key in self._extract_template_keys(term))

		reqs = {}
		keys = set()
		pieces = []
		for i, (is_key, val) in enumerate(terms):
			if is_key:
				reqs[i] = val
				keys.add(val)
			pieces.append(val)
		self.reqs = reqs
		self.keys = keys
		self.terms = pieces
		self.delimiter = delimiter


	@staticmethod
	def _extract_template_keys(template: str):
		for match in re.finditer(r'\{([^\}]+)\}', template):
			if match.start() > 0:
				yield False, template[:match.start()]
			yield True, match.group(1)
			template = template[match.end():]
		if len(template):
			yield False, template


	def fill_in(self, reqs: Dict[str, str]):
		return self.delimiter.join(reqs[self.reqs[i]] if i in self.reqs else term
								   for i, term in enumerate(self.terms)
								   if (i not in self.reqs and term is not None)
								   or (i in self.reqs and self.reqs[i] in reqs))



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



class CapitalizedTemplater(AbstractTool):
	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		out = super().grab_from(ctx, gizmo)
		return out[0].upper() + out[1:]



class Verbalization(SimpleFrame):
	def identity(self):
		keys = {decision.key for decision in self._owner.decisions()}
		return {key: self[key] for key in self.cached() if key in keys}



class Verbalizer(SimpleCrawler):
	_SubCrawler = Verbalization

	def __init__(self, *args, seed=None, **kwargs):
		super().__init__(*args, **kwargs)
		# self.rng = np.random.RandomState(seed)

	def decisions(self, gizmo: str = None) -> Iterator[Decision]:
		for vendor in self._vendors(gizmo):
			if isinstance(vendor, Decision):
				yield vendor



def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	default = Decision('default', full['default-structure'])

	claim = Decision('claim', full['frequency']['structure'])
	verb.include(claim)

	decision = Decision('freq_text', full['frequency']['options'])
	verb.include(claim)



	# StatisticalTerm.marginal_style = MarginalStyle()
	# StatisticalTerm.conditional_style = ConditionalStyle()

	verb.include(SentenceTemplate('sentence', '{claim}.'))
	verb.include(SentenceChoice('sentence', full['conditional-structure']))

	term_defaults = full['default-structure']
	terms = [StaticTemplater(key, val) for key, val_options in term_defaults.items() for val in val_options]
	verb.include(*terms)

	freq_text = TemplateChoice('freq_text', full['frequency']['options'])
	verb.include(freq_text)

	event = TemplateChoice('event_text', full['event-structure'])
	verb.include(event)

	builders = ClaimChoice()
	verb.include(builders)

	freq = TemplateChoice('freq', {str(i): {'freq': code} for i, code in enumerate(full['frequency']['structure'])})
	builders.register_claim(freq)

	return verb


def test_templater():
	seed = 0
	ctx = default_vocabulary(seed=seed)

	known = list(ctx.gizmos())

	out = ctx['freq_text']
	impl = ctx['implication']
	id_info = ctx.identity

	print(out)




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



class SentenceTemplate(CapitalizedTemplater, StaticTemplater):
	pass
class SentenceChoice(CapitalizedTemplater, TemplateChoice):
	pass



def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	# StatisticalTerm.marginal_style = MarginalStyle()
	# StatisticalTerm.conditional_style = ConditionalStyle()

	verb.include(SentenceTemplate('sentence', '{claim}.'))
	verb.include(SentenceChoice('sentence', full['conditional-structure']))

	term_defaults = full['default-structure']
	terms = [StaticTemplater(key, val) for key, val_options in term_defaults.items() for val in val_options]
	verb.include(*terms)

	freq_text = TemplateChoice('freq_text', full['frequency']['options'])
	verb.include(freq_text)

	event = TemplateChoice('event_text', full['event-structure'])
	verb.include(event)

	builders = ClaimChoice()
	verb.include(builders)

	freq = TemplateChoice('freq', {str(i): {'freq': code} for i, code in enumerate(full['frequency']['structure'])})
	builders.register_claim(freq)

	return verb


def _get_template_data():
	path = util.assets_root() / 'templates.yml'
	assert path.exists(), f'path {path!r} does not exist'
	return load_yaml(path)


def test_templater():
	seed = 0
	ctx = default_vocabulary(seed=seed)

	known = list(ctx.gizmos())

	out = ctx['freq_text']
	impl = ctx['implication']
	id_info = ctx.identity

	print(out)


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

