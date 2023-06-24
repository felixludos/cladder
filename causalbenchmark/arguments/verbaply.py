
from .imports import *
from omniply.novo.test_novo import *



class Template:
	pass


class Atom(AbstractTool):
	'''selects between static options'''
	def possibilities(self):
		raise NotImplementedError



class Decision(Atom):
	@property
	def name(self):
		raise NotImplementedError
	@property
	def num_options(self):
		raise NotImplementedError
	@property
	def options(self):
		raise NotImplementedError
	def random_choice(self, gen: np.random.RandomState):
		return gen.choice(list(self.options))



class TemplateSelector(Template):
	'''selects between (tools)'''
	pass


class SimpleTemplater:
	def __init__(self, terms: Iterator[str] = None, delimiter=' '):
		if terms is None:
			terms = []
		elif isinstance(terms, str):
			terms = self._extract_template_keys(terms)
		elif isinstance(terms, Iterator):
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



class StaticTemplater(Atom, SimpleTemplater):
	def __init__(self, key: str, template: Union[str, Sequence[str]]):
		self.key = key
		self.template = template


	def gizmos(self) -> Iterator[str]:
		yield self.key


	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		out = self.fill_in(reqs)
		return out



class CapitalizedTemplater(Atom):
	def grab_from(self, ctx: Optional['AbstractContext'], gizmo: str) -> Any:
		out = super().grab_from(ctx, gizmo)
		return out[0].upper() + out[1:]



class Verbalizer(Cached, Context, LoopyKit, MutableKit, Decision):
	def __init__(self, *args, identity=None, seed=None, **kwargs):
		super().__init__(*args, **kwargs)
		if identity is None:
			identity = {}
		self.identity = identity
		# self._sentences = []
		self.gen = np.random.RandomState(seed)
		self._variable = None
		self._base = None


	@property
	def variable(self):
		return self._variable
	@variable.setter
	def variable(self, var: 'Variable'):
		self._variable = var


	@property
	def base(self):
		return self._base
	@base.setter
	def base(self, base: 'Verbalizer'):
		self._base = base


	def condition(self, parent: 'Variable') -> 'Verbalizer':
		new = self.copy()
		new.variable = parent
		new.base = self
		return new


	def _vendors(self, gizmo: Optional[str] = None) -> Iterator[AbstractTool]:
		if self.variable is not None:
			yield from self.variable.vendors(gizmo)
		yield from super()._vendors(gizmo)


	# @property
	# def num_options(self):
	# 	return len(self._sentences)
	#
	#
	# @property
	# def options(self):
	# 	yield from self._sentences
	#
	#
	# def random_choice(self, gen: np.random.RandomState = None):
	# 	if gen is None:
	# 		gen = self.gen
	# 	return super().random_choice(gen)


	def identify(self, decision: Decision):
		name = decision.name
		if name not in self.identity:
			self.identity[name] = decision.random_choice(self.gen)
		return self.identity[name]


	# def include(self, *tools: AbstractTool) -> 'MutableKit':
	# 	super().include(*tools)
	# 	for t in tools:
	# 		if isinstance(t, SentenceBuilder):
	# 			self._sentences.append(t)
	# 	return self

	pass



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
		assert len(ctx.variable.given), f'no conditioning found: {ctx.variable}'

		keys = ['head', 'cond-subject', 'phrase', 'subject', 'verb']
		parent_ctxs = [ctx.condition(parent) for parent in ctx.variable.given]
		terms = [tuple(self._attempt_grab(pctx, key) for key in keys) for pctx in parent_ctxs]

		if len(terms):
			pass

		pass



class SentenceChoice(Decision):
	def __init__(self, sentences: Optional[Iterator[Atom]] = None):
		super().__init__()
		self.sentences = {}
		if sentences is not None:
			for s in sentences:
				self.register_sentence(s)


	def gizmos(self) -> Iterator[str]:
		yield 'sentence'


	@property
	def name(self):
		return 'sentence'


	@property
	def num_options(self):
		return len(self.sentences)


	@property
	def options(self):
		yield from self.sentences.keys()


	def register_sentence(self, template: str, name: Optional[str] = None):
		if name is None:
			assert isinstance(template, Decision), f'must provide name if template is not a Decision: {template}'
			name = template.name
		if not isinstance(template, Atom):
			assert isinstance(template, str), f'template must be a string or Atom: {template}'
			template = StaticTemplater(name, template)
		if name in self.sentences:
			raise ValueError(f'already have a sentence type named {name}')
		self.sentences[name] = template
		return template


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		ID = ctx.identify(self)
		return self.sentences[ID].grab_from(ctx, ID)



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
		self.content = values.get(self.value, {})


	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates(self.content.keys(),
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
		if gizmo in self.content:
			return self.content[gizmo]
		if gizmo in self.data:
			return self.data[gizmo]
		raise ToolFailedError(gizmo)



class StatisticalTerm(Variable):
	marginal_style = None
	conditional_style = None

	def __init__(self, data, *, conditions=None, **kwargs):
		super().__init__(data, **kwargs)
		self.given = conditions or {}


	def gizmos(self) -> Iterator[str]:
		yield 'sentence'
		yield from super().gizmos()


	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo == 'sentence':
			custom = self.conditional_style if len(self.given) else self.marginal_style
			if custom is not None:
				return custom.grab_from(ctx, gizmo)
		return super().grab_from(ctx, gizmo)


class MarginalStyle(CapitalizedTemplater, StaticTemplater):
	def __init__(self, name = 'sentence', template = '{sentence}.'):
		super().__init__(name, template)
class ConditionalStyle(CapitalizedTemplater, TemplateChoice):
	def grab_from(self, ctx: Verbalizer, gizmo: str) -> Any:
		if gizmo == self.name:
			cond = ctx['cond'] # always check the conditional clause first
			if cond is None:
				sentence = ctx["sentence"]
				sentence = sentence[0].upper() + sentence[1:]
				return sentence + '.'
		return super().grab_from(ctx, gizmo)


def default_vocabulary(seed=None):
	verb = Verbalizer(seed=seed)

	full = _get_template_data()

	StatisticalTerm.marginal_style = MarginalStyle()
	StatisticalTerm.conditional_style = ConditionalStyle()

	term_defaults = full['default-structure']
	terms = [StaticTemplater(key, val) for key, val_options in term_defaults.items() for val in val_options]
	verb.include(*terms)

	freq_text = TemplateChoice('freq_text', full['frequency']['options'])
	verb.include(freq_text)

	event = TemplateChoice('event_text', full['event-structure'])
	verb.include(event)

	sentences = SentenceChoice()
	verb.include(sentences)

	freq = TemplateChoice('freq', {str(i): code for i, code in enumerate(full['frequency']['structure'])})
	sentences.register_sentence(freq)

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
	verb = story.term(term)
	ctx.include(verb)

	# out = ctx['freq_text']

	text = ctx['sentence']

	_verb_str = str(ctx)


	print(out)

	pass

