from ..imports import *

from omniply.core.gaggles import LoopyGaggle, MutableGaggle
from .base import AbstractDecision, AbstractCrawler, SimpleCrawler, SimpleFrame



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



class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		keys = set(self._extract_template_keys(template))
		self.keys = keys
		self.template = template


	@staticmethod
	def _extract_template_keys(template: str):
		for match in re.finditer(r'\{([^\}]+)\}', template):
			yield match.group(1)


	def fill_in(self, reqs: Dict[str, str]):
		return self.template.format(**reqs)



class FixedTemplate(AbstractGadget, SimpleTemplater):
	def __init__(self, gizmo: str, template: str, **kwargs):
		super().__init__(template, **kwargs)
		self.gizmo = gizmo


	def gizmos(self) -> Iterator[str]:
		yield self.gizmo


	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str) -> Any:
		assert gizmo == self.gizmo
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		out = self.fill_in(reqs)
		return out



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



class StaticTemplater(AbstractGadget, SimpleTemplater):
	def __init__(self, key: str, template: Union[str, Sequence[str]], **kwargs):
		super().__init__(template, **kwargs)
		self.key = key
		# self.template = template


	def gizmos(self) -> Iterator[str]:
		yield self.key


	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str) -> Any:
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		out = self.fill_in(reqs)
		return out



class AsSentence(AbstractGadget):
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

	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str) -> Any:
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



class Verbalizer(SimpleCrawler, LoopyGaggle, MutableGaggle):
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
				except GadgetError as e:
					ctx['error'] = e
					ctx[gizmo] = empty_value
				ctx['identity'] = ctx.identity()
			yield ctx

	def spawn_gizmo(self, gizmo: str, *, empty_value=None) -> Iterator[str]:
		for ctx in self.spawn(gizmo, empty_value=empty_value):
			if ctx[gizmo] is not empty_value:
				yield ctx[gizmo]







