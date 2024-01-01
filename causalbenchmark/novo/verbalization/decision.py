from typing import Any, Iterator, Callable
import random
from omnibelt import filter_duplicates
from omniply import Context, tool, ToolKit
from omniply.core.abstract import AbstractMultiGadget, AbstractGadget

# from omniply.core.gadgets import SingleGadgetBase
from omniply.core import GadgetFailure
from omniply.core.gaggles import CraftyGaggle
from omniply.core.tools import ToolCraft, ToolDecorator, ToolSkill, AutoToolCraft



class AbstractDecision(AbstractMultiGadget):
	def gizmos(self) -> Iterator[str]:
		yield from filter_duplicates(super().gizmos(), self.identity_keys())


	def identity_keys(self):
		raise NotImplementedError



class Decision(CraftyGaggle, AbstractDecision):
	_name = None
	def __init__(self, choices: list['AbstractGadget'] | dict[str, 'AbstractGadget'] = None,
				 name: str = None, id_name: str = None, choice_validator=None, **kwargs):
		if name is None:
			name = self._name
		assert name or id_name, f'Either name or id_name must be specified'
		if id_name is None:
			id_name = f'{name}_id'

		super().__init__(**kwargs)
		self._name = name
		self._id_name = id_name
		self._choice_validator = choice_validator
		self._choice_ids = []
		self._choices = {}
		self._process_crafts() # process `choice`s
		if choices is not None and len(choices) > 0:
			self._add_choices(choices)


	def __repr__(self):
		return f'Decision({self._name} with {len(self)} options)'


	def _add_choices(self, choices: list['AbstractGadget'] | dict[str, 'AbstractGadget']):
		if isinstance(choices, dict):
			return self.add_choice(**choices)
		return self.add_choice(*choices)
	def add_choice(self, *choices: AbstractGadget, **named_choices: AbstractGadget):
		named_choices.update({i+len(self): choice for i, choice in enumerate(choices)})
		for key, option in named_choices.items():
			if key in self._choice_ids:
				raise KeyError(f'Choice {key!r} already exists: {option}')
			self._choice_ids.append(key)
			self._choices[key] = option


	# def merge_into(self, decision: 'Decision'):
	# 	raise NotImplementedError


	def validate_choice(self, ctx, choice_id):
		if self._choice_validator is not None:
			return self._choice_validator(ctx, self._choices[choice_id])
		return True


	def _valid_choices(self, ctx):
		return [choice_id for choice_id in self._choice_ids if self.validate_choice(ctx, choice_id)]


	def identity_keys(self):
		yield self._id_name


	def __len__(self):
		return len(self._choice_ids)


	@property
	def product(self):
		return self._name


	def gizmos(self):
		past = set()
		if self._id_name is not None:
			past.add(self._id_name)
			yield self._id_name
		if self._name is not None:
			past.add(self._name)
			yield self._name
		for choice in self._choices.values():
			possible = ()
			if isinstance(choice, dict):
				possible = choice.keys()
			elif isinstance(choice, AbstractGadget):
				possible = choice.gizmos()
			for gizmo in possible:
				if gizmo not in past:
					past.add(gizmo)
					yield gizmo


	def choose(self, ctx: 'Verbalizer' = None):
		options = self._valid_choices(ctx)
		if len(options) == 0:
			raise GadgetFailure(f'None of the {len(self)} options for {self._id_name} are valid in this context')
		if len(options) == 1:
			return options[0]
		return random.choice(options) if ctx is None else ctx.select(options)


	def apply_choice(self, ctx: Context, choice: dict[str, Any] | AbstractGadget | Any, gizmo: str):
		if isinstance(choice, AbstractGadget):
			return choice.grab_from(ctx, gizmo)
		if isinstance(choice, dict):
			return choice[gizmo]
		if callable(choice):
			return choice(ctx)
		return choice


	def grab_from(self, ctx: Context, gizmo: str = None):
		if gizmo is None:
			gizmo = self._name
		if gizmo == self._id_name:
			return self.choose(ctx)
		return self.apply_choice(ctx, self._choices[ctx[self._id_name]], gizmo)



class ChoiceCraftBase(ToolCraft):
	'''
	This decorator can only be used for methods of subclasses of Decisions!
	'''
	def __init__(self, fn, *, gizmo=None, **kwargs):
		super().__init__(gizmo=gizmo, fn=fn, **kwargs)
		self._choice_id = None


	def __call__(self, fn):
		self._choice_id = self._fn
		self._fn = fn
		return self


	def as_skill(self, owner: Decision) -> ToolSkill:
		if not isinstance(owner, Decision):
			raise TypeError(f'To use a `choice`, the owner must be an AutoDecision')

		unbound_fn = self._wrapped_content_leaf()
		fn = unbound_fn.__get__(owner, type(owner))
		skill = self._ToolSkill(owner.product, fn=fn, unbound_fn=unbound_fn, base=self)
		if self._choice_id is None:
			owner.add_choice(skill)
		else:
			owner.add_choice(**{self._choice_id: skill})
		return skill



class choice(ChoiceCraftBase, AutoToolCraft):
	class from_context(ChoiceCraftBase):
		pass





