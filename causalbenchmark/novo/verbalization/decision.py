from typing import Any
import random
from omniply import Context
from omniply.core.abstract import AbstractMultiGadget, AbstractGadget



class Decision(AbstractMultiGadget):
	def __init__(self, choices: list['AbstractGadget'] | dict[str, 'AbstractGadget'],
				 name: str = None, id_name: str = None, **kwargs):
		assert name or id_name, f'Either name or id_name must be specified'
		if id_name is None:
			id_name = f'{name}_id'
		if not isinstance(choices, dict):
			choices = {i: c for i, c in enumerate(choices)}
		super().__init__(**kwargs)
		self._choice_ids = list(choices.keys())
		self._choices = choices
		self._name = name
		self._id_name = id_name


	def identity(self):
		return self._id_name


	def __len__(self):
		return len(self._choice_ids)


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
		return random.choice(self._choice_ids) if ctx is None else ctx.select(list(self._choice_ids))


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



# class DictGadget(AbstractGadget):
# 	_no_default = object()
# 	def __init__(self, data: dict[str, Any], default_value=_no_default, gizmos: list[str] = None, **kwargs):
# 		if gizmos is None:
# 			gizmos = list(data.keys())
# 		super().__init__(**kwargs)
# 		self._data = data
# 		self._gizmos = gizmos
# 		self._default_value = default_value
#
# 	def gizmos(self):
# 		yield from self._gizmos
#
# 	def grab_from(self, ctx: Context, gizmo: str = None):
# 		value = self._data.get(gizmo, self._default_value)
# 		if value is self._no_default:
# 			raise KeyError(f'No value for {gizmo}')
# 		return value






