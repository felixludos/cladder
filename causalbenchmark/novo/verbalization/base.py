from typing import Any
import random

import omnifig as fig
from omniply import Context

import torch

from .. import misc
from .decision import Decision



class Verbalization(Context, fig.Configurable):
	def __init__(self, rng=None, **kwargs):
		super().__init__(**kwargs)
		self._rng = misc.get_rng(rng)


	def decisions(self, gizmo: str = None):
		for gadget in self._vendors(gizmo=gizmo):
			if isinstance(gadget, Decision):
				yield gadget


	def identity(self, gizmo: str = None):
		identity = {decision.identity(): None for decision in self.decisions(gizmo=gizmo)}
		identity.update({key: self[key] for key in self.cached() if key in identity})
		if gizmo is None:
			return {k:v for k,v in identity.items() if v is not None}
		return identity.get(gizmo)


	def select(self, choices: list[Any]):
		'''use self.rng (torch.Generator) to choose an element'''
		return choices[torch.randint(0, len(choices), (1,), generator=self._rng).item()]















