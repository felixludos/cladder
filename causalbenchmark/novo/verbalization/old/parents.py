from ..imports import *

from omniply.core.gaggles import MutableGaggle
from .decisions import TemplateDecision



class ParentVerbalization(MutableGaggle, ToolKit):
	def __init__(self, structure=None, heads=None, **kwargs):
		if structure is None:
			structure = TemplateDecision('sentence', {
				'prefix': {'sentence': '{cond}, {claim}'},
				'suffix': {'sentence': '{claim} {cond}'},
			})
		super().__init__(**kwargs)
		self.structure = structure
		self.heads = heads
		self.include(*structure, heads)

	@tool('cond')
	def join_parents(self, parents):
		raise NotImplementedError















