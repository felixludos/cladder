from .imports import *
from omnibelt import pformat, pformat_vars


class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		keys = list(pformat_vars(template))
		self.keys = keys
		self.template = template


	def fill_in(self, reqs: dict[str, str] = {}, **vals: str):
		# use pformat?
		vals.update({key: reqs[key] for key in self.keys if key not in vals})
		# return self.template.format(**vals)
		return pformat(self.template, reqs, **vals)



class FixedTemplate(AbstractGadget, SimpleTemplater):
	def __init__(self, gizmo: str, template: str, **kwargs):
		super().__init__(template, **kwargs)
		self.gizmo = gizmo


	def gizmos(self) -> Iterator[str]:
		yield self.gizmo


	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str = None) -> Any:
		# assert gizmo == self.gizmo
		# reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		# out = self.fill_in(reqs)
		# return out
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		return self.fill_in(reqs)
















