from .imports import *



class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		keys = set(self._extract_template_keys(template))
		self.keys = keys
		self.template = template


	@staticmethod
	def _extract_template_keys(template: str):
		# return re.match("^[a-zA-Z_][a-zA-Z0-9_]*$", name) is not None
		# for match in re.finditer(r'\{([^\}]+)\}', template):
		for match in re.finditer(r'\{([_a-zA-Z][_a-zA-Z0-9]*)\}', template):
			yield match.group(1)



	def fill_in(self, reqs: Dict[str, str]):
		# use pformat?
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
















