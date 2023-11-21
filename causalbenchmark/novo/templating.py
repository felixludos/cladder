from .imports import *
from omnibelt import pformat


class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		keys = set(self._extract_template_keys(template))
		self.keys = keys
		self.template = template


	@staticmethod
	def _extract_template_keys(template: str):
		# pattern = r'\{([_a-zA-Z][_a-zA-Z0-9]*)\}'
		# pattern = r'(?<!\{)\{([_a-zA-Z][_a-zA-Z0-9]*)\}(?!\})'
		pattern = r'\{([_a-zA-Z][_a-zA-Z0-9]*(?:!s|!r|!a)?(?:\:[^}]*)?)\}'
		# pattern = r'(?<!\{)\{([_a-zA-Z][_a-zA-Z0-9]*(?:!s|!r|!a)?(?:\:[^}]*)?)\}(?!\})'
		for match in re.finditer(pattern, template):
			if '{' not in match.group(1) and '}' not in match.group(1):
				yield match.group(1).split('!')[0].split(':')[0]



	def fill_in(self, reqs: Dict[str, str] = {}, **vals: str):
		# use pformat?
		vals.update({key: reqs[key] for key in self.keys if key not in vals})
		# return self.template.format(**vals)
		return pformat(self.template, **vals)



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
















