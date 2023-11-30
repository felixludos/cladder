from .imports import *
from omnibelt import pformat, pformat_vars, pathfinder

from . import misc



class SimpleTemplater:
	def __init__(self, template: str, **kwargs):
		super().__init__(**kwargs)
		self.keys = list(pformat_vars(template))
		self.template = template


	def fill_in(self, reqs: dict[str, str] = None, **vals: str):
		return pformat(self.template, reqs, **vals)



class FixedTemplate(SingleGadgetBase, SimpleTemplater):
	def __init__(self, template: str, gizmo: str = None, **kwargs):
		super().__init__(gizmo=gizmo, template=template, **kwargs)


	def gizmos(self) -> Iterator[str]:
		yield self._gizmo


	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str = None) -> Any:
		# assert gizmo == self.gizmo
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		return self.fill_in(reqs)



load_template = pathfinder(default_dir=misc.prompt_root(), default_suffix='txt', must_exist=True,
						   validate=lambda p: p.is_file())



class FileTemplate(SimpleTemplater):
	_find_template_path = load_template

	def __init__(self, template_name: str = None, *, template_path=None, **kwargs):
		template_path = self._find_template_path(template_name, path=template_path)
		template = template_path.read_text()
		super().__init__(template=template, **kwargs)
		self.template_path = template_path



class LoadedTemplate(FileTemplate, FixedTemplate):
	def __init__(self, template_name: str = None, gizmo: str = None, **kwargs):
		super().__init__(template_name=template_name, gizmo=gizmo, **kwargs)



class TokenTemplater:
	def __init__(self, tokens: list[str], **kwargs):
		super().__init__(**kwargs)
		self.tokens = tokens
		self.keys = set(key for keys in map(pformat_vars, tokens) for key in keys)


	def fill_in(self, reqs: dict[str, str] = None, **vals: str):
		reqs = reqs or {}
		vals.update({key: reqs[key] for key in self.keys if key not in vals})
		return [pformat(token, reqs, **vals) for token in self.tokens]



class TokenTemplate(SingleGadgetBase, TokenTemplater):
	def __init__(self, tokens: list[str], gizmo: str = None, **kwargs):
		super().__init__(gizmo=gizmo, tokens=tokens, **kwargs)


	def grab_from(self, ctx: Optional[AbstractGig], gizmo: str = None) -> Any:
		reqs = {key: ctx.grab_from(ctx, key) for key in self.keys}
		return self.fill_in(reqs)



class Template(SingleGadgetBase):
	def __init__(self, template: str | list[str], gizmo: str = None, **kwargs):
		if isinstance(template, str):
			template = [template]
		super().__init__(gizmo=gizmo, **kwargs)
		self._tokens = template
		self._template_keys = set(key for keys in map(pformat_vars, template) for key in keys)


	def _fill_in(self, *srcs: dict[str, str], **vals: str):
		return [pformat(token, *srcs, **vals) for token in self._tokens]

	def fill_in(self, *srcs: dict[str, str], **vals: str):
		return self.detok(self._fill_in(*srcs, **vals))

	def grab_from(self, ctx: 'AbstractGig', gizmo: str = None) -> Any:
		reqs = {key: ctx.grab(key) for key in self._template_keys}
		return self.fill_in(reqs)


	@staticmethod
	def detok(tokens: list[str] | str, capitalize: bool = False, sentence: bool = False):
		if isinstance(tokens, str):
			if capitalize:
				tokens = tokens[0].upper() + tokens[1:]
			if sentence:
				tokens = tokens + '.'
			return tokens

		assert len(tokens), f'Cannot detokenize empty list: {tokens}'
		toks = [token + ' ' if next_token not in [',', '.'] else token
				for token, next_token in zip(tokens, tokens[1:])] + [tokens[-1]]
		if capitalize:
			toks[0] = toks[0][0].upper() + toks[0][1:]
		if sentence:
			toks.append('.')
		return ''.join(toks)



# class template:
# 	def __init__(self, gizmo: str):
# 		self.gizmo = gizmo
#
# 	def __call__(self, template: str):
# 		return FixedTemplate(self.gizmo, template)











