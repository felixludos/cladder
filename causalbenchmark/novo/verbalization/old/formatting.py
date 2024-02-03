from ..imports import *



class NumberFormatter(ToolKit):
	def __init__(self, lower_condition=0.01, upper_condition=0.99, **kwargs):
		super().__init__(**kwargs)
		self.upper_condition = upper_condition
		self.lower_condition = lower_condition


	@tool.from_context('bound')
	def compute_bound(self, ctx):
		side = ctx['bound_side']
		other_side = 'upper_bound' if side == 'lower_bound' else 'lower_bound'
		val = ctx[f'{other_side}_value']
		if ((side == 'upper_bound' and val <= self.lower_condition)
				or (side == 'lower_bound' and val >= self.upper_condition)):
			return ctx[side]
		raise MissingGadget('bound')


	@tool('mean')
	def format_number(self, value):
		return f'{value:.0%}'


	@tool('lower_bound')
	def format_lower_bound(self, lower_bound_value):
		return f'{lower_bound_value:.0%}'


	@tool('lower_bound_100')
	def format_lower_bound_100(self, lower_bound_value):
		return f'{lower_bound_value*100:.0f}'


	@tool('upper_bound')
	def format_upper_bound(self, upper_bound_value):
		return f'{upper_bound_value:.0%}'


	@tool('implication')
	def infer_implication(self, lower_bound_value, upper_bound_value):
		return [lower_bound_value, upper_bound_value]









