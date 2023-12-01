from omniply import ToolKit, tool



class PrecisePercent(ToolKit):
	def __init__(self, *, epsilon=0.01, **kwargs):
		super().__init__(**kwargs)
		self.epsilon = epsilon


	@tool('quantity')
	def format_probability(self, mean: float):
		return f'{mean * 100:.0f}%'


	@tool('implication')
	def from_precise(self, mean: float):
		return [max(0., mean - self.epsilon/2), min(1., mean + self.epsilon/2)]


























