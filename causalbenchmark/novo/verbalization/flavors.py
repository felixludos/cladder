from omniply import ToolKit, tool



class PrecisePercent(ToolKit):
	epsilon = 0.01

	@tool('quantity')
	def format_probability(self, mean: float):
		return f'{mean * 100:.0f}%'


	@tool('implication')
	def from_precise(self, mean: float):
		return [max(0., mean - self.epsilon/2), min(1., mean + self.epsilon/2)]


























