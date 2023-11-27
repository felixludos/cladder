from .imports import *

from ..util import repo_root, assets_root


def repo_root():
	return Path(__file__).parent.parent.parent


def data_root():
	return repo_root() / 'novo-data'


def assets_root():
	return repo_root() / 'assets'


def prompt_root():
	return data_root() / 'prompts'


def temp_root():
	return data_root() / 'temp'



@fig.autocomponent('rng')
def get_rng(seed=None, reset_master=True):
	if seed is None:
		seed = np.random.randint(0, 2**16-1) # TODO what is the limit
	if reset_master:
		torch.manual_seed(seed)
	return torch.Generator().manual_seed(seed)













