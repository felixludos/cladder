from pathlib import Path
from omnibelt import load_json
import omnifig as fig
from omniply import tool, ToolKit, Context, Scope, Selection

import torch

from .. import misc



@fig.component('news-headlines-sparks')
class NewsHeadlines(ToolKit, fig.Configurable):
	def __init__(self, dataset_root=None, *, rng=None, locs=None, **kwargs):
		if rng is None:
			rng = misc.get_rng(rng)
		if dataset_root is None:
			dataset_root = Path('/home/fleeb/workspace/local_data/nnn/babel-briefings-v1')
		super().__init__(**kwargs)
		self.rng = rng
		self.dataset_root = dataset_root
		self._locs = locs
		self.paths = None
		self.articles = None
		self.article_IDs = None


	def load(self, *, pbar=None):
		paths = list(self.dataset_root.glob('**/*.json'))
		if self._locs is not None:
			paths = [path for path in paths if path.stem.split('-')[-1] in self._locs]
		self.paths = paths
		self.articles = {}
		itr = self.paths if pbar is None else pbar(self.paths)
		for path in itr:
			if pbar is not None:
				itr.set_description(f'Loading {path.name}')
			self.articles.update({art['ID']: art for art in self._load_article_path(path)})
		self.article_IDs = list(self.articles.keys())
		return self


	@staticmethod
	def _load_article_path(path):
		articles = load_json(path)
		return [{
			'ID': article['ID'],
			'title': article.get('en-title', article['title']),
			'description': article.get('en-description', article['description']),
			'content': article.get('en-content', article['content']),
			'language': article['language']} for article in articles]


	@tool('seed')
	def generate_context_seed(self) -> int:
		if self.articles is None:
			self.load()
		seed = torch.randint(0, len(self.article_IDs), (1,), generator=self.rng).item()
		return self.article_IDs[seed]


	@tool('article')
	def get_article(self, seed) -> dict[str, str]:
		return self.articles[seed]


	@tool('title')
	def get_title(self, article) -> str:
		return article['title']


	@tool('description')
	def get_description(self, article) -> str:
		return article['description']


	@tool('language')
	def get_language(self, article) -> str:
		return self._lang_names[article['language']]
	_lang_names = {'en': 'English', 'ko': 'Korean', 'ru': 'Russian', 'es': 'Spanish', 'pt': 'Portuguese', 'cs': 'Czech',
				  'tr': 'Turkish', 'nl': 'Dutch', 'ar': 'Arabic', 'fr': 'French', 'bg': 'Bulgarian', 'id': 'Indonesian',
				  'sk': 'Slovak', 'el': 'Greek', 'he': 'Hebrew', 'sr': 'Serbian', 'hu': 'Hungarian', 'th': 'Thai',
				  'zh': 'Chinese', 'no': 'Norwegian', 'sl': 'Slovenian', 'sv': 'Swedish', 'de': 'German',
				  'lv': 'Latvian',
				  'pl': 'Polish', 'it': 'Italian', 'ro': 'Romanian', 'lt': 'Lithuanian', 'ja': 'Japanese',
				  'uk': 'Ukrainian'}


	@tool('spark')
	def get_spark(self, title, description, language) -> str:
		lines = [
			f'Title: {title}',
			f'Description: {description}',
			f'Original Language: {language}',
		]
		return "\n".join(lines)



















