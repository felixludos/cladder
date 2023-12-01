
import omnifig as fig

from .decision import Decision
from ..templating import Template



class VerbalizationBase(fig.Configurable):
	_DecisionType = Decision
	_TemplateType = Template
