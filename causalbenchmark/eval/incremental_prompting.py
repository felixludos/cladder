from typing import Dict, Any, Iterable

long_query_types = {
	'correlation': 'Correlation',
	'marginal': 'Marginal Distribution',
	'exp_away': 'Expaining Away Effect',
	'ate': 'Average Treatment Effect',
	'backadj': 'Backdoor Adjustment Set',
	'collider_bias': 'Collider Bias',
	'ett': 'Effect of the Treatment on the Treated',
	'nde': 'Natural Direct Effect',
	'nie': 'Natural Indirect Effect',
	'det-counterfactual': 'Counterfactual'
}

known_steps = ['variables', 'graph', 'query_type', 'formal_form', 'available_data',  # parsing
              'estimand', 'estimate', 'interpretation'  # reasoning
              ]


def sub_question_prompts(data: Dict[str, Any], include_steps: Iterable =('variables', 'graph')):
	'''
	when given an unordered collection of known steps returns a dict of solved sub-questions and their answers
	when given a list or tuple of known steps returns the corresponding list of solved sub-questions and their answers
	
	Note: adjustement set questions are not supported
	'''
	assert all([step in known_steps for step in include_steps]), f'include_steps must be a subset of {known_steps}'
	
	terms = {}
	
	if 'variables' in include_steps:
		terms['variables'] = data['reasoning']['step0']
	
	if 'graph' in include_steps:
		step1 = data['reasoning']['step1'].replace(',', ', ')
		terms['graph'] = f'Step 1) Extract the causal graph: ' \
		                 f'The causal graph expressed in the context is: "{step1}".'
	
	qtype = data['meta']['query_type']
	query_title = long_query_types[qtype].lower()
	if 'query_type' in include_steps:
		terms['query_type'] = f'Step 2) Identify the query type: ' \
		                      f'The query type of the above question is "{query_title}".'
	
	formal_form = data['meta']['formal_form']  # should be equivalent to data['reasoning']['step2']
	if 'formal_form' in include_steps:
		terms['formal_form'] = f'Step 3) Formalize the query: ' \
		                       f'The query can be formalized as "{formal_form}".'
	
	if 'available_data' in include_steps:
		step4 = data['reasoning']['step4'].replace('\n', '; ')
		if len(step4) == 0:
			terms['available_data'] = ''
		else:
			terms['available_data'] = f'Step 4) Parse all the available data: ' \
			                          f'The available data are: "{step4}".'
	
	estimand = data['meta'].get('estimand') # should be equivalent to data['reasoning']['step3']
	if estimand is None: # for rung 1 questions, estimand is the same as formal_form
		estimand = formal_form
		if qtype == 'collider_bias':
			estimand = '0'
	if 'estimand' in include_steps:
		terms['estimand'] = f'Step 5) Derive the estimand: ' \
		                    f'We use causal inference to derive the estimand ' \
		                    f'implied by the causal graph for the the query type "{query_title}":\n' \
		                    f'{formal_form}\n' \
		                    f'= {estimand}' \
		
	if 'estimate' in include_steps:
		estimate = data['reasoning']['step5']
		terms['estimate'] = f'Step 6) Estimate the estimand: By plugging in the available data, we can calculate:\n' \
		                    f'{estimand}\n' \
							f'= {estimate}'
	
	end = data['reasoning']['end']
	answer = data['answer']
	if 'interpretation' in include_steps:
		if qtype == 'collider_bias':
			reasoning = data['reasoning']['step3'].replace('.', '')
			terms['interpretation'] = f'Since {reasoning}, ' \
			                          f'the overall answer to the question is {answer}.'
		else:
			terms['interpretation'] = f'Since the estimate for the estimand is {end}, ' \
			                          f'the overall answer to the question is {answer}.'
	
	if isinstance(include_steps, (list, tuple)):
		return [terms[step] for step in include_steps]
	return terms



########################################################################################################################

# Unit tests


def test_sub_question_prompts():
	
	lines1 = sub_question_prompts(_corr_example, include_steps=('variables', 'graph', 'query_type', 'formal_form'))
	assert len(lines1) == 4
	assert isinstance(lines1, list)
	
	lines2 = sub_question_prompts(_corr_example, include_steps={'variables', 'graph', 'query_type', 'formal_form',})
	assert len(lines2) == 4
	assert isinstance(lines2, dict)
	
	lines3 = sub_question_prompts(_corr_example, include_steps=known_steps)
	lines4 = sub_question_prompts(_nde_example, include_steps=known_steps)
	lines5 = sub_question_prompts(_ate_example, include_steps=known_steps)
	lines6 = sub_question_prompts(_collider_example, include_steps=known_steps)
	
	assert len(lines3) == 6
	
	
	
_corr_example = {"question_id": 1863, "desc_id": "college_wage-IV-correlation-model437-spec17-q0",
                 "given_info": "The overall probability of college degree or higher is 47%. The probability of high school degree or lower and high salary is 18%. The probability of college degree or higher and high salary is 41%.",
                 "question": "Is the chance of high salary larger when observing college degree or higher?",
                 "answer": "yes", "meta": {"story_id": "college_wage", "graph_id": "IV",
                                           "given_info": {"P(X=1)": 0.47354802854314104,
                                                          "P(Y=1, X=0)": 0.17650584399857128,
                                                          "P(Y=1, X=1)": 0.4059889107674839}, "treated": True,
                                           "polarity": True, "groundtruth": 0.5220598272771784,
                                           "query_type": "correlation", "rung": 1, "formal_form": "P(Y | X)",
                                           "treatment": "X", "outcome": "Y", "model_id": 437}, "reasoning": {
		"step0": "Let V2 = proximity to a college; V1 = unobserved confounders; X = education level; Y = salary.",
		"step1": "V1->X,V2->X,V1->Y,X->Y", "step2": "P(Y | X)",
		"step3": "P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)",
		"step4": "P(X=1=1) = 0.47\nP(Y=1, X=0=1) = 0.18\nP(Y=1, X=1=1) = 0.41",
		"step5": "0.41/0.47 - 0.18/0.53 = 0.52", "end": "0.52 > 0"}}
_nde_example = {"question_id": 1989, "desc_id": "encouagement_program-mediation-nde-model480-spec0-q1",
                "given_info": "For students who are not encouraged and do not study hard, the probability of high exam score is 27%. For students who are not encouraged and study hard, the probability of high exam score is 52%. For students who are encouraged and do not study hard, the probability of high exam score is 47%. For students who are encouraged and study hard, the probability of high exam score is 82%. For students who are not encouraged, the probability of studying hard is 31%. For students who are encouraged, the probability of studying hard is 57%.",
                "question": "If we disregard the mediation effect through studying habit, would encouragement level negatively affect exam score?",
                "answer": "no",
                "meta": {"story_id": "encouagement_program", "graph_id": "mediation", "mediators": ["V2"],
                         "polarity": False, "groundtruth": 0.23454564025526198, "query_type": "nde", "rung": 3,
                         "formal_form": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]", "given_info": {
		                "p(Y | X, V2)": [[0.26899135063231316, 0.5200497020408603],
		                                 [0.47177200935207114, 0.8239014053227091]],
		                "p(V2 | X)": [0.3142836969097487, 0.566019651573423]},
                         "estimand": "\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]",
                         "treatment": "X", "outcome": "Y", "model_id": 480},
                "reasoning": {"step0": "Let X = encouragement level; V2 = studying habit; Y = exam score.",
                              "step1": "X->V2,X->Y,V2->Y", "step2": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]",
                              "step3": "\\sum_{V2=v} P(V2=v|X=0)*[P(Y=1|X=1,V2=v) - P(Y=1|X=0, V2=v)]",
                              "step4": "P(Y=1 | X=0, V2=0) = 0.27\nP(Y=1 | X=0, V2=1) = 0.52\nP(Y=1 | X=1, V2=0) = 0.47\nP(Y=1 | X=1, V2=1) = 0.82\nP(V2=1 | X=0) = 0.31\nP(V2=1 | X=1) = 0.57",
                              "step5": "0.31 * (0.82 - 0.47) + 0.57 * (0.52 - 0.27) = 0.23", "end": "0.23 > 0"}}
_ate_example = {"question_id": 4015, "desc_id": "smoke_birthWeight-arrowhead-ate-model931-spec1-q0",
                "given_info": "For infants with nonsmoking mothers, the probability of high infant mortality is 37%. For infants with smoking mothers, the probability of high infant mortality is 70%.",
                "question": "Will smoking mother increase the chance of high infant mortality?", "answer": "yes",
                "meta": {"story_id": "smoke_birthWeight", "graph_id": "arrowhead", "treated": True, "result": True,
                         "polarity": True, "groundtruth": 0.32771023874167765, "query_type": "ate", "rung": 2,
                         "formal_form": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
                         "given_info": {"p(Y | X)": [0.3735625760338536, 0.7012728147755312]},
                         "estimand": "P(Y=1|X=1) - P(Y=1|X=0)", "treatment": "X", "outcome": "Y", "model_id": 931},
                "reasoning": {
	                "step0": "Let V2 = health condition; X = maternal smoking status; V3 = infant's birth weight; Y = infant mortality.",
	                "step1": "X->V3,V2->V3,X->Y,V2->Y,V3->Y", "step2": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
	                "step3": "P(Y=1|X=1) - P(Y=1|X=0)", "step4": "P(Y=1 | X=0) = 0.37\nP(Y=1 | X=1) = 0.70",
	                "step5": "0.70 - 0.37 = 0.33", "end": "0.33 > 0"}}
# _adjset_example = {"question_id": 1897, "desc_id": "college_wage-IV-backadj-model448-spec28-q1", "given_info": "Method 1: We look at how education level correlates with salary case by case according to unobserved confounders. Method 2: We look directly at how education level correlates with salary in general.", "question": "To understand how education level affects salary, is it more correct to use the Method 1 than Method 2?", "answer": "yes", "meta": {"story_id": "college_wage", "graph_id": "IV", "flipped": False, "polarity": True, "groundtruth": ["V1"], "bad_candidate_set": [], "given_info": [["V1"], []], "query_type": "backadj", "rung": 2, "formal_form": "[backdoor adjustment set for Y given X]", "treatment": "X", "outcome": "Y", "model_id": 448}, "reasoning": None}
_collider_example = {"question_id": 1903, "desc_id": "elite_students-collision-collider_bias-model450-spec0-q0",
                     "given_info": "For students accepted to elite institutions, the correlation between talent and being hard-working is -0.05.",
                     "question": "If we look at students accepted to elite institutions, does it mean that talent affects being hard-working?",
                     "answer": "no",
                     "meta": {"story_id": "elite_students", "graph_id": "collision", "treated": True,
                              "result": True, "baseline": True, "polarity": True, "collider": "V3",
                              "groundtruth": "no", "given_info": {
		                     "P(Y = 1 | X = 1, V3 = 1) - P(Y = 1 | X = 0, V3 = 1)": -0.05188335790700426},
                              "formal_form": "E[Y = 1 | do(X = 1), V3 = 1] - E[Y = 1 | do(X = 0), V3 = 1]",
                              "query_type": "collider_bias", "rung": 2, "treatment": "X", "outcome": "Y",
                              "model_id": 450},
                     "reasoning": {"step0": "Let Y = effort; X = talent; V3 = elite institution admission status.",
                                   "step1": "X->V3,Y->V3",
                                   "step2": "E[Y = 1 | do(X = 1), V3 = 1] - E[Y = 1 | do(X = 0), V3 = 1]",
                                   "step3": "X and Y do not affect each other.", "step4": "", "step5": "0",
                                   "end": "no"}}
















