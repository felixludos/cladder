



default_story_prompt_template = \
'''We are interested in building a causal model based on the explicit and implicit assumptions contained in the specified article and then using causal inference to evaluate the reasoning. Answer the following questions to design an interesting, simple, and most importantly realistic causal model from a news article.

To help understand the instructions here are some tips:
- all variables are always binary and (at least in principle) measurable, so when creating and selecting variables, make sure it is reasonable to treat them as binary
- whenever you propose a variable, make sure to define the meaning of each value it can take, and mention whether it is observable or not
- outcome variables are always observable
- treatment variables are always observable and intervenable, that means it must be possible to (at least in principle) change their value if desired
- confounder variables may or may not be observable, and should always have at least two causal children (for example, a treatment and outcome variable) and no causal parents
- mediator variables may or may not be observable, and should always have at least one causal parents (for example, a treatment variable) and at least one causal child (for example, the outcome variable)
- collider variables may or may not be observable, and should always have at least two causal parents (for example, a treatment and outcome variable) and no causal children

Answer concisely and precisely in the desired format. Carefully complete each of these tasks in order.

1. Write a short description of what the article is about and what causal model make inform the reasoning behind an argument the might make.
2. Propose 1 outcome variable that the news article is either implicitly or explicitly addressing that people are most likely to be interested in studying, especially if people tend to have misconceptions about it
3. Propose 2 treatment variables that either directly or indirectly affect the selected outcome variable and are the most interesting to study
4. Propose 3 confounder variables that affect some reasonable combination of the outcome and treatment variables
5. Propose 3 mediator variables that affect and are affected by some reasonable combination of any other variables
6. Propose 2 collider variables that are affected by some reasonable combination of any other variables

Here is the beginning of the new article:
```
{spark}
```

The variables and causal graph should, where possible, use specific details such as names and locations mentioned in the article. Also, generally the variable value "0" should correspond to the negative or control while the value "1" should correspond to the positive or greater value.

Take a deep breath and think step-by-step about how you will do this.'''



default_graph_prompt_template = \
'''1. From all proposed confounder, mediator, and collider variables, select between 3-5 variables that are most interesting to study and together with the treatment and outcome variables result in a realistic interesting causal graph. Important: Make sure the causal graph is a DAG and that no node has more than 3 parents!
2.  List all the edges in the causal graph, and make sure to mention which edges are observable and which are not.
3. Provide a python list of all the nodes in the proposed graph. For each node, provide the following information in the form of a python dict:
- `name`: the name of the variable
- `description`: a short description of the variable
- `type`: the type of the variable, which can be one of the following: `outcome`, `treatment`, `confounder`, `mediator`, `collider`
- `observed`: a boolean value indicating whether the variable is observable or not
- `values`: a python list of the descriptions of the values the variable can take (corresponding to the index)
- `parents`: a python list of the names of the parents of the variable (make sure they match the corresponding `name` field of the parent nodes)

Take a deep breath and think step-by-step about how you will do this.'''



default_stats_prompt_template = \
'''We have a causal bayes net based on the following article:

```
{spark}
```

Now we would like to estimate the probabilities of certain events.

Using commonsense, estimate the probabilities of the following events:

{questions}

Where the variables are defined as:
{descriptions}

For each question above, answer with the lower and upper bound estimates of the probabilities as a python dictionary where the key corresponds to the question index in exactly the following format:

```python
probabilities = {{
  1: [0.3, 0.4],
  ...
}}
```

Answer concisely and precisely in the desired format.'''


default_verb_prompt_template = '''We would like to find very natural verbalizations of the following binary variables selected from a statistical model. The verbalizations should sound more natural and organic, so that you could imagine them being used in a news paper article or casual conversation, while still being semantically equivalent to the original variable and value.

Here are the templates that we use to verbalize the variables:

1. One of the variables is {{variable}}.
2. We estimate {{subject}} [often/usually/sometimes/rarely/etc.] {{value}}.
3. There is a [number]% chance that {{value}}.
4. [number]% of {{domain}} {{value}}.
5. [number]% of the time {{value}}.
6. Conditional sentence: {{value}}, [some consequence].
7. Interventional: If {{value}}, [some effect].

{variable_description}

Here are some examples of verbalizations of the variable. For each of the templates fill in the blanks (and include 2-3 examples for each template). Format your response as a python dictionary. When replacing the key "value", make sure to include some examples for both values that the variables can take. For example, a variable "Smoking Rate" which takes the values "Low Smoking Rate"=0 and "High Smoking Rate"=1 can be verbalized like this:

```python
verbalizations = {{
  "Smoking Rate": {{
    1: {{"variable": ["the smoking rate", "proportion of smokers"]}},
    2: {{"subject": "people", "value": {{0: ["do not smoke", "are non-smokers"], 1: ["smoke", "are smokers"]}}}},
    3: {{"value": {{0: ["people do not smoke", "people are non-smokers"], 1: ["people smoke", "people are smokers"]}}}},
    4: {{"domain": "people", "value": {{0: ["do not smoke", "are non-smokers"], 1: ["smoke", "are smokers"]}}}},
    5: {{"value": {{0: ["the smoking rate decreases", "people stop smoking"], 1: ["the smoking rate increases", "people start smoking"]}}}},
    6: {{"value": {{0: ["For people that do not smoke", "Among non-smokers"], 1: ["For people that smoke", "Among smokers"]}}}},
    7: {{"value": {{0: ["people quite smoking", "people become non-smokers"], 1: ["people become smokers", "people take up smoking"]}}}},
  }},
  ...
}}
```

Answer concisely and precisely in the desired format, and only replace the blanks in curly braces. Do not add any additional comments or discussion. Most importantly, the verbalizations should not contain quantitative information like "often" or "rarely", and instead always assert the corresponding value.'''


default_question_prompt_template = '''We have a causal bayes net based on the following article:

```
{spark}
```

Now we would like to use this model to generate some interesting research questions:

1. Write a short two sentence introduction to describe motivation and purpose of the causal model to provide context to the questions in an conversational/debate setting. Where possible, include relevant details from the original article headline or related information you can infer therefrom. Avoid discussing specific structural properties or assumptions in the causal graph, and instead focus on the overall motivation and potential applications for the model. Refrain from mentioning words like "causal inference" or "model".
Make sure not to use any technical terms from causal inference so that the introduction is easy to understand even for a layperson, as if this were in a newspaper or casual discussion.
2. Write a short two sentence overview of the structure of the causal graph including noteworthy properties and assumptions. Use an intuitive conversational style to describe the causal graph, and avoid using technical terms or jargon, but you can describe the variables and their relationships (including using terms like "confounder"/"confounding", "mediator"/"mediates") as long as you explain them in a way that is understandable to a layperson.
3. Verbalize an interesting "yes"/"no" question where the correct answer depends computation of each of the following quantities:

{query_description}

Where the definitions of "ATE('X')" (average treatment effect) and "CATE('X' | 'U'=u)" are:

ATE('X') = E['Y' | do('X' = 1)] - E['Y' | do('X' = 0)]
CATE('X' | 'U'=u) = E['Y' | 'U'=u, do('X' = 1)] - E['Y' | 'U'=u, do('X' = 0)]

Where 'X' is a treatment variable, 'Y' is the outcome variable, and 'U' is a confounder variable.

4. In addition to the questions and associated answers include a one sentence explanation for each question, and "wrong_explanation" which is just like the explanation but argues for the opposite answer.

The meanings of all the variables (all of which are binary) in the statistical model are:

{variable_description}

Answer in the form of a python dictionary in the following format:

```python
setting = {{
  "introduction": "[introduction]",
  "overview": "[overview]",
  "questions": {{
    1: {{"question": "[question]", "answer": "yes", "explanation": "[explanation]", "wrong_explanation": "[wrong_explanation]"}},
    ...
  }},
}}
```
Answer concisely and precisely in the desired format. Do not add any additional comments or discussion. Most importantly, do not use any technical terms from causal inference such as ATE or CATE in the questions and explanations so that everything is easy to understand even for a layperson.'''


