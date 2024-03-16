

import omnifig as fig

import os
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.set_default_device("cuda")





@fig.script('ping')
def ping(cfg: fig.Configuration):
	print(os.environ.get('HF_HOME', None))
	print('pong')

	if (model_name := cfg.pull('model', None)): # "microsoft/phi-2"
		print()
		print(f'Loading model: {model_name!r}')

		# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
		model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
													 trust_remote_code=True, device_map='cuda', load_in_4bit=True)
		tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map='cuda')

		pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

		with torch.no_grad():
			messages = [
				{
					"role": "system",
					"content": "You are a friendly chatbot who always responds in the style of a pirate",
				},
				{"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
			]
			prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
		print(outputs[0]["generated_text"])

























