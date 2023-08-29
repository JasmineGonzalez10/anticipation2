import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from anticipation.convert import *
from anticipation.vocab import SEPARATOR
from anticipation.ops import print_tokens

# initialize the model and tokenizer
model_name = f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# set the seed for reproducibility
torch.manual_seed(42)

# define the prompt
input_ids = torch.tensor([[62, 67, 128, 38, 3, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026]]).to(device)

# define the number of tokens to generate
num_tokens_to_generate = 1000

# generate the tokens
for _ in range(num_tokens_to_generate):
    # generate the logits and past_key_values
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]

    # sample the next token
    probabilities = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)

    # append the next token to the input_ids
    input_ids = torch.cat([input_ids, next_token], dim=-1)

# decode the generated sequence
generated_sequence = tokenizer.decode(input_ids.squeeze().tolist())
generated = input_ids.squeeze().tolist()
# print the generated sequence
print(generated[16:])
print_tokens(generated[16:])
mid = events_to_midi(generated[16:])
mid.save(f'/jagupard26/scr1/gonzalez2/model_output/unconditional.mid')
