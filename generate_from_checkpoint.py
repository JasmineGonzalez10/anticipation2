import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/jagupard26/scr1/gonzalez2/anticipation')

from transformers import GPT2LMHeadModel

from anticipation import ops
from anticipation.vocab import *
from anticipation.sample import generate
from anticipation.convert import *

from anticipation.vocab import SEPARATOR

LENGTH_IN_SECONDS = 15

model_name = 'absurd-deluge-4'
#step_number = 10000

model = GPT2LMHeadModel.from_pretrained(f'/nlp/scr/jthickstun/absurd-deluge-4/step-100000/hf').cuda()

prompt = [SEPARATOR, SEPARATOR, SEPARATOR]
#prompt = []
controls = midi_to_events(f'/jagupard26/scr1/gonzalez2/model_input/twinkle_final.mid')

###########
'''
compound = events_to_compound(controls)
for i in range(len(compound[2::5])):
    compound[i*5 + 2] = compound[i*5 + 2] - 4
controls = compound_to_events(compound)
''' ###########

controls = [tok + CONTROL_OFFSET for tok in controls]
# adding delay for sampling
#controls[0::3] = [tok + 500 for tok in controls[0::3]]

#PAD WITH INSTRUMENT CONTROLS
instruments = [62, 67, 128, 38, 3, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026, 55026]
controls = instruments + controls

for i in range(5):
    generated_tokens = generate(model, 0, LENGTH_IN_SECONDS, prompt, controls, top_p=0.98, debug=True)
    mid = events_to_midi(generated_tokens)
    mid.save(f'/jagupard26/scr1/gonzalez2/model_output/{model_name}/generated-twinkle-824-{i}.mid')
