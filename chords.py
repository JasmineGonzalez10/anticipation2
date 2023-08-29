"""
Utilities for working with chords.
"""

from collections import defaultdict

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from anticipation.convert import *
from mido import MidiFile

from HSA.preprocessing.exported_midi_chord_recognition.main import transcribe_midi

import pandas as pd

CHORD_DICT = {
    #           1     2     3  4     5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }

CHORD_TYPE_ENCODING = {
    'maj':     0,
    'min':     1,
    'aug':     2,
    'dim':     3,
    'sus4':    4,
    'sus4(b7)':5,
    'sus4(b7,9)':6,
    'sus2':    7,
    '7':       8,
    'maj7':    9,
    'min7':    10,
    'minmaj7': 11,
    'maj6':    12,
    'min6':    13,
    '9':       14,
    'maj9':    15,
    'min9':    16,
    '7(#9)':   17,
    'maj6(9)': 18,
    'min6(9)': 19,
    'maj(9)':  20,
    'min(9)':  21,
    'maj(11)': 22,
    'min(11)': 23,
    '11':      24,
    'maj9(11)':25,
    'min11':   26,
    '13':      27,
    'maj13':   28,
    'min13':   29,
    'dim7':    30,
    'hdim7':   31
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }

CHORD_NOTE_ENCODING = {
    "N": 0,
    "C": 1, 
    "C#": 2,
    "D": 3,
    "Eb": 4,
    "E": 5,
    "F": 6,
    "F#": 7,
    "G": 8,
    "Ab": 9,
    "A": 10,
    "Bb": 11, 
    "B": 12
}

MIDI_ENCODING = {
    "N": 0,
    "C": 60, 
    "C#": 61,
    "D": 62,
    "Eb": 63,
    "E": 64,
    "F": 65,
    "F#": 66,
    "G": 67,
    "Ab": 68,
    "A": 69,
    "Bb": 70, 
    "B": 71
}

''' INFRASTRUCTURE FOR ENCODING COMEPLETE CHORD TEXT FILES.
'''

def seconds_to_10ms(time):
    result = round(float(time) * 100)
    return result

def encode_chord(chord_name):
    info_list = chord_name.split(':')
    chord = info_list[0]
    if len(info_list) > 1:
        type = info_list[1]
        type = type.split('/')[0]
    else:
        type = 'maj'
    
    chord_metric = CHORD_NOTE_ENCODING[chord]
    type_metric = CHORD_TYPE_ENCODING[type]

    if chord_metric == 0:
        result = 0
    else: 
        result = 12*type_metric + chord_metric

    return result

def encode_text_file(text_file):
    df = pd.DataFrame(columns=['Start', 'End', 'Chord'])
    with open(text_file, 'r') as f:
        text_chords = f.read().splitlines()
        for line in text_chords:
            df.loc[len(df.index)] = line.split()

    encoding = []
    for index, row in df.iterrows():
        encoding.append(seconds_to_10ms(row['Start']))
        encoding.append(seconds_to_10ms(row['End']) - seconds_to_10ms(row['Start']))
        '''if encode_chord(row['Chord']) < 0:
            print(row)'''
        encoding.append(encode_chord(row['Chord']))
        
    encoding = [tok + CONTROL_OFFSET for tok in encoding]

    return encoding

def encode(midi):
    text_chords = transcribe_midi(midi)
    encoding = encode_text_file(text_chords)
    return encoding

''' INFRASTRUCTURE FOR SONIFYING A SET OF CHORD ENCODING LABELS
'''
def get_chord_type(encoding):
    encoding = encoding - CONTROL_OFFSET
    multiple = encoding // 12

    #get the type key
    type = list(CHORD_TYPE_ENCODING.keys())[list(CHORD_TYPE_ENCODING.values()).index(multiple)]

    return type

def get_chord_base_note(encoding):
    encoding = encoding - CONTROL_OFFSET
    multiple = encoding // 12
    offset = encoding % 12

    if multiple != 0 and offset == 0:
        offset = 12

    base_note = list(CHORD_NOTE_ENCODING.keys())[list(CHORD_NOTE_ENCODING.values()).index(offset)]
    
    return base_note

def get_full_chord(encoding):
    #ACCOUNT FOR IT BEING 'N'
    type = get_chord_type(encoding)
    base_note = get_chord_base_note(encoding)
    
    midi_base = MIDI_ENCODING[base_note]
    if midi_base == 0:
        return []

    notes = []
    
    chord_structure = CHORD_DICT[type]
    for index in range(len(chord_structure)):
        if chord_structure[index] == 1:
            notes.append(midi_base + index)

    return notes

def get_chord_with_timing(token_triple):
    notes = get_full_chord(token_triple[2])
    '''updated_notes =[]
    for note in notes:
        note = note + 11000
        updated_notes.append(note)'''

    tokens = []
    #for note in updated_notes:
    for note in notes:
        tokens.append(token_triple[0] + TIME_OFFSET)
        tokens.append(token_triple[1] + DUR_OFFSET)
        tokens.append(note + ANOTE_OFFSET)

    return tokens
    #append this tokenized sequence to a total sequence that's being built up IN MAIN

def chordify(tokens):
    result_tokens = []
    
    while len(tokens) > 0:
        result_tokens = result_tokens + get_chord_with_timing(tokens[:3])
        tokens = tokens[3:]
        
    return result_tokens


if __name__ == '__main__':
    import sys
    if(len(sys.argv) != 2):
        print('Usage: chords.py midi_path')
        exit(0)
    encoding = encode(sys.argv[1])
    chord_tokens = [tok - CONTROL_OFFSET for tok in chordify(encoding)]
    #print(chord_tokens)
    orig_midi = MidiFile(sys.argv[1])
    tokens = midi_to_events(orig_midi)
    #print(tokens)
    tokens = tokens + chord_tokens
    print(tokens)
    mid = events_to_midi(tokens)
    #mid = events_to_midi(chord_tokens)
    mid.save(f'chordified_midi.mid')
    print(f' Tokenized MIDI Length: {mid.length} seconds ({len(tokens)} tokens)')
