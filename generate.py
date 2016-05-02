#! /usr/bin/env python
"""
Filename: generate.py
Author: Emily Daniels
Date: May 2016
Purpose: Sentences are generated from the trained model and split into haiku
poems.
"""
from train import Train
import numpy as np
import string


class Generate(object):

    def __init__(self, vocabulary_size, hidden_dim, learning_rate, nepoch,
                 enable_training, model_file, train_file, num_sentences):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.nepoch = nepoch
        self.enable_training = enable_training
        self.train_file = train_file
        self.model_file = model_file
        self.poems = []
        self.new_sent = ""
        self.exclude = set(string.punctuation)
        self.first = ""
        self.second = ""
        self.third = ""
        self.num_sentences = num_sentences
        self.add_lines()

    def generate_sentence(self, train, model):
        # start the sentence with the start token
        new_sentence = [train.word_to_index[train.sentence_start_token]]
        # repeat until we get an end token
        while not new_sentence[-1] == train.word_to_index[
            train.sentence_end_token]:
            next_word_probs = model.forward_propagation(new_sentence)
            sampled_word = train.word_to_index[train.unknown_token]
            # don't sample unknown words
            while sampled_word == train.word_to_index[train.unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [train.index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str

    def count_syllables(self, line):
        vowels = "aeiouy"
        num_vowels = 0
        last_vowel = False
        for wc in line:
            found_vowel = False
            for v in vowels:
                if v == wc:
                    # don't count diphthongs
                    if not last_vowel:
                        num_vowels += 1
                    found_vowel = last_vowel = True
                    break
            if not found_vowel:
                last_vowel = False
        # remove es - usually silent
        if len(line) > 2 and line[-2:] == "es":
            num_vowels -= 1
        # remove silent e
        elif len(line) > 1 and line[-1:] == "e":
            num_vowels -= 1
        return num_vowels

    def add_lines(self):
        train = Train(self.vocabulary_size, self.hidden_dim,
                      self.learning_rate, self.nepoch, self.enable_training,
                      self.model_file, self.train_file)
        for i in range(self.num_sentences):
            sent = self.generate_sentence(train, train.model)
            new_sent = " ".join(sent)
            # strip out punctuation- usually not correct
            line = ''.join(ch for ch in new_sent if ch not in self.exclude)
            if len(line) != 0:
                # syllable count in haiku poems are 5-7-5
                if self.count_syllables(line) == 5 \
                        or self.count_syllables(line) == 7:
                    if self.count_syllables(line) == 5 \
                            and len(self.first) == 0:
                        self.first = line
                    elif self.count_syllables(line) == 5 \
                            and len(self.third) == 0:
                        self.third = line
                    elif len(self.second) == 0:
                        self.second = line
            # if the temp variables are full, add the poem and reset the temps
            if len(self.first) != 0 and len(self.second) != 0 \
                    and len(self.third) != 0:
                self.poems.append(
                    self.first + "\n" + self.second + "\n" + self.third + "\n")
                self.first = ""
                self.second = ""
                self.third = ""
        return self.poems
