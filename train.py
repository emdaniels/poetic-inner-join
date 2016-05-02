#! /usr/bin/env python
"""
Filename: train.py
Author: Denny Britz, Emily Daniels
Date: May 2016
Purpose: Trains the model on the text specified. This class is a
re-implementation of an RNN Theano example created by Denny Britz:
https://github.com/dennybritz/rnn-tutorial-rnnlm
"""
from __future__ import print_function
import csv
import itertools
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from rnn import RNN


class Train(object):

    def __init__(self, vocabulary_size, hidden_dim, learning_rate, nepoch,
                 enable_training, model_file, train_file):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.nepoch = nepoch
        self.enable_training = enable_training
        self.train_file = train_file
        self.model_file = model_file
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.sentences = []
        self.word_to_index = {}
        self.index_to_word = []
        self.tokenized_sentences = []
        self.split_sentences()
        self.tokenize_sentences()
        self.model = self.create_training_data()

    def train_with_sgd(self, model, X_train, y_train, learning_rate, nepoch,
                       evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                      (time, num_examples_seen, epoch, loss))
                # adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate *= 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                self.save_model_parameters("./data/rnn-%s-%d-%d-%s.npz" % (
                    os.path.basename(self.train_file), model.hidden_dim,
                    model.word_dim, time), model)
            for i in range(len(y_train)):
                model.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

    def save_model_parameters(self, outfile, model):
        U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
        np.savez(outfile, U=U, V=V, W=W)
        print("Saved model parameters to %s." % outfile)

    def load_model_parameters(self, path, model):
        npzfile = np.load(path)
        U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" %
              (path, U.shape[0], U.shape[1]))

    def split_sentences(self):
        print("Reading CSV file...")
        with open(self.train_file, 'rU') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # extra decoding to account for non UTF-8 characters
            self.sentences = itertools.chain(*[nltk.sent_tokenize(
                x[0].decode('latin-1').encode('utf-8').decode('utf-8').lower())
                                               for x in reader])
            self.sentences = ["%s %s %s" % (
            self.sentence_start_token, x, self.sentence_end_token) for x in
                              self.sentences]
        print("Parsed %d sentences." % (len(self.sentences)))

    def tokenize_sentences(self):
        # tokenize the sentences into words and count the word frequencies
        # get most common words, build index_to_word and word_to_index vectors
        self.tokenized_sentences = [nltk.word_tokenize(sent) for sent in
                                    self.sentences]
        word_freq = nltk.FreqDist(itertools.chain(*self.tokenized_sentences))
        print("Found %d unique word tokens." % len(word_freq.items()))

        vocab = word_freq.most_common(self.vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(self.unknown_token)
        self.word_to_index = dict(
            [(w, i) for i, w in enumerate(self.index_to_word)])

        print("Using vocabulary size %d." % self.vocabulary_size)
        print(
            "The least frequent word is '%s' appearing %d times." % (
            vocab[-1][0], vocab[-1][1]))

        # replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [
                w if w in self.word_to_index else self.unknown_token for w in
                sent]

    def create_training_data(self):
        X_train = np.asarray(
            [[self.word_to_index[w] for w in sent[:-1]] for sent in
             self.tokenized_sentences])
        y_train = np.asarray(
            [[self.word_to_index[w] for w in sent[1:]] for sent in
             self.tokenized_sentences])

        model = RNN(self.vocabulary_size, self.hidden_dim, 4)
        t1 = time.time()
        model.sgd_step(X_train[10], y_train[10], self.learning_rate)
        t2 = time.time()
        print("SGD step time: %f milliseconds" % ((t2 - t1) * 1000.))

        if self.model_file is not None:
            self.load_model_parameters(self.model_file, model)
        if self.enable_training:
            self.train_with_sgd(model, X_train, y_train, self.learning_rate,
                                self.nepoch)
        return model
