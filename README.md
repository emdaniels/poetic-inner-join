Poetic Inner Join
=================

If you were to take an author's works as the starting point for what they could 
potentially create if they continued to be influenced by their peers and their 
environment, it might sound something like the output of this program, where 
the words chosen reflect the author's inner voice as well as other influences on 
the author's work.

When relying on generative models to produce new work, it's worth remembering 
that the only data they contain is what has been created in the past. This makes 
it difficult to truly predict what someone would say next, to elaborate on what 
might be going on in an artist internally when they are creating something new 
and are part of the world. This program attempts to show how an artist's work 
could continue to evolve past the generative reworking of old words.

With this in mind, imagine stepping into a scene where Jane Austen, having 
dreamed of inivisible men, hybrid beings, and creatures from another world 
the night before, sits ruminating over the strange dream at breakfast. Rain 
falls softly outside, filling her with an irate sadness of not being able to go 
for a walk afterwards. She sits there, developing her thoughts, as you 
regenerate the production of a morning inside Austen's head. 

---
                   
I used a recurrent neural network (RNN) to build a language model of Austen's 
work to pick the most probable next sentence she would write, given her past 
work. The probability of a sentence is the product of probabilities of each 
word given the words that came before it. The sentence "Life seems but a quick 
succession of busy nothings." would be the probability of "nothings" given 
"Life seems but a quick succession", multiplied by the probability of "busy" 
given "Life seems", and so on.

The way the RNN works is, given an existing sequence of words, we sample a next 
word from the predicted probabilities, and repeat the process until there is a 
full sentence. Context is often lost between sentences, so I chose to create 
haiku poems from the Austen-like sentences generated to tie the pieces together 
more easily without having them have to make sense all the time.

There are some excellent resources that go into greater detail on the process of
creating and implementing RNNs that I highly recommend:

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

---

After generating Austen-like sentences and ensuring they fit into the 5-7-5 
syllable structure of a haiku, the poems go through an evaluation process where 
internal and external influences are used to filter them. By building a map of 
words with associations to eight basic emotions (anger, fear, anticipation, 
trust, surprise, sadness, joy, and disgust), only the poems that contain words 
that correspond with the feelings I've described are kept. 

The external influence of an additional author here serves as a way to look at 
the text with a different set of goggles on, where a variant of a [Naive Bayes 
classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is used to 
predict whether the text of poem falls more easily into the group of work from 
Austen or Wells. Since we already know that the sentences generated are based on 
Austen's work, this filter is used to find poems that could also fall easily 
into H.G. Wells body of work, meaning it is highly likely that both Austen or 
Wells could have written such a group of sentences.

These influences represent only a fraction of what could be built to shape the 
raw output of a neural network, providing a good starting point to create what 
could be seen as an overseer of unstructured thought.



References
==========

Project Gutenberg
-----------------

All associated files of various formats for Jane Austen's works:

  - [Sense and Sensibility](http://www.gutenberg.org/ebooks/161)
  - [Pride and Prejudice](http://www.gutenberg.org/ebooks/1342)
  - [Persuasion](http://www.gutenberg.org/ebooks/105)
  - [Northanger Abbey](http://www.gutenberg.org/ebooks/121)
  - [Mansfield Park](http://www.gutenberg.org/ebooks/141)
  - [Lady Susan](http://www.gutenberg.org/ebooks/946)
  - [Emma](http://www.gutenberg.org/ebooks/158)


All associated files of various formats for H.G. Wells' works:

  - [The War of the Worlds](http://www.gutenberg.org/ebooks/36)
  - [The Time Machine](http://www.gutenberg.org/ebooks/35)
  - [The Invisible Man: A Grotesque Romance](http://www.gutenberg.org/ebooks/5230)
  - [The Red Room](http://www.gutenberg.org/ebooks/23218)
  - [The Island of Doctor Moreau](http://www.gutenberg.org/ebooks/159)
  - [The First Men in the Moon](http://www.gutenberg.org/ebooks/1013)
  - [The Door in the Wall, and Other Stories](http://www.gutenberg.org/ebooks/456)


These eBooks are for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with these eBooks or online at www.gutenberg.org


NRC Emotion Lexicon
-------------------

A list of English words and their associations with eight basic emotions and two 
sentiments. The annotations were manually done by crowdsourcing. 
Version 0.92 is the latest version as of 10 July 2011 and can be found at:

http://saifmohammad.com/WebPages/lexicons.html


License
=======

Copyright 2016 Emily Daniels

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.