#!/usr/bin/python3

import json
import re
import difflib
import random
import sqlite3
import sys
import json
import numpy as np
import pickle
import Levenshtein
import os.path

from logging import basicConfig, debug, info, warning, DEBUG, INFO, WARNING, ERROR, CRITICAL
basicConfig(stream=sys.stderr, level=INFO)

import qae


class WebQuestion():
    def __init__(self):
        self.webquestions_test_file = '../Data/wq.txt'
        #self.webquestions_test_file = '../Preprocess/webquestions.examples.train.json'
        self.freebase_key_file = '../Data/mid-en-key.txt'
        self.freebase_name_file = '../Data/mid-en-name.txt'
        self.freebase_name_datafile = '../Data/mid-en-name.dat'
        self.word_weight_index_datafile = '../Data/words.dat'
        self.answer_weight_index_datafile = '../Data/word_answers.dat'
        conn = sqlite3.connect('../Data/Freebase.db')
        self.c = conn.cursor()
        self.questions, self.answers, self.problems, self.subjects = [], [], [], []
        self.loading_webquestions()
        self.initialize_freebase_key_mid()
        self.initialize_freebase_names()
        with open(self.word_weight_index_datafile, 'rb') as f:
            self.W = pickle.load(f)
        with open(self.answer_weight_index_datafile, 'rb') as f:
            self.WA = pickle.load(f)
        info("Attempting to load weights.")
        qae.loadweights()
        info("Loaded weights.")

    def loading_webquestions(self):
        info('Loading WebQuestions')
        with open(self.webquestions_test_file) as f:
            d = json.loads( f.read() )
        for e in d:
            s = e['url']
            s = s.replace('http://www.freebase.com/view/en/', '')
            a = re.findall('"([^"]*)"', e['targetValue'])
            a = a + re.findall(' ([^\ ")]*)\)', e['targetValue'])
            q = e['utterance']
            q = q.strip('?\n')
            self.questions.append(q)
            for ans in a:
                self.answers.append(ans)
            self.problems.append((s, q, a))
            self.subjects.append(s)

    def initialize_freebase_key_mid(self):
        info('Initializing mid - key map')
        #only 150 MB
        self.mid = dict()
        self.key = dict()
        with open(self.freebase_key_file) as f:
            for line in f:
                m, e = line.split()
                self.mid[e] = m
                self.key[m] = e

    def initialize_freebase_names(self):
        #takes a while 1.2GB
        if os.path.exists(self.freebase_name_datafile):
            info('Initializing freebase english names.')
            self.en = pickle.load( open(self.freebase_name_datafile, 'rb') )
        else:
            info('Initializing freebase english names... be patient... 1.2GB of names')
            self.en = dict()
            with open(self.freebase_name_file) as f:
                for line in f:
                    words = line.split()
                    m, e = words[0], ' '.join(words[1:])
                    self.en[m] = e
            pickle.dump(self.en, open(self.freebase_name_datafile, 'wb') )

    def score(self, q, a, path=False):
        if not path:
            a = self.key[a]
        if a not in self.WA:
            warning('I have not seen' + str(a) + 'before.')
            return -float('inf')
        #print([self.W[w] for w in q.split()], self.WA[a])
        ql = [self.W[w] for w in q.split()]
        a = self.WA[a]
        return qae.s(*([a] + ql))

    def multiscore(self, al, q):
        for i, a in enumerate(al):
            a = self.key[a]
            if a not in self.WA:
                warning('I have not seen' + str(a) + 'before.')
                return -float('inf')
            al[i] = self.WA[a]
        ql = [self.W[w] for w in q.split()]
        return qae.s_multianswer(*(al + [-1] + ql))

    def spellcheck(self, sentence):
        return ' '.join( [ sorted( { Levenshtein.ratio(x, word):x for x in self.W }.items(), reverse=True)[0][1] for word in sentence.split() ] )

    def fast_spellcheck(self, sentence):
        return ' '.join( [word for word in sentence.split() if word in self.W ] )

    def score_path(self, candidate_answers, ques, path):
        path_score = 0
        answer_score = 0
        #sum
        for a in candidate_answers:
            answer_score += self.score(ques, a)
        #average
        answer_score = answer_score / len(candidate_answers)
        path_score = self.score(ques, path, path=True)
        sys.stdout.flush()
        #print( 'question is:%s\n' % (ques))
        #print( 'answers are:%s\n\n' % ([self.en[x] for x in candidate_answers]))
        #print( 'path is:%s\n\n' % (path))
        #print( 'avg answer score is:%s path score is:%s' % (answer_score, path_score))
        #print( '-----------------------\n\n\n\n')
        sys.stdout.flush()
        #return path_score
        #return answer_score
        #return answer_score + path_score
        #return answer_score + (path_score * .2)
        #return ( (answer_score / path_score) + (path_score / answer_score) ) / 2
        #return 2*answer_score*path_score / (path_score + answer_score)
        return answer_score + path_score

    def multiscore_path(self, candidate_answers, ques):
        path_score = self.multiscore(candidate_answers, ques)
        return path_score

    def best_answers(self):
        for s, q, a in self.problems:
            path_scores = {}
            #spellcheck
            debug('You asked: %s' % q)
            #q = self.spellcheck(q)
            q = self.fast_spellcheck(q)
            debug('I heard: %s' % q)
            freebase_mid = self.mid[ s ]


##            #A(q) = C1
#            C1 = [x[0] for x in list(self.c.execute('SELECT DISTINCT C FROM Freebase where A=?', (freebase_mid,)))]
#            #C1 += [x[0] for x in list(self.c.execute('SELECT DISTINCT A FROM Freebase where C=?', (freebase_mid,)))]
#            highest_score = float('-inf')
#            best_answer = 'Bad Question'
#            for answer in C1:
#                answer_score = self.score(q, freebase_mid)
#                if answer_score > highest_score:
#                    highest_score = answer_score
#                    best_answer = answer
#
#            if best_answer == 'Bad Question':
#                english_answers = 'Bad Question'
#                print('%s\t%s\t[%s]' % (q, json.dumps(a).replace('", "', '","'), json.dumps(english_answers).replace('", "', '","')) )
#                continue
#            english_answers  = self.en[best_answer]
#            print('%s\t%s\t[%s]' % (q, json.dumps(a).replace('", "', '","'), json.dumps(english_answers).replace('", "', '","')) )
#            sys.stdout.flush()
#

#            PATH SCORING by answer, not path
            paths = [x[0] for x in list(self.c.execute('SELECT DISTINCT B FROM Freebase where A=?', (freebase_mid,)))]
            for path in paths:
                answers = [x[0] for x in list(self.c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, path)))]
                path_scores[path] = self.score_path(answers, q, path)
                #path_scores[path] = self.multiscore_path(answers, q)

            debug(sorted(path_scores.items(), key=lambda x: x[1]), 'question', q, 'answer', a, 'sub', s)
            if path_scores == {}:
                warning('%s is a question that could not be processed, subject was %s and answer was %s' % (q, s, a))
                english_answers = ['Bad Question']
            else:
                top_path = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)[0][0] #select only the key of the top path...
                debug("The top path was %s" % top_path)
                answers = [x[0] for x in list(self.c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, top_path)))]
                english_answers  = [self.en[x] for x in answers]

            print('%s\t%s\t%s' % (q, json.dumps(a).replace('", "', '","'), json.dumps(english_answers).replace('", "', '","')) )
            sys.stdout.flush()

wq = WebQuestion()
wq.best_answers()
