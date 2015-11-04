#!/usr/bin/python3

from random import choice, sample, randint, seed
from pickle import load, dump
from sqlite3 import connect
from os.path import exists
from sys import stdin, stdout, argv, stderr, exit
from time import asctime, time
import multiprocessing as mp
import Levenshtein
import json
import qae
import re

from logging import basicConfig, debug, info, warning, DEBUG, INFO, WARNING, ERROR, CRITICAL
basicConfig(stream=stderr, level=INFO)

test_questions, test_answers, test_problems, test_subjects = [], [], [], []
en = {}

P, A, AP = [], [], []
Q, W, WA = {}, {}, {}
mid, key = {}, {}
paths    = []
pa_cache, pp_cache = {}, {}
answer_set = set()
n_threads = 2
features = 256
#features = int(argv[1])
print('Number of features', features)
stdout.flush()
margin = 0.0001
learning_rate = 0.003
qae.update_n(learning_rate)
epochs = 100
max_questions = 1<<200
max_paraphrases = 1<<20
parentfolder = '../Data/'
question_set =          parentfolder + 'questions.txt'
paraphrase_set =        parentfolder + 'random_paraphrase-uniq.txt'
questions_load_file =   parentfolder + 'questions.dat'
answers_load_file =     parentfolder + 'paraphrases.dat'
words_load_file =       parentfolder + 'words.dat'
word_answer_load_file = parentfolder + 'word_answers.dat'
mid_key_file =          parentfolder + 'mid-en-key.txt'
answer_datafile =       parentfolder + 'answer.dat'
uniq_words_file =       parentfolder + 'unique_wordsJan-19-2015.txt'
uniq_paths_file =       parentfolder + 'unique_paths'
database =              parentfolder + 'Freebase.db'
paraphrases =           open(paraphrase_set)
initialize = True

max_W_index = 8000000
curr_W_index = 0

#Required to stop training once it's past a certain point
prev_paraphrase_precent = 0
prev_freebase_precent   = 0
train_freebase   = True
train_paraphrase = True
cur_paraphrases = 0

def saving_data():
    with open(questions_load_file, 'wb') as f:
        dump(Q, f)
    with open(answers_load_file, 'wb') as f:
        dump(P, f)
    with open(words_load_file, 'wb') as f:
        dump(W, f)
    with open(word_answer_load_file, 'wb') as f:
        dump(WA, f)

def loading_data():
    global Q, P, W, WA
    with open(questions_load_file, 'rb') as f:
        Q = load(f)
    with open(answers_load_file, 'rb') as f:
        P = load(f)
    with open(words_load_file, 'rb') as f:
        W = load(f)
    with open(word_answer_load_file, 'rb') as f:
        WA = load(f)
    print('%s Subjects' % len(Q))
    avg_n_answers = sum( [ len(v) for v in Q.values() ] ) / len(Q)
    print('%s Avg number of answers' % avg_n_answers)

def initialize_data():
    seed(time())
    global A, AP, Q, curr_W_index
    with open('skynet.conf') as f:
        line = f.readline().split('\t')
        curr_W_index = int( line[1] )
    AP = {x[0] for x in list(c.execute('SELECT DISTINCT B FROM Freebase'))}
    if exists(answer_datafile):
        A = load( open(answer_datafile, 'rb') )
    else:
        with open(mid_key_file) as mkf:
            for line in mkf:
                a = line.split()[1]
                A.append(a)
            dump( A, open(answer_datafile, 'wb') )
    #More than N questions necessary
#    with open(questions_load_file, 'rb') as f:
#        Q = load(f)
    i = 0
    max_questions = 20000000
    with open(question_set) as f:
        for line in f:
            i += 1
            if i == max_questions:
                break
            print('Processing question', i) if i % 100000 == 0 else False
            stdout.flush()
            line = line.strip('\n')
            items = line.split('\t')
            q = items[0].replace('?', '')
            a = items[1]
            answer_set.add(a)
            if len( items ) == 5:
               p = [ items[3] ]
            else:
                p = items[3:4]
            if q not in Q:
                Q[q] = [ (a, p) ]
            else:
                Q[q].append( (a, p) )
#    with open(questions_load_file, 'wb') as f:
#        dump(Q, f)
    with open(mid_key_file) as f:
        for line in f:
            m, k = line.split()
            mid[k] = m
            key[m] = k
    print('%s Subjects' % len(Q))
    avg_n_answers = sum( [ len(v) for v in Q.values() ] ) / len(Q)
    print('%s Avg number of answers' % avg_n_answers)
    print('opening paraphrases')
    initialize_paraphrases()

def initialize_paraphrases():
    global P, max_paraphrases, cur_paraphrases
    paraphrase_count, next_paraphrases = 0, 0
    P = []
    start_point = max_paraphrases
    max_paraphrases = start_point*2
    for line in paraphrases:
        paraphrase_count += 1
        if (paraphrase_count < start_point):
            continue
        if paraphrase_count >= max_paraphrases:
            break
        line = line.replace('?', '')
        P.append( line.split('\t') )

def next_paraphrases():
    global cur_paraphrases
    P = []
    paraphrase_count = 0
    cur_paraphrases += max_paraphrases
    for line in paraphrases:
        paraphrase_count += 1
        if (paraphrase_count < cur_paraphrases):
            continue
        if paraphrase_count >= cur_paraphrases:
            break
        line = line.replace('?', '')
        P.append( line.split('\t') )



def initialize_weights():
    global WA, W, curr_W_index
    print('counting words')
    uniq_word = open(uniq_words_file)
    uniq_path = open(uniq_paths_file)
    curr_W_index = 0
    for a in A:
        WA[a] = curr_W_index
        curr_W_index += 1
    for p in uniq_path:
        p = p.strip('\n')
        WA[p] = curr_W_index
        curr_W_index += 1
        paths.append(p)
    for q in uniq_word:
        for word in q.split():
            if word not in W:
                W[word] = curr_W_index
                curr_W_index += 1
    print('initializing weights')

    ###THIS ONE MAKES NEW RANDOM WEIGHTS
    #stdout.flush()
    #qae.initialize(max_W_index, features)
    qae.loadweights()

    print("curr_W_index, length W, length WA, features ", curr_W_index, len(W), len(WA), features)
    print('%s Words' % len(W))
    print('%s Answers' % len(A))
    print('%s Paraphrase Sets' % len(P))

def score(q, a):
    if a not in WA:
        warning('I have not seen ' + str(a) + ' before.')
        return -float('inf')
    #print([W[w] for w in q.split()], WA[a])
    ql = [W[w] for w in q.split()]
    a = WA[a]
    return qae.s(*([a] + ql))

def scoreparaphrase( q1, q2):
    ql1 = [W[w] for w in q1.split()]
    ql2 = [W[w] for w in q2.split()]
    return qae.sp(*(ql1 + [-1] + ql2))

def update_paraphrase( w1, w2):
    qae.update(W[w1], W[w2])

def update_dec_paraphrase( w1, w3):
    qae.update_dec(W[w1], W[w3])

def update( q, a):
    qae.update(W[q], WA[a])

def update_dec( q, a_):
    qae.update_dec(W[q], WA[a_])

def possible_answers( subj):
    if subj in pa_cache:
        return pa_cache[subj]
    pa = [x[0] for x in list(c.execute('SELECT C FROM Freebase where A=?', (subj,)))]
    pa += [x[0] for x in list(c.execute('SELECT A FROM Freebase where C=?', (subj,)))]
    pa = [key[x] for x in pa]
    pa_cache[subj] = pa
    return pa

def possible_paths(subj):
    global pp_cache
    if subj in pp_cache:
        return pp_cache[subj]
    pp = [x[0] for x in list(c.execute('SELECT B FROM Freebase where A=?', (subj,)))]
    pp += [x[0] for x in list(c.execute('SELECT B FROM Freebase where C=?', (subj,)))]
    pp_cache[subj] = pp
    return pp


def answer( ques, subj):
    '''
    Input: question
    Return: answer
    '''
    highest_value = -float('inf')
    highest_a = 'Bad Question'
    subj = mid[subj]
    pa = possible_answers(subj)

    for a in pa:
        if highest_value < score(ques, a):
            highest_value = score(ques, a)
            highest_a = a
    return highest_a

def learn(q, a, a_, updates, not_updates):
    if margin - score(q, a) + score(q, a_) > 0:
        updates += 1
        for word in q.split():
            update(word, a)
            update_dec(word, a_)
    else:
        not_updates += 1
    return updates,not_updates

def train(epoch):
    global prev_paraphrase_precent, train_paraphrase, \
           prev_freebase_precent,   train_freebase

    if (train_freebase == False and train_paraphrase == False):
        return

    updates = 0
    not_updates = 0
    updates1 = 0
    not_updates1 = 0
    print('Training Epoch', epoch, 'Time is', asctime())
    stdout.flush()
    
    for q in Q:
        if (train_freebase == True):
            s = q.split()[-1]
            s = mid[s]
            a  = choice( Q[ q ] )[0]
            aset = {a[0] for a in Q[q]}
            pa = set(possible_answers(s))
            wrong_answers = pa - aset
            if len(wrong_answers) <= 0:
                a_ = choice( A )
            else:
                a_ = sample( wrong_answers, 1 )[0]
            updates1,not_updates1 = learn(q, a, a_, updates1, not_updates1)

            paths = choice(Q[q])[1]
            for path in paths:
                pset = {path for p in Q[q]}
                pp = set(possible_paths(s))
                wrong_paths = pp - pset
                if len(wrong_paths) <= 0:
                    path_ = sample(AP, 1)[0]
                else:
                    path_ = sample(wrong_paths, 1)[0]
                updates1,not_updates1 = learn(q, path, path_, updates1, not_updates1)

        if (train_paraphrase == True):
            p1, p2 = choice(P)
            p_ = choice(P)[0]
            if margin - scoreparaphrase(p1, p2) + scoreparaphrase(p1, p_) > 0:
                updates += 1
                for word1 in p1.split():
                    for word2 in p2.split():
                        update_paraphrase(word1, word2)
                    for word2 in p_.split():
                        update_dec_paraphrase(word1, word2)
            else:
                not_updates += 1

    if (train_paraphrase == True):
        corr_perc = not_updates / (not_updates + updates)
        print('paraphrase correct percentage', corr_perc, updates, not_updates)
        if (corr_perc < prev_paraphrase_precent): #If precent correct has decreased
            #experimental
            print('Doing more paraphrases')
            next_paraphrases()
            train_freebase = True
            return
            #experimental
            #train_paraphrase = False
        else:
            prev_paraphrase_precent = corr_perc

    if (train_freebase == True):
        corr_perc = not_updates1 / (not_updates1 + updates1)
        print('freebase correct percentage', corr_perc, updates1, not_updates1)
        if (corr_perc < prev_freebase_precent): #If precent correct has decreased
            print("Finished training on freebase!")
            train_freebase = False
        else:
            prev_freebase_precent = corr_perc

'''
CHANGE INITIAL MEMORY ALLOCATION TO ALLOW FOR EXTRA WORD WEIGHT SLOTS FOR LATER USE
    In the future make it so the memory map is reallocated once it's out of free word slots

First: Get a question with the last word, using underscores instead of spaces, is the subject; optional answer column
Print out guess answer, if there's an answer column train against it
    If the subject is unknown add it to new answers and subjects
    If any of the words are unknown add it to new words
    Give the words, subject, answer slots from available word weights
'''

def Interactive():
    global curr_W_index
    for line in stdin:
        #Parse input line into variables question, answer
        line = line.strip()
        line = line.split('\t')
        question = line[0]
        answer = None
        if (len(line) > 1):
            answer = line[1]

        #Learn any unknown words
        for word in question.split(' ')[:-1]:
            if (W.get(word, None) == None):
                LearnWord(word)
        #Learn any unknown subjects
        subject = question.split(' ')[-1]
        subject_index = WA.get(subject, None)
        if (subject_index == None):
            LearnSubject(subject, answer, question)
        #Learn any unknown answers
        if (answer != None):
            answer_index = WA.get(answer, None)
            if (answer_index == None):
                LearnSubject(subject, answer, question)
        #Answer Question Tuple
        path_scores = {}
        freebase_mid = mid[ subject ]
        paths = [x[0] for x in list(c.execute('SELECT DISTINCT B FROM Freebase where A=?', (freebase_mid,)))]
        for path in paths:
            answers = [x[0] for x in list(c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, path)))]
            path_scores[path] = score_path(answers, question, path)

        debug(sorted(path_scores.items(), key=lambda x: x[1]), 'question', question, 'answer', answer, 'sub', subject)
        if path_scores == {}:
            answers = None
            warning('%s is a question that could not be processed, subject was %s and answer was %s' % (question, subject, answer))
        else:
            top_path = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)[0][0] #select only the key of the top path...
            debug("The top path was %s" % top_path)
            answers = [x[0] for x in list(c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, top_path)))]
            answers = [key[x] for x in answers]
            print('path scores',path_scores)
            print('top path',top_path)
            print('answers', answers)
        if answer is not None and answers is not None:
            print('Testing on new question - answer', question, answer)
            if answer not in answers:
                #train
                for a_ in answers:
                    print('Training on question - answer', question, a_)
                    learn(question, answer, a_, 0, 0)
            else:
                print('Correct')
        #Answer Again?
        else:
            continue


    #Train if Possible against Tuple when answer was wrong
    print("Finished")
    exit(0)
    return

def LearnWord(word):
    global curr_W_index
    curr_W_index += 1
    if (curr_W_index > max_W_index):
        Resize_W()
    W[word] = curr_W_index
    print('I now know the word', word, 'as', W[word])
    return

def LearnSubject(subj, answer, question):
    global curr_W_index
    curr_W_index += 1
    if (curr_W_index > max_W_index):
        Resize_W()
    WA[subj] = curr_W_index
    mid[subj] = ('g_' + ''.join( [ chr(randint(97,97+25)) for temp in range(7) ] ) )
    top_path_score = float('-inf')
    top_path       = None
    for p in paths:
        if (score(question, path) > top_path_score):
            top_path = p
    #c.execute('INSERT INTO Freebase SET A=?, B=? C=?, D=?', (subj, path, answer, 'True'))
    print('I now know the subject', subj, 'as', WA[subj], 'with path', top_path)
    return

def TextToPath(question):
    #Get list of paths
    #Score question against every path
    #return highest scored path
    return

def Resize_W():
    print("W IS TO SMALL, ERROR! ERROR! ERROR! THE BUGS ARE EVERYWHERE")
    exit(1)
    return

def mt_train():
    for x in range(epochs):
        train(x)
    #train(-1)
    #pool = mp.Pool(n_threads, maxtasksperchild=1)
    #pool.map(train, [x for x in range(epochs)])

def test():
    i = 0
    first_questions = list(Q.items())[:1000]
    #for q, ap_set in Q.items():
    for q, ap_set in first_questions:
        correct_answers = [x[0] for x in ap_set]
        s = q.split()[-1]
        a = answer(q, s)
        if a in correct_answers:
            i += 1
            print('Right answer:', a, 'question:', q, 'correct answers', ap_set, '--', s)
        else:
            print('Wrong answer:', a, 'question:', q, 'correct answers', ap_set, '--', s)
        print()
    print('%s right out of %s %s %% right on %s Epochs' % (i, len(Q), 100*i/len(Q), epochs))
    #print('%s right out of %s %s %% right on %s Epochs' % (i, len(Q), 100*i/100, epochs))

def multiscore( al, q):
    for i, a in enumerate(al):
        a = key[a]
        if a not in WA:
            warning('I have not seen' + str(a) + 'before.')
            return -float('inf')
        al[i] = WA[a]
    ql = [W[w] for w in q.split()]
    return qae.s_multianswer(*(al + [-1] + ql))

def spellcheck( sentence):
    return ' '.join( [ sorted( { Levenshtein.ratio(x, word):x for x in W }.items(), reverse=True)[0][1] for word in sentence.split() ] )

def fast_spellcheck( sentence):
    return ' '.join( [word for word in sentence.split() if word in W ] )

def score_path(candidate_answers, ques, path):
    path_score = 0
    answer_score = 0
    #sum
    for a in candidate_answers:
        a = key[a]
        answer_score += score(ques, a)
    #average
    answer_score = answer_score / len(candidate_answers)
    path_score = score(ques, path)
    #stdout.flush()
    #warning( '%s %s' % (answer_score, path_score))
    #return path_score
    #return answer_score
    #return answer_score + path_score
    #return answer_score + (path_score * .2)
    #return ( (answer_score / path_score) + (path_score / answer_score) ) / 2
    #return 2*answer_score*path_score / (path_score + answer_score)
    return abs(answer_score) + abs(path_score)

def multiscore_path(candidate_answers, ques):
    path_score = multiscore(candidate_answers, ques)
    return path_score

def best_answers():
    freebase_name_file = '../Data/mid-en-name.txt'
    freebase_name_datafile = '../Data/mid-en-name.dat'
    en = dict()
    if exists(freebase_name_datafile):
        info('Initializing freebase english names.')
        en = load( open(freebase_name_datafile, 'rb') )
    else:
        info('Initializing freebase english names... be patient... 1.2GB of names')
        with open(freebase_name_file) as f:
            for line in f:
                words = line.split()
                m, e = words[0], ' '.join(words[1:])
                en[m] = e
        dump(en, open(freebase_name_datafile, 'wb') )
    
    info('Loading WebQuestions')
    webquestions_test_file = '../Data/wq.txt'
    with open(webquestions_test_file) as f:
        d = json.loads( f.read() )
    for e in d:
        s = e['url']
        s = s.replace('http://www.freebase.com/view/en/', '')
        a = re.findall('"([^"]*)"', e['targetValue'])
        a = a + re.findall(' ([^\ ")]*)\)', e['targetValue'])
        q = e['utterance']
        q = q.strip('?\n')
        test_questions.append(q)
        for ans in a:
            test_answers.append(ans)
        test_problems.append((s, q, a))
        test_subjects.append(s)
    
    print(len(test_problems))
    for s, q, a in test_problems:
        path_scores = {}
        #spellcheck
        debug('You asked: %s' % q)
        #q = spellcheck(q)
        q = fast_spellcheck(q)
        debug('I heard: %s' % q)
        freebase_mid = mid[ s ]

        paths = [x[0] for x in list(c.execute('SELECT DISTINCT B FROM Freebase where A=?', (freebase_mid,)))]
        for path in paths:
            answers = [x[0] for x in list(c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, path)))]
            path_scores[path] = score_path(answers, q, path)
            #path_scores[path] = multiscore_path(answers, q)

        debug(sorted(path_scores.items(), key=lambda x: x[1]), 'question', q, 'answer', a, 'sub', s)
        if path_scores == {}:
            warning('%s is a question that could not be processed, subject was %s and answer was %s' % (q, s, a))
            english_answers = ['Bad Question']
        else:
            top_path = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)[0][0] #select only the key of the top path...
            debug("The top path was %s" % top_path)
            answers = [x[0] for x in list(c.execute('SELECT C FROM Freebase where A=? and B=?', (freebase_mid, top_path)))]
            english_answers  = [en[x] for x in answers]

        print('%s\t%s\t%s' % (q, json.dumps(a).replace('", "', '","'), json.dumps(english_answers).replace('", "', '","')) )
        stdout.flush()

conn = connect(database,check_same_thread=False)
c = conn.cursor()
initialize_data()
initialize_weights()
saving_data()
mt_train()
qae.saveweights()
#Interactive()
#loading_data()
#test()

best_answers()
