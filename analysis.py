import numpy as np
import codecs
import sklearn 
import nltk
from nltk.tree import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

#import matplotlib.pyplot as plt

def AttributeLines():
    f = codecs.open('debate_transcript.txt', 'r')
    lines = f.readlines()
    h = codecs.open('hillary_lines.txt', 'w')
    t = codecs.open('trump_lines.txt', 'w')
    m = codecs.open('moderator_lines.txt', 'w')
    count = 0
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        if line == ['']:
            continue
        if 'CLINTON:' in line:
            sentence = ''
            for word in line:
                if word == 'CLINTON:':
                    continue
                sentence = sentence + ' ' + word
                if word == 'Trump':
                    count += 1
            h.write(sentence)
            h.write('\n')
        elif 'TRUMP:' in line:
            sentence = ''
            for word in line:
                if word == 'TRUMP:':
                    continue
                sentence = sentence + ' ' + word
            t.write(sentence)
            t.write('\n')
        elif 'HOLT:' in line:
            sentence = ''
            for word in line:
                if word == 'HOLT:':
                    continue
                sentence = sentence + ' ' + word
            m.write(sentence)
            m.write('\n')
    print count

def AnalyzeLines(filename):
    h = codecs.open(filename, 'r')
    lines = h.readlines()
    named_entities = {}
    sentiment_nums = np.zeros(len(lines))    
    count = 0
    for line in lines:
        sentiment_nums[count] = FindSentiment(line)
        count += 1
        new_entities = FindNamedEntities(line)
        if new_entities != []:
            for entity in new_entities:
                if entity not in named_entities.keys():
                    named_entities[entity] = 1
                named_entities[entity] += 1
    sentiment_avg = np.mean(sentiment_nums)  
    return (named_entities, sentiment_avg)


def FindNamedEntities(line):
    prev = None
    continuous_chunk = []
    current_chunk = []
    chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(line)))
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

def FindSentiment(line):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(line)
    return ss["compound"]

if __name__ == "__main__": 
    # Scrape debate transcript for lines from the three players
    AttributeLines()
    trump_NE, trump_sent = AnalyzeLines('trump_lines.txt')
    hillary_NE, hillary_sent = AnalyzeLines('hillary_lines.txt')
    print "Hillary sent: " + str(hillary_sent)
    print "Trump sent: " + str(trump_sent)
    hillary_sorted_NE = sorted(hillary_NE.items(), key=operator.itemgetter(1)) 
    trump_sorted_NE = sorted(trump_NE.items(), key=operator.itemgetter(1)) 
    print hillary_sorted_NE[-1]
    print trump_sorted_NE[-1]
