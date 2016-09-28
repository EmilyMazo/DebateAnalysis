import numpy as np
import codecs
import sklearn 
import nltk
from nltk.tree import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator
import sys  
import matplotlib.pyplot as plt

reload(sys)  
sys.setdefaultencoding('utf8')


def AttributeLines():
    ''' This method separates the debate transcript raw text into three files by the speaker of the line
    (Hillary Clinton, Donald Trump, or Lester Holt (Moderator) and the "line number," which is the count 
        of the statement in the NYT transcript.
    In this method I also keep track of when speakers are interrupted (lovingly annotated by the NYT as '...')
    and who was interrupted. '''

    f = codecs.open('debate_transcript.txt', 'r')
    lines = f.readlines()
    h = codecs.open('hillary_lines.txt', 'w')
    t = codecs.open('trump_lines.txt', 'w')
    m = codecs.open('moderator_lines.txt', 'w')
    i = codecs.open('interruption_counter.txt', 'w')
    count = 0
    for line in lines:
        count += 1
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
            h.write(str(count) + '\t' + sentence)
            h.write('\n')
            if '...' in line:
                i.write(str(count) + '\t' + 'h')
                i.write('\n')
        elif 'TRUMP:' in line:
            sentence = ''
            for word in line:
                if word == 'TRUMP:':
                    continue
                sentence = sentence + ' ' + word
            t.write(str(count) + '\t' + sentence)
            t.write('\n')
            if '...' in line:
                i.write(str(count) + '\t' + 't')
                i.write('\n')
        elif 'HOLT:' in line:
            sentence = ''
            for word in line:
                if word == 'HOLT:':
                    continue
                sentence = sentence + ' ' + word
            m.write(str(count) + '\t' + sentence)
            m.write('\n')
            if '...' in line:
                i.write(str(count) + '\t' + 'm')
                i.write('\n')

        

def AnalyzeLines(filename):
    ''' This method splits a candidate's text into statements (as separated by the NYT with newlines)
    and finds the sentiment score (according to the VADER sentiment analyzer model) of each, and averages them
    as well as stores them with their line number (for plotting). This method also extracts the named named entities 
    of each statement.'''
    h = codecs.open(filename, 'r')
    lines = h.readlines()
    named_entities = {}
    sentiment_nums = np.zeros(len(lines))
    sentiment_by_line_count = []  
    count = 0
    for line in lines:
        line = line.split('\t')
        line_num = line[0]
        sentence = line[1]
        sentiment_nums[count] = FindSentiment(sentence)
        sentiment_by_line_count.append([line_num, sentiment_nums[count]])
        count += 1
        new_entities = FindNamedEntities(sentence)
        if new_entities != []:
            for entity in new_entities:
                if entity not in named_entities.keys():
                    named_entities[entity] = 1
                named_entities[entity] += 1
    sentiment_avg = np.mean(sentiment_nums)  
    return (named_entities, sentiment_avg, sentiment_by_line_count)


def FindNamedEntities(line):
    ''' This method uses NLTK's built-in chunker and POS-tagger to extract named entities '''
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
    ''' This method uses VADER's trained sentiment analysis model to create compound sentiment scores 
    for each line. '''
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(line)
    return ss["compound"]


def DrawSentGraphs(sents1, sents2):
    '''This method plots the compound sentiment score of each statement of each candidate, 
    over the time point in the debate which is measured by the number of statements 
    (as annotated by the NYT in their debate transcript) that have been spoken by all
    members of the debate (candidates and moderator).
    Also the NYT helpfully annotated all interruptions with elipses ('...'), 
    so I kept track of that too. Those are plotted as vertical black lines by time point.'''
    # Keep track of interruptions (line counts)
    i = codecs.open('interruption_counter.txt', 'r')
    interruption_line_numbers = i.readlines()
    # Separate line number and sentiment score (compound) data into X and Y coordinate data point lists
    sents1_x = []
    sents2_x = []
    sents1_y = []
    sents2_y = []
    for i in range(len(sents1)):
        sents1_x.append(sents1[i][0])
        sents1_y.append(sents1[i][1])
    for j in range(len(sents2)):
        sents2_x.append(sents2[j][0])
        sents2_y.append(sents2[j][1])
    # Plot data
    plt.ylim(-2.0, 2.0)
    plt.plot(sents1_x, sents1_y, 'blue')
    plt.plot(sents2_x, sents2_y, 'red')
    for interruption in interruption_line_numbers:
        interruption = interruption.strip()
        interruption = interruption.split('\t')
        interruption[0] = int(interruption[0])
        if interruption[1] == 'h':
            plt.plot((interruption[0], interruption[0]), (-2.0, 2.0), 'blue')
        elif interruption[1] == 't':
            plt.plot((interruption[0], interruption[0]), (-2.0, 2.0), 'red')
        elif interruption[1] == 'm':
            plt.plot((interruption[0], interruption[0]), (-2.0, 2.0), 'black')
    # Add metadata to plot
    plt.xlabel("Time Point in Debate")
    plt.ylabel("Compound Sentiment Score")
    plt.title("Sentiment Score for Candidate Statements Over the Debate Timeline")
    # Show plot
    plt.show()


if __name__ == "__main__": 
    # Scrape debate transcript for lines from the three players
    AttributeLines()
    # Extract sentiment and NE data for each speaker
    trump_NE, trump_sent, trump_sent_nums = AnalyzeLines('trump_lines.txt')
    hillary_NE, hillary_sent, hillary_sent_nums = AnalyzeLines('hillary_lines.txt')
    print "Hillary sent: " + str(hillary_sent)
    print "Trump sent: " + str(trump_sent)
    hillary_sorted_NE = sorted(hillary_NE.items(), key=operator.itemgetter(1)) 
    trump_sorted_NE = sorted(trump_NE.items(), key=operator.itemgetter(1)) 
    print hillary_sorted_NE[-1]
    print trump_sorted_NE[-1]
    # Plot sentiment vs. time for each speaker
    DrawSentGraphs(hillary_sent_nums, trump_sent_nums)
