"""util.py:
   tools to process the documents
"""
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from find_ne import find_ne
from pprint import pprint
import re
import string

def normalize(s):
    for p in string.punctuation:
        s = s.replace(p,'')
    words = word_tokenize(s)
    return [w.lower() for w in words]

def find_abr_fn(sent,query):
    """search for the query's(if the query is an abbreviation) fullname within one sentence."""
    i = 0
    j = 0

    while i < len(sent) and i + len(query) - 1 < len(sent):
        if sent[i].istitle() and sent[i][0] == query[j]:
#            import pdb
#            pdb.set_trace()
            k = 1
            m = 1
            fullname = [sent[i]]
            while m < len(query):
                if sent[i+k].istitle() and sent[i+k][0] == query[j+m]:
                    fullname.append(sent[i+k])
                    k+=1
                    m+=1
                elif sent[i+k] in ['of','in']:
                    fullname.append(sent[i+k])
                    k+=1
                else:
                    i += k
                    break
            if m == len(query):
                return fullname
            else:
                i+=1
        i+=1

    return -1


def find_abr_fullname(doc,query,Num):
    """Find the query(abbreviation's) full name within the document.
       Parameters:
       doc: the document to be searched for(specified format) 
       query: the abbreviation
       Num: the number of sentences before the query to be looked for fullname
       (here we asume that all the fullname of the query appeared before the query)
    """
    sents = [word_tokenize(t) for t in sent_tokenize(doc)]
    for i,sent in enumerate(sents):
        if query in sent:
            fullname = find_abr_fn(sent,query)
            if fullname != -1:
                return fullname
            else:
                j = 1
                while i-j >= 0 and j <= Num: 
                    if find_abr_fn(sent[i-j],query) == -1:
                        j+=1
                    else:
                        return find_abr_fn(sent[i-j],query)
                
    raise Exception('No query in the document.')


def find_query_fullname(doc,query):
    """Find the query's(no abbreviation) fullname within the doc's entity list
       given by standford NER.
    """
    entities = find_ne(doc)
    #pprint(entities)
    q = normalize(query)
    print q
    result = []
    for tok in q:
        for en in entities:
            fullname = [r[0].lower() for r in en]
        #        print fullname
            fn_set = set(fullname)
            if tok in fn_set:
                result.append(fullname)
    return result

if __name__== "__main__":    
    test_f = open('support_doc.txt','r')
    test_doc = test_f.read()
    test_abr = "CANA"
    test_query = "Columbia"
    #for l in test_f.readlines():
    #    if l != '<P>\n' and l != '</P>\n':
    #        print l
    #        if l[-2:] != '-\n':
    #            test_doc += l.strip()+' '
    #        else:
    #            test_doc += l.strip()
    test_doc = re.sub('<[^<]+?>','',test_doc)
    test_doc = re.sub('[\r\n]+',' ',test_doc)
    print test_doc

    print "Example 1: find full name for CANA, abbreviation stuff..."
    print find_abr_fullname(test_doc,test_abr,2)

    print "find query full name for Columbia, non-abbr..."
    print find_query_fullname(test_doc,test_query)
