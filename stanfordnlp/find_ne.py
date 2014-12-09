"""client.py:
    serves as the wrapper
"""
import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 2346)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))

#nlp = StanfordNLP()
#result = nlp.parse("Stanford University is located in Irvine, California. It is a great university, founded in 1891. It used to be a bad university. I need to go to grocery store. I want some vegetables.")
#pprint(result)

##from nltk.tree import Tree
##tree = Tree.parse(result['sentences'][0]['parsetree'])
##pprint(tree)

def find_ne(doc):
    nlp = StanfordNLP()
    result = nlp.parse(doc)
    tokens = result['tokens']
    entities = []
    #import pdb
    #pdb.set_trace()
    for sentence in tokens:
        flag = False
        for token in sentence:
            if token[5] != 'O' and flag == False:
                #first word of NE
                entity = [token]
                flag = True
            elif token[5]!= 'O' and flag == True:
                #other word of NE
                entity.append(token)
            elif token[5]=='O' and flag == True:
                #NE ends
                entities.append(entity)
                flag = False
            else:
                pass
        #the NE is the last word
        if flag == True:
            entities.append(entity)
    
    return entities
                


if __name__ == "__main__":
    test_doc = "Stanford University is located in Irvine, California. It is a great university, founded in 1891. It used to be a bad university. I need to go to grocery store. I want some vegetables."
    entities = find_ne(test_doc)
    pprint(entities)


