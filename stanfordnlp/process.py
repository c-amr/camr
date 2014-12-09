import json, re
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 2346)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))

nlp = StanfordNLP()
text = "He left the office just before the lunch."

pprint(nlp.parse(text))
##from nltk.tree import Tree
##tree = Tree.parse(result['sentences'][0]['parsetree'])
##pprint(tree)

##m = re.match(r'(?P<lemma>.+)-(?P<index>\d+)', 'asdfdsf-3')
##print m.group('lemma')
##print m.group('index')
