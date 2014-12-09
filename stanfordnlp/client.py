"""client.py:
    serves as the wrapper
"""

from corenlp import *
import json
from pprint import pprint
'''
class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 2346)))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))
                
'''

if __name__ == "__main__":
    #test_doc = "Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas"
    test_file = 'data/sample-dep-sentences.txt'
    corenlp = StanfordCoreNLP()
    instances = corenlp.parse(test_file)
    pprint(instances[1].toJSON())


