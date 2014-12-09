import json
from collections import defaultdict
from constants import ROOT_FORM,ROOT_LEMMA,ROOT_POS,EMPTY


class Data():
    '''represent dota instance of one sentence'''
    
    current_sen = 1
    
    def __init__(self):
        self.tree = None
        self.coreference = None
        #self.dependency = []
        self.text = None
        self.tokens = []
        self.amr = None
        self.gold_graph = None
        self.sentID = self.current_sen
        
        
        self.tokens.append({'id':0,'form':ROOT_FORM,'lemma':ROOT_LEMMA,'pos':ROOT_POS,'ne':'O','rel':EMPTY})
        
    @staticmethod
    def newSen():
        Data.current_sen += 1  # won't be pickled
        #self.dependency.append([])
        #self.tokens.append([])

    def addTree( self, tree ):
        self.tree = tree
        
    def addText( self, sentence ):
        self.text = sentence
        
    def addToken( self, token, offset_begin, offset_end, lem, pop, ne ):
        tok_inst = {}
        tok_inst['id'] = len(self.tokens)
        tok_inst['form'] = token 
        #tok_inst['offset_begin'] = offset_begin
        #tok_inst['offset_end'] = offset_end
        tok_inst['lemma'] = lem
        tok_inst['pos'] = pop
        tok_inst['ne'] = ne
        tok_inst['rel'] = EMPTY
        self.tokens.append(tok_inst)

    def addCoref( self, coref_set):
        self.coreference = coref_set

    def addDependency( self, rel, l_token, r_token, l_index, r_index):
        '''CoNLL dependency format'''
        assert int(r_index) == self.tokens[int(r_index)]['id'] and int(l_index) == self.tokens[int(l_index)]['id']
        self.tokens[int(r_index)]['head'] = int(l_index)
        self.tokens[int(r_index)]['rel'] = rel
        
        
        #self.dependency[-1].append((rel, l_lemma, r_lemma, l_index, r_index))
    def addAMR(self,amr):
        self.amr = amr
        
    def addGoldGraph(self,gold_graph):
        self.gold_graph = gold_graph


    def get_ne_span(self,tags_to_merge):
        pre_ne_id = None
        ne_span_dict = defaultdict(list)
        for tok in self.tokens:
            if tok['ne'] in tags_to_merge:
                if pre_ne_id is None:
                    ne_span_dict[tok['id']].append(tok['id'])
                    pre_ne_id = tok['id']
                else:
                    ne_span_dict[pre_ne_id].append(tok['id'])
            else:
                pre_ne_id = None
        return ne_span_dict

    def toJSON(self):
        json = {}
        json['tree'] = self.tree
        json['coreference'] = self.coreference
        #json['dependency'] = self.dependency
        json['text'] = self.text
        json['tokens'] = self.tokens
        json['amr'] = self.amr
        return json

##    def find
