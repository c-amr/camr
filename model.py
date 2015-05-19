#!/usr/bin/python

import bz2,contextlib
import numpy as np
import sys
import json
import cPickle as pickle
#import simplejson as json
from constants import *
from common.util import Alphabet,ETag,ConstTag
import importlib
from collections import defaultdict
 
_FEATURE_TEMPLATES_FILE = './feature/basic_feats_joint.templates'

class Model():
    """weights and templates"""
    #weight = None
    #n_class = None
    #n_rel = None
    #n_tag = None
    indent = " "*4
    #feature_codebook = None
    #class_codebook = None
    #feats_generator = None 
    def __init__(self,elog=sys.stdout):
        self.elog = elog
        self.weight = None
        self.aux_weight = None
        self.avg_weight = None # for store the averaged weights
        #self.n_class = n_class
        #self.n_rel = n_rel
        #self.n_tag = n_tag
        self._feats_templates_file = None
        self._feature_templates_list = []
        self._feats_gen_filename = None
        self.feats_generator = None
        self.token_to_concept_table = defaultdict(set)
        self.pp_count_dict = defaultdict(int)
        self.total_num_words = 0
        self.token_label_set = defaultdict(set)
        self.class_codebook = None
        self.feature_codebook = None
        self.rel_codebook = Alphabet()
        self.tag_codebook = {
            'Concept':Alphabet(),
            'ETag':Alphabet(),
            'ConstTag':Alphabet()
        }
        
    def setup(self,action_type,instances,feature_templates_file=None):

        self._feats_templates_file = feature_templates_file
        self.class_codebook = Alphabet.from_dict(dict((i,k) for i,(k,v) in enumerate(ACTION_TYPE_TABLE[action_type])),True)
        self.feature_codebook = dict([(i,Alphabet()) for i in self.class_codebook._index_to_label.keys()])
        self.read_templates()
        
        n_rel,n_tag = self._set_rel_tag_codebooks(instances)
        self._set_class_weight(self.class_codebook.size(),n_rel)
        self._set_statistics(instances)
        self.output_feature_generator()

    def _set_statistics(self,instances):
        #pp_count_dict = defaultdict(int)
        for inst in instances:
            sent = inst.tokens
            self.total_num_words += len(sent)
            for token in sent:
                if token['pos'] == 'IN' and token['rel'] == 'prep':
                    self.pp_count_dict[token['form'].lower()] += 1
    
    def _set_rel_tag_codebooks(self,instances):
        #TODO
        self.rel_codebook.add(NULL_EDGE)
        #self.tag_codebook['Concept'].add(NULL_TAG)

        for inst in instances:
            gold_graph = inst.gold_graph
            gold_nodes = gold_graph.nodes
            #gold_edges = gold_graph.edges 
            sent_tokens = inst.tokens
            for g,d in gold_graph.tuples():
                gnode = gold_nodes[g]
                g_span_wds = [tok['lemma'] for tok in sent_tokens if tok['id'] in range(gnode.start,gnode.end)] if g!= 'x' else gnode.words
                g_span_ne = sent_tokens[g]['ne']
                g_entity_tag = gold_graph.get_node_tag(g)
                #if len(g_span_wds) > 1:  
                    #for gwd in g_span_wds:
                    #    self.token_to_concept_table[gwd].add(g_entity_tag)
                if g_span_ne not in ['O','NUMBER']:                    
                    self.token_to_concept_table[g_span_ne].add(g_entity_tag)
                self.token_to_concept_table[','.join(g_span_wds)].add(g_entity_tag)
                if isinstance(g_entity_tag,ETag):
                    self.tag_codebook['ETag'].add(g_entity_tag)
                elif isinstance(g_entity_tag,ConstTag):
                    self.tag_codebook['ConstTag'].add(g_entity_tag)
                else:
                    self.tag_codebook['Concept'].add(g_entity_tag)

                dnode = gold_nodes[d]
                d_span_wds = [tok['lemma'] for tok in sent_tokens if tok['id'] in range(dnode.start,dnode.end)] if d != 'x' else dnode.words
                d_span_ne = sent_tokens[d]['ne']
                d_entity_tag = gold_graph.get_node_tag(d)
                #if len(d_span_wds) > 1:  
                #    for dwd in d_span_wds:
                #        self.token_to_concept_table[dwd].add(d_entity_tag)
                if d_span_ne not in ['O','NUMBER']:                    
                    self.token_to_concept_table[d_span_ne].add(d_entity_tag)
                self.token_to_concept_table[','.join(d_span_wds)].add(d_entity_tag)
                
                if isinstance(d_entity_tag,ETag):
                    self.tag_codebook['ETag'].add(d_entity_tag)
                elif isinstance(d_entity_tag,ConstTag):
                    self.tag_codebook['ConstTag'].add(d_entity_tag)
                else:
                    self.tag_codebook['Concept'].add(d_entity_tag)
                #self.tag_codebook.add(d_entity_tag)

                g_edge_label = gold_graph.get_edge_label(g,d)
                if g_span_ne not in ['O','NUMBER']:                    
                    self.token_label_set[g_span_ne].add(g_edge_label)
                self.token_label_set[','.join(g_span_wds)].add(g_edge_label)
                self.rel_codebook.add(g_edge_label)
                
        n_rel = [1]*self.class_codebook.size()
        n_tag = [1]*self.class_codebook.size()

        for k,v in self.class_codebook._index_to_label.items():
            if v in ACTION_WITH_TAG:
                n_tag[k] = reduce(lambda x,y: x+y, map(lambda z: self.tag_codebook[z].size(), self.tag_codebook.keys()))
            if v in ACTION_WITH_EDGE:
                n_rel[k] = self.rel_codebook.size()
        return n_rel,n_tag
        
    def _set_class_weight(self,n_class,n_rel=None,init_feature_dim = 5*10**5):
        
        #if n_rel == None:
        #    n_rel = [1]*n_class
        #assert len(n_rel) == n_class

        
        #self.weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.aux_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]
        #self.avg_weight = [np.zeros(shape = (init_feature_dim,nt,nr),dtype=WEIGHT_DTYPE) for nr,nt in zip(n_rel,n_tag)]

        self.weight = [np.zeros(shape = (init_feature_dim,nr),dtype=WEIGHT_DTYPE) for nr in n_rel]
        self.aux_weight = [np.zeros(shape = (init_feature_dim,nr),dtype=WEIGHT_DTYPE) for nr in n_rel]
        self.avg_weight = [np.zeros(shape = (init_feature_dim,nr),dtype=WEIGHT_DTYPE) for nr in n_rel]

    
    def read_templates(self): 

        ff_name = self._feats_templates_file if self._feats_templates_file else _FEATURE_TEMPLATES_FILE
        for line in open(ff_name,'r'):
            line = line.strip()
            if not line:
                pass
            elif line.startswith('#'):
                pass
            else:
                elements = line.split()
                #elements.extend(['tx'])
                template = "'%s=%s' %% (%s)"%('&'.join(elements),'%s_'*len(elements),','.join(elements))
                self._feature_templates_list.append((template,elements))

    def output_feature_generator(self):
        """based on feature autoeval method in (Huang,2010)'s parser"""
            
        import time
        self._feats_gen_filename = 'feats_gen_'+self._feats_templates_file.split('/')[-1].split('.')[0] #str(int(time.time()))
        output = open('./temp/'+self._feats_gen_filename+'.py','w')
        
        output.write('#generated by model.py\n')
        output.write('from constants import *\n')
        output.write('def generate_features(state,action):\n')
        output.write(Model.indent+'s0,b0,a0=state.get_feature_context_window(action)\n')
        
        element_set = set([])
        definition_str = Model.indent+'feats=[]\n'
        append_feats_str = ''

        definition_str += Model.indent+"act_idx = state.model.class_codebook.get_index(action['type'])\n"
        definition_str += Model.indent+"tx = action['tag'] if 'tag' in action else EMPTY\n"
        #definition_str += Model.indent+"lx = action['edge_label'] if 'edge_label' in action else EMPTY\n"
        #definition_str += Model.indent+"print state.model.class_codebook._label_to_index\n"
        #print self._feature_templates_list
        
        for template,elements in self._feature_templates_list:

            for e in elements: # definition
                if e not in element_set:
                    sub_elements = e.split('_')
                    if len(sub_elements) == 2:
                        definition_str += "%s%s=%s['%s'] if %s else EMPTY\n" % (Model.indent,e,sub_elements[0],FEATS_ABBR[sub_elements[1]],sub_elements[0])
                    elif len(sub_elements) == 3:
                        definition_str += "%s%s=%s['%s']['%s'] if %s and %s['%s'] else EMPTY\n" % (Model.indent,e,sub_elements[0],FEATS_ABBR[sub_elements[1]],FEATS_ABBR[sub_elements[2]],sub_elements[0],sub_elements[0],FEATS_ABBR[sub_elements[1]])
                    else:
                        pass
                    element_set.add(e)
                else:
                    pass

            
            #append_feats_str += "%sif [%s] != %s*[None]:feats.append(%s)\n" % (Model.indent,','.join(elements),len(elements),template)
            append_feats_str += "%sfeats.append(%s)\n" % (Model.indent,template)
            #append_feats_str += "%sstate.model.feature_codebook[act_idx].add(%s)\n" % (Model.indent,template)
        #definition_str += "%seqpfx=s0['lemma'][:4]==tx[:4] if tx is not EMPTY and len(s0['lemma'])>4 else EMPTY\n"%(Model.indent)
        #append_feats_str += "%sfeats.append('s0_len&eqpfx=%%s_%%s_'%%(s0['len'],eqpfx))\n"%(Model.indent)
        #definition_str += "%spath=[(state.sent[i]['pos'],state.sent[i]['rel']) for i in state.deptree.get_path(b0['id'],a0['id'])] if b0 and a0 else EMPTY\n"%(Model.indent)
        definition_str += "%sdist1=abs(s0['id']-b0['id']) if b0 else EMPTY\n"%(Model.indent)
        definition_str += "%sif dist1 > 10: dist1=10\n"%(Model.indent)
        definition_str += "%sdist2=abs(a0['id']-b0['id']) if b0 and a0 else EMPTY\n"%(Model.indent)
        definition_str += "%sif dist2 > 10: dist2=10\n"%(Model.indent)
        output.write(definition_str)
        output.write(append_feats_str)
        
        output.write('%sreturn feats' % Model.indent)
        output.close()
        
        #sys.path.append('/temp/')
        print "Importing feature generator!"
        self.feats_generator = importlib.import_module('temp.'+self._feats_gen_filename).generate_features

    def toJSON(self):
        print 'Converting model to JSON'
        print 'class size: %s \nrelation size: %s \ntag size: %s'%(self.class_codebook.size(),self.rel_codebook.size(),map(lambda x:'%s->%s '%(x,self.tag_codebook[x].size()),self.tag_codebook.keys()))
        print 'feature codebook size: %s' % (','.join(('%s:%s')%(i,f.size()) for i,f in self.feature_codebook.items()))
        print 'weight shape: %s' % (','.join(('%s:%s')%(i,w.shape) for i,w in enumerate(self.avg_weight)))
        print 'token to concept table: %s' % (len(self.token_to_concept_table))
        model_dict = {
            '_feature_templates_list': self._feature_templates_list,
            '_feats_gen_filename':self._feats_gen_filename,
            #'weight':[w.tolist() for w in self.weight],
            #'aux_weight':[axw.tolist() for axw in self.aux_weight],
            'avg_weight':[agw.tolist() for agw in self.avg_weight],
            'token_to_concept_table': dict([(k,list(v)) for k,v in self.token_to_concept_table.items()]),
            'class_codebook':self.class_codebook.to_dict(),
            'feature_codebook':self.feature_codebook.to_dict(),
            'rel_codebook':self.rel_codebook.to_dict(),
            'tag_codebook':dict([(k,self.tag_codebook[k].to_dict()) for k in self.tag_codebook])
        }
        return model_dict
        
    def save_model(self,model_filename):
        #pickle.dump(self,open(model_filename,'wb'),pickle.HIGHEST_PROTOCOL)
        print >> self.elog, 'Model info:'
        print >> self.elog,'class size: %s \nrelation size: %s \ntag size: %s'%(self.class_codebook.size(),self.rel_codebook.size(),map(lambda x:'%s->%s '%(x,self.tag_codebook[x].size()),self.tag_codebook.keys()))
        print >> self.elog,'feature codebook size: %s' % (','.join(('%s:%s')%(i,f.size()) for i,f in self.feature_codebook.items()))
        #print 'weight shape: %s' % (self.avg_weight.shape)
        print >> self.elog,'weight shape: %s' % (','.join(('%s:%s')%(i,w.shape) for i,w in enumerate(self.avg_weight)))
        print >> self.elog,'token to concept table: %s' % (len(self.token_to_concept_table))

        with contextlib.closing(bz2.BZ2File(model_filename, 'wb')) as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        #json.dump(self.toJSON(),open(model_filename,'wb'))
        
    @staticmethod
    def load_model(model_filename):
        with contextlib.closing(bz2.BZ2File(model_filename, 'rb')) as f:
            model = pickle.load(f)
        return model
        #return pickle.load(open(model_filename,'rb'))
        '''
        model_dict = json.load(open(model_filename,'rb'))
        model_instance = Model()
        model_instance._feature_templates_list = model_dict['_feature_templates_list']
        model_instance._feats_gen_filename = model_dict['_feats_gen_filename']
        model_instance.feats_generator = importlib.import_module('temp.'+model_instance._feats_gen_filename).generate_features
        #model_instance.weight = [np.array(w) for w in model_dict['weight']]
        #model_instance.aux_weight = [np.array(axw) for axw in model_dict['aux_weight']]
        model_instance.avg_weight = [np.array(agw) for agw in model_dict['avg_weight']]
        model_instance.token_to_concept_table = defaultdict(set,[(k,set(v)) for k,v in model_dict['token_to_concept_table'].items()])
        model_instance.class_codebook = Alphabet.from_dict(model_dict['class_codebook'])
        model_instance.feature_codebook = Alphabet.from_dict(model_dict['feature_codebook'])
        model_instance.rel_codebook = Alphabet.from_dict(model_dict['rel_codebook'])
        model_instance.tag_codebook = Alphabet.from_dict(model_dict['tag_codebook'])
        return model_instance
        '''
