# all the constants 
import numpy as np
import re
from os import listdir
from collections import defaultdict


# flags
# TODO using gflag
FLAG_COREF=False
FLAG_PROP=False
FLAG_RNE=False
FLAG_VERB=False
FLAG_DEPPARSER='stanford'
FLAG_ONTO='wsj'

# constants
NOT_APPLY='_NOT_APPLY_'
EMPTY = None
NOT_ASSIGNED = None
NULL_EDGE = 'null_edge'
NULL_TAG = 'null_tag'
#MERGE_TAG = '_MERGE_TAG_'

ROOT_FORM = '_ROOT_FORM_'
ROOT_LEMMA = '_ROOT_LEMMA_'
ROOT_POS = '_ROOT_POS_'
ROOT_CPOS = '_ROOT_CPOS_'

FAKE_ROOT_VAR = 'x'
FAKE_ROOT_CONCEPT = 'xconcept'
FAKE_ROOT_EDGE = 'x'

ABT_PREFIX = 'ap'
ABT_FORM = '_ABT_FORM_'
ABT_LEMMA = '_ABT_LEMMA_'
ABT_POS = '_ABT_POS_'
ABT_NE = '_ABT_NE_'
ABT_REL = '_ABT_REL_'
ABT_TOKEN={'form':ABT_FORM,'lemma':ABT_LEMMA,'pos':ABT_POS,'ne':ABT_NE,'rel':ABT_REL}

START_ID = 'st'
START_FORM = '_START_FORM_'
START_LEMMA = '_START_LEMMA_'
START_POS = '_START_POS_'
START_NE = '_START_NE_'
START_REL = '_START_REL_'
START_TOKEN={'form':START_FORM,'lemma':START_LEMMA,'pos':START_POS,'ne':START_NE,'rel':START_REL}
START_EDGE='_START_EDGE_'
#START_TAG='_START_TAG_'

# error types
NODE_MATCH_ERROR = '#1'
NODE_TYPE_ERROR = '#2'

EDGE_MATCH_ERROR = '#1'
EDGE_TYPE_ERROR = '#2'

NEXT1 = 0
NEXT2 = 1
REATTACH = 2
DELETENODE = 3
SWAP = 4
REENTRANCE = 5
REPLACEHEAD = 6
MERGE = 7
INFER = 8
ADDCHILD = 9


PRE_MERGE_NETAG = ['PERSON','LOCATION','ORGANIZATION','MISC','DATE']
INFER_NETAG = set(['PERSON','LOCATION','ORGANIZATION','MISC'])
FUNCTION_TAG = ['IN','DT','TO','RP']

WEIGHT_DTYPE=np.float32

DETERMINE_TREE_TO_GRAPH_ORACLE = 1
DETERMINE_TREE_TO_GRAPH_ORACLE_SC = 2
DET_T2G_ORACLE_ABT = 3
DETERMINE_STRING_TO_GRAPH_ORACLE = None

ACTION_WITH_TAG = set([INFER])
ACTION_WITH_EDGE = set([NEXT1,REATTACH,SWAP,REENTRANCE])

ACTION_TYPE_TABLE = {}
ACTION_TYPE_TABLE['basic'] = [
    (NEXT1,'next1'),(NEXT2,'next2'),
    (REATTACH,'reattach'),
    (DELETENODE,'delete_node'),
    (SWAP,'swap'),
    (REENTRANCE,'reentrance'),
    (REPLACEHEAD,'replace_head'),
    (MERGE,'merge'),
    (INFER,'infer')
    #(ADDCHILD,'add_child')
]

FEATS_ABBR = {}
FEATS_ABBR['w'] = 'form'
FEATS_ABBR['lemma'] = 'lemma'
FEATS_ABBR['t'] = 'pos'
FEATS_ABBR['tag'] = 'tag'
FEATS_ABBR['ne'] = 'ne'
FEATS_ABBR['dl'] = 'rel'
FEATS_ABBR['p1'] = 'p1'
FEATS_ABBR['lsb'] = 'lsb'
FEATS_ABBR['rsb'] = 'rsb'
FEATS_ABBR['r2sb'] = 'r2sb'
FEATS_ABBR['prs1'] = 'prs1'
FEATS_ABBR['prs2'] = 'prs2'

FEATS_ABBR['pathx'] = 'pathx'
FEATS_ABBR['pathp'] = 'pathp'
FEATS_ABBR['pathprep'] = 'pathprep'
FEATS_ABBR['apathx'] = 'apathx'
FEATS_ABBR['apathp'] = 'apathp'
FEATS_ABBR['apathprep'] = 'apathprep'
FEATS_ABBR['pathpwd'] = 'pathpwd'
FEATS_ABBR['apathpwd'] = 'apathpwd'

FEATS_ABBR['nswp'] = 'nswp'
FEATS_ABBR['reph'] = 'reph'
FEATS_ABBR['isand'] = 'isand'
FEATS_ABBR['len'] = 'len'
FEATS_ABBR['ppcat'] = 'ppcat'

FEATS_ABBR['pfx'] = 'pfx' # preffix
FEATS_ABBR['cpt'] = 'concept' # concept
FEATS_ABBR['dch'] = 'dch'  # deleted children
FEATS_ABBR['cap'] = 'cap'
FEATS_ABBR['isne'] = 'isne'
FEATS_ABBR['eqne'] = 'eqne'
FEATS_ABBR['isnom'] = 'isnom'
FEATS_ABBR['nech'] = 'nech'
FEATS_ABBR['c1lemma'] = 'c1lemma'
FEATS_ABBR['c1dl'] = 'c1dl'
FEATS_ABBR['istrace'] = 'istrace'
FEATS_ABBR['rtr'] = 'rtr'
FEATS_ABBR['iscycle'] = 'iscycle'
FEATS_ABBR['lsl'] = 'lsl'
FEATS_ABBR['arg0'] = 'arg0'
FEATS_ABBR['arg1'] = 'arg1'
FEATS_ABBR['arg2'] = 'arg2'
FEATS_ABBR['hastrace'] = 'hastrace'
FEATS_ABBR['hasnsubj'] = 'hasnsubj'

FEATS_ABBR['brown4'] = 'brown4'
FEATS_ABBR['brown6'] = 'brown6'
FEATS_ABBR['brown8'] = 'brown8'
FEATS_ABBR['brown10'] = 'brown10'
FEATS_ABBR['brown20'] = 'brown20'

FEATS_ABBR['isarg'] = 'isarg'
FEATS_ABBR['arglabel'] = 'arglabel'
FEATS_ABBR['isprd'] = 'isprd'
FEATS_ABBR['prdlabel'] = 'prdlabel'
FEATS_ABBR['frmset'] = 'frmset'
FEATS_ABBR['eqfrmset'] = 'eqfrmset'
FEATS_ABBR['2merge'] = '2merge'
FEATS_ABBR['isleaf'] = 'isleaf'
FEATS_ABBR['txv'] = 'txv'
FEATS_ABBR['txn'] = 'txn'
FEATS_ABBR['txdelta'] = 'txdelta'

DEFAULT_RULE_FILE = './rules/dep2amrLabelRules'

def _load_rules(rule_file):
    rf = open(rule_file,'r')
    d = {}
    for line in rf.readlines():
        if line.strip():
            dep_rel,amr_rel,_ = line.split()
            if dep_rel not in d: d[dep_rel] = amr_rel[1:]
        else:
            pass
    return d

__DEP_AMR_REL_TABLE = _load_rules(DEFAULT_RULE_FILE)
def get_fake_amr_relation_mapping(dep_rel):
    return __DEP_AMR_REL_TABLE[dep_rel]

DEFAULT_NOM_FILE = './resources/nombank-dict.1.0'

def _read_nom_list(nombank_dict_file):
    nomdict = open(nombank_dict_file,'r')
    nomlist = []
    token_re = re.compile('^\\(PBNOUN :ORTH \\"([^\s]+)\\" :ROLE-SETS')
    for line in nomdict.readlines():
        m = token_re.match(line.rstrip())
        if m:
            nomlist.append(m.group(1))
    return nomlist

NOMLIST = _read_nom_list(DEFAULT_NOM_FILE)

DEFAULT_BROWN_CLUSTER = './resources/wclusters-engiga'
    
def _load_brown_cluster(dir_path,cluster_num=1000):
    cluster_dict = defaultdict(str)
    for fn in listdir(dir_path):
        if re.match('^.*c(\d+).*$',fn).group(1) == str(cluster_num) and fn.endswith('.txt'):
            with open(dir_path+'/'+fn,'r') as f:
                for line in f:
                    bitstring, tok, freq = line.split()
                    cluster_dict[tok]=bitstring

    return cluster_dict

BROWN_CLUSTER=_load_brown_cluster(DEFAULT_BROWN_CLUSTER)

PATH_TO_VERB_LIST = './resources/verbalization-list-v1.01.txt'

def _load_verb_list(path_to_file):
    verbdict = {}
    with open(path_to_file,'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                if not line.startswith('DO-NOT-VERBALIZE'):
                    verb_type, lemma, _, subgraph_str = re.split('\s+',line,3)
                    subgraph = {}
                
                    #if len(l) == 1: 
                    #else: # have sub-structure
                    root = re.split('\s+', subgraph_str, 1)[0]
                    subgraph[root] = {}
                    for match in re.finditer(':([^\s]+)\s*([^\s:]+)',subgraph_str):
                        relation = match.group(1)
                        concept = match.group(2)
                        subgraph[root][relation] = concept
                        
                    verbdict[lemma] = verbdict.get(lemma,[])
                    verbdict[lemma].append(subgraph)

    return verbdict

VERB_LIST = _load_verb_list(PATH_TO_VERB_LIST)

PATH_TO_COUNTRY_LIST='./resources/country-list.csv'

def _load_country_list(path_to_file):
    countrydict = {}
    with open(path_to_file,'r') as f:
        for line in f:
            line = line.strip()
            country_name, country_adj, _ = line.split(',', 2)
            countrydict[country_adj] = country_name

    return countrydict
    
COUNTRY_LIST=_load_country_list(PATH_TO_COUNTRY_LIST)
                

# given different domain, return range of split corpus #TODO: move this part to config file
def get_corpus_range(corpus_section,corpus_type):
    DOMAIN_RANGE_TABLE={ \
        'train':{
            'proxy':(0,6603),
            'bolt':(6603,7664),
            'dfa':(7664,9367),
            'mt09sdf':(9367,9571),
            'xinhua':(9571,10312)
        },
        'dev':{
            'proxy':(0,826),
            'bolt':(826,959),
            'consensus':(959,1059),
            'dfa':(1059,1269),
            'xinhua':(1269,1368)
        },
        'test':{
            'proxy':(0,823),
            'bolt':(823,956),
            'consensus':(956,1056),
            'dfa':(1056,1285),
            'xinhua':(1285,1371)
        }
    }

    return DOMAIN_RANGE_TABLE[corpus_type][corpus_section]