# all the constants 
import numpy as np

# flags
FLAG_DEPPARSER='stanford'

EMPTY = None
NOT_ASSIGNED = None
NULL_EDGE = 'null_edge'
NULL_TAG = 'null_tag'

ROOT_FORM = '_ROOT_FORM_'
ROOT_LEMMA = '_ROOT_LEMMA_'
ROOT_POS = '_ROOT_POS_'
ROOT_CPOS = '_ROOT_CPOS_'
FAKE_ROOT_VAR = 'x'
FAKE_ROOT_CONCEPT = 'xconcept'
FAKE_ROOT_EDGE = 'x'

NEXT1 = 0
NEXT2 = 1
REATTACH = 2
DELETENODE = 3
SWAP = 4
REENTRANCE = 5
REPLACEHEAD = 6
MERGE = 7
#DELETEEDGE = 8
ADDCHILD = 9


PRE_MERGE_NETAG = ['PERSON','LOCATION','ORGANIZATION','MISC','DATE']


WEIGHT_DTYPE=np.float32

DETERMINE_TREE_TO_GRAPH_ORACLE = 1
DETERMINE_TREE_TO_GRAPH_ORACLE_SC = 2
DETERMINE_STRING_TO_GRAPH_ORACLE = None

ACTION_WITH_TAG = set([NEXT2])
ACTION_WITH_EDGE = set([NEXT1,ADDCHILD,REATTACH,SWAP,REENTRANCE])

ACTION_TYPE_TABLE = {}
ACTION_TYPE_TABLE['basic'] = [
    (NEXT1,'next1'),(NEXT2,'next2'),
    (REATTACH,'reattach'),
    (DELETENODE,'delete_node'),
    (SWAP,'swap'),
    (REENTRANCE,'reentrance'),
    #(ADDCHILD,'add_child'),
    (REPLACEHEAD,'replace_head'),
    (MERGE,'merge')
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
    


