#!/usr/bin/python

"""
aligner for AMR and its corresponding sentence
"""
import re,sys
from common.util import *
from common.util import english_number,to_order
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from span import Span
#from nltk.corpus import wordnet as wn
#class regex_pattern:
#    TempQuantity = '(?P<quant>[0-9]+)(?P<unit>year|month?)s?'




class SearchException(Exception):
    pass

class Atype:
    WORD_ALIGN = 1
    SPAN_ALIGN = 2 
    SPAN_ALIGN2 = 3

class Aligner():
    ENTITY_TAG_TABLE = {'NameEntity':'NE',
                        'QuantityEntity':'QTY',
                        'organization':'ORG',
                        'haveOrgRole91':'HOR91',
                        'RateEntity':'RE',
                        'Number':'O',
                        'NegPolarity':'NEG',
                        'NegForm':'NEGFORM',
                        'thing':'TH',
                        'cause':'CAUSE',
                        'organization-name':'ORGNE',
                        'person':'PER',
                        'person-name':'PERNE',
                        'country-name':'COUNTRYNE',
                        'city-name':'CITYNE',
                        'state-name':'STATENE',
                        'desert-name':'DESERTNE',
                        'picture':'PIC',
                        'picture-name':'PICNE',
                        #'planet-name':'PLANE',
                        'ago':'AGO',
                        'RelativePosition':'RELATIVEPOS',
                        'SingleConcept':'O',
                        'OrdinalEntity':'ORDNE',
                        'DateEntity':'DATE'
                    }

    align_table = {Atype.WORD_ALIGN:'word_align',
                   Atype.SPAN_ALIGN:'span_align',
                   Atype.SPAN_ALIGN2:'span_align2'
                   }

    neg_polarity = ['no','not','non','without','never',"n't",'neither','nor']
    neg_prefix = ['dis','un','im','in']
    plural_suffix = ['s']
    fuzzy_max_len = 4 # maximum prefix match between string
    special_morph_table = {'see':['saw','sighted'],
                           'be-located-at':['is','was','were'],
                           'possible':['able','could','can','may','might','perhaps'],
                           'i':['my','me'],
                           'contrast':['but','however','yet'],
                           'resemble':['like'],
                           'recommend':['should'],
                           'cause':['since','why','so','as','thus','from','because','for'],
                           'so':['such'],
                           'person':['people'],
                           'you':['your'],
                           'they':['them','their'],
                           'he':['him','his'],
                           'she':['her'],
                           'we':['us'],
                           'or':['nor'],
                           'live':['life'],
                           'die':['death'],
                           'appear':['apparition'],
                           'inhabit':['habitation'],
                           'age':['old'],
                           'place':['where'],
                           'amr-unknown':['where','which','what','how'],
                           'this':['these'],
                           'dollar':['$'],
                           'prove':['proof'],
                           'temporal-quantity':['when'],
                           'relative-position':['in','from'],
                           }
    lmtzr = WordNetLemmatizer()
    concept_lex_rule = [('NameEntity','name$'),
                        ('QuantityEntity','.*-quantity'),
                        ('DateEntity','date-entity'),
                        ('haveOrgRole91','have-org-role-91'),
                        ('RateEntity','rate-entity-91'),
                        ('organization','organization'),
                        ('Number','[0-9]+'),
                        ('NegPolarity','-'),
                        ('thing','thing'),
                        ('person','person'),
                        ('picture','picture'),
                        #('planet','planet'),
                        ('country','country'),
                        ('state','state$'),
                        ('city','city'),
                        ('desert','desert'),
                        ('OrdinalEntity','ordinal-entity'),
                        ('multiple','multiple'),
                        ('RelativePosition','relative-position'),
                        ('SingleConcept',r'[^:\s,]+')
                        ]
    def __init__(self,align_type=3,verbose=0):
        self.align_type = align_type
        self.verbose = verbose
        self.concept_patterns = self._compile_regex_rule(Aligner.concept_lex_rule)
        
    @staticmethod
    def readJAMRAlignment(amr,JAMR_alignment):
        alignment = defaultdict(list)

        for one_alignment in JAMR_alignment.split():
            offset, fragment = one_alignment.split('|')
            start = int(offset.split('-')[0])+1
            end = int(offset.split('-')[1])+1
            posIDs = fragment.split('+')
            if len(posIDs) == 1:
                variable = amr.get_variable(posIDs[0])
                if variable in amr.node_to_concepts:
                    concept = amr.node_to_concepts[variable]
                    span = Span(start,end,[concept],concept)
                else: # constant variable
                    concept = variable
                    span = Span(start,end,[concept],ConstTag(concept))
                alignment[variable].append(span)
            else:
                tokens = []
                tags = []
                level = 0
                all_variables = []
                variable = None
                while posIDs:
                    pid = posIDs.pop()
                    #if pid == '0.2.1.0.0.0.0.0.0':
                    #    import pdb
                    #    pdb.set_trace()
                    pre_level = level
                    level = len(pid.split('.'))
                    variable = amr.get_variable(pid)
                    if variable == None:
                        import pdb
                        pdb.set_trace()
                    
                    if pre_level > level:
                        concept = amr.node_to_concepts[variable]
                        tags.insert(0,concept)
                        all_variables.append(variable)
                    else:
                        if variable in amr.node_to_concepts:
                            concept = amr.node_to_concepts[variable]
                            tokens.insert(0,concept)
                            all_variables.append(variable)
                        else:
                            if variable == '-': # negation
                                tags.insert(0,variable)
                            tokens.insert(0,variable)
                span = Span(start,end,tokens,ETag('+'.join(tags)))
                for v in all_variables:alignment[v].append(span)

        return alignment
        
    def apply_align(self,sent,amr):
        """apply the alignment for sentence and its amr"""
        return getattr(self,Aligner.align_table[self.align_type])(sent,amr)
    
    def _compile_regex_rule(self,rules):
        regexstr = '|'.join('(?P<%s>%s)' % (name,rule) for name,rule in rules)
        return re.compile(regexstr,re.IGNORECASE)

    def span_align2(self,sentence,amr):
        sent = sentence[:]
        alignment = defaultdict(list)
        alignment['root'] = 0
        tokens = [(i+1,x) for i,x in enumerate(sent.split())]
        
        node_seq,amr_triples = amr.bfs()
        unmatched_vars = node_seq
        triples = amr_triples
        #print unmatched_vars
        while unmatched_vars:
            cur_node = unmatched_vars.pop(0)
            cur_triple = triples.pop(0)
            cur_var = cur_node.node_label
            success , sent, tokens = self.align_single_concept(sent,tokens,cur_var,amr,alignment,unmatched_vars,triples)

        return alignment 
    
    def word_align(self,sentence,amr):
        """
           use set of rules greedily align concepts to words, for special concepts like name,date-entity,etc., they 
           stay unaligned
           details: Flanigan,2014 ACL
        """
        
        sent = sentence[:] # copy the sentence 
        alignment = defaultdict(list)
        alignment['root'] = 0
        
        tokens = [(i+1,x) for i,x in enumerate(sent.split())]
        #tagged_tokens = nltk.pos_tag(tokens)

        # single root graph
        unmatched_variables = list(set([var for var in amr.bfs()[0] if not isinstance(var,StrLiteral)]))

        while unmatched_variables:
            cur = unmatched_variables.pop(0)
            if cur in amr.node_to_concepts:                 
                cur_concept = amr.node_to_concepts[cur]
            else: #not have concepts
                cur_concept = cur
            match = self.concept_patterns.match(cur_concept)
            #import pdb
            #pdb.set_trace()
            if match:
                rule_type = match.lastgroup
                matched_variable = cur
                span = None
                update = True
                #matched_variable_pos = int(match.group(0).split(':')[0])
                #matched_variable = unmatched_variables[matched_variable_pos]

                if rule_type == 'NameEntity':
                    NE_items = [v[0] for k,v in amr[matched_variable].items()]
                    #spans = [(j,len(NEStr)) for j in range(len(sent_list)) if sent_list[j:j+len(NEStr)] == NEStr]
                    NE_pattern = re.compile("\s".join(NE_items),re.IGNORECASE)

                    '''
                    m = name_re.match(sent)
                    span = [(i,len(NEStr)) for i in range(len(sent_list)) if (sum(map(lambda x:len(x)+1,sent_list[:i])),sum(map(lambda x:len(x)+1,sent_list[:i]))+len(m.group())) == m.span()]
                    alignment[matched_variable].append(span)

                    span = [range(i,i+len(NEStr)) for i in range(len(sent_list)) if sent_list[i:i+len(NEStr)] == NEStr]
                    '''
                    span = self._search_sent(NE_pattern,sent,tokens)
                    for sid, n in zip(range(span[0],span[1]), NE_items):
                        alignment[n].append(sid)
                    alignment[matched_variable].append(matched_variable)

                elif rule_type == 'QuantityEntity':
                    quantity = ''
                    unit = ''
                    unit_node = None
                    for k,v in amr[matched_variable].items():
                        if k == 'quant':
                            quantity = v[0]
                        elif k == 'unit':
                            unit = amr.node_to_concepts[v[0]]
                            unit_node = v[0]
                        else:
                            # other modifier
                            pass
                    if quantity and unit:
                        TEMP_pattern = re.compile('(%s|%s)\s+(%s)s?' % (quantity,english_number(int(quantity)),unit),re.IGNORECASE)
                        TEMP_items = [quantity,unit_node]
                    else:
                        missing = ''
                        if quantity == '':
                            missing += ' quantity'
                        if unit == '':
                            missing += ' unit'
                        
                        raise Exception('Quantity Entity %s does not contain %s'%(cur_concept,missing))
                    '''
                    temp_re = re.compile(r'(%s|%s)\s+(%s)s?' % (quantity,english_number(int(quantity)),unit))
                    #temp_re = re.compile(regex_pattern.TempQuantity,re.IGNORECASE)
                    m = temp_re.match(sent)
                    span = [range(i,i+2) for i in range(len(sent_list)) if (sum(map(lambda x:len(x)+1,sent_list[:i])),sum(map(lambda x:len(x)+1,sent_list[:i]))+len(m.group())) == m.span()]
                    #alignment[matched_variable].append(span)
                    '''
                    span = self._search_sent(TEMP_pattern,sent,tokens)
                    for sid, n in zip(range(span[0],span[1]), TEMP_items):
                        alignment[n].append(sid)
                    alignment[matched_variable].append(matched_variable)
    
                    self.remove_aligned_concepts(unmatched_variables,amr[matched_variable].items())

                elif rule_type == 'NegPolarity':
                    aligned = False
                    for i,token in tokens:
                        if token.lower() in Aligner.neg_polarity:
                            aligned = True
                            break
                    if aligned:
                        span = (i,i+1)
                        alignment[matched_variable].append(i)
                    else:
                        update = False
                    
                elif rule_type == 'SingleConcept':
                    tmp = cur_concept.rsplit('-',1)
                    sense = None 
                    if len(tmp) == 2:
                        sense = tmp[1]
                    cur_concept = tmp[0].lower()
                    for idx,token in tokens:
                        t = token.lower()
                        if t == cur_concept:  # exact match
                            span = (idx,idx+1)
                            break
                        elif self.fuzzy_match(t,cur_concept,Aligner.fuzzy_max_len):
                            span = (idx,idx+1)
                            break
                        elif self.WN_lemma_match(t,cur_concept,sense):
                            span = (idx,idx+1)
                            break
                        elif self.is_spec_form(t,cur_concept):
                            span = (idx,idx+1)
                            break
                        else:
                            pass

                    if span:
                        alignment[matched_variable].append(idx)
                    else:
                        print >> sys.stderr, 'WARNING: Variable %s/%s cannot be aligned'%(matched_variable,cur_concept)
                        alignment[matched_variable].append(matched_variable)
                        update = False
                else:
                    pass
                
                # update
                if update:
                    tokens = [(i,tk) for i,tk in tokens if i not in range(span[0],span[1])]
                    sent = ' '.join(x for i,x in tokens)
                    if self.verbose > 2:
                        print >> sys.stderr, "Concept '%s' Matched to span '%s' "%(cur_concept,' '.join(w for i,w in enumerate(sentence.split()) if i+1 in range(span[0],span[1])))
                        print sent
                        print alignment
                
                        #raw_input('ENTER to continue')

        return alignment
    
    def span_align(self,sentence,amr):
        '''
        use rules to align amr concepts to sentence spans 
        '''
        sent = sentence[:]
        alignment = defaultdict(list)
        alignment['root'] = 0
        tokens = [(i+1,x) for i,x in enumerate(sent.split())]
        
        unmatched_vars = list(set([var for var in amr.bfs()[0] if not isinstance(var,StrLiteral)]))
        
        while unmatched_vars:
            cur = unmatched_vars.pop(0)
            if cur in amr.node_to_concepts:
                cur_concept = amr.node_to_concepts[cur]
            else:
                cur_concept = cur
            match = self.concept_patterns.match(cur_concept)
            if match:
                rule_type = match.lastgroup
                span = None
                update = True
                if rule_type == "NameEntity":
                    NE_items = [v[0] for k,v in amr[cur].items()]
                    NE_pattern = re.compile(r"\s".join(NE_items),re.IGNORECASE)
                    
                    start,end = self._search_sent(NE_pattern,sent,tokens)
                    assert end-start == len(NE_items)
                    span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],NE_items)
                    alignment[cur].append(span)
                
                elif rule_type == "QuantityEntity":
                    quantity = ''
                    unit = ''
                    unit_var = None
                    for k,v in amr[cur].items():
                        if k == 'quant':
                            quantity = v[0]
                        elif k == 'unit':
                            unit_var = v[0]
                            unit = amr.node_to_concepts[v[0]]
                        else:
                            pass
                    if quantity and unit:
                        QTY_pattern = re.compile('(%s|%s)\s+(%s)s?' % (quantity,english_number(int(quantity)),unit),re.IGNORECASE)
                        QTY_items = [quantity,unit]
                        start,end = self._search_sent(QTY_pattern,sent,tokens)
                        assert end - start == len(QTY_items)
                        span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],QTY_items)
                        alignment[cur].append(span)
                        
                        self.remove_aligned_concepts(unmatched_vars,amr[cur].items())
                elif rule_type == "NegPolarity":
                    aligned = False
                    for i,token in tokens:
                        if token.lower() in Aligner.neg_polarity:
                            aligned = True
                            break
                    if aligned:
                        span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE[rule_type],[token])
                        alignment[cur].append(span)
                    else:
                        update = False
                        
                elif rule_type == "SingleConcept":
                    tmp = cur_concept.rsplit('-',1)
                    sense = None 
                    if len(tmp) == 2:
                        sense = tmp[1]
                    cur_concept = tmp[0].lower()
                    for idx,token in tokens:
                        t = token.lower()
                        if t == cur_concept:  # exact match
                            span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                            break
                        elif self.fuzzy_match(t,cur_concept,Aligner.fuzzy_max_len):
                            span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                            break
                        elif self.is_neg_form(t,cur_concept):
                            span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                            break
                        elif self.WN_lemma_match(t,cur_concept,sense):
                            span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                            break
                        elif self.is_spec_form(t,cur_concept):
                            span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                            break
                        else:
                            pass

                    if span:
                        alignment[cur].append(span)
                    else:
                        print >> sys.stderr, 'Variable/Concept %s/%s cannot be aligned'%(cur,cur_concept)
                        #alignment[matched_variable].append(matched_variable)
                        update = False                    
            else:
                raise Exception('Can not find type of concept %s / %s'%(cur,cur_concept))

            # update
            if update:
                tokens = [(i,tk) for i,tk in tokens if i not in range(span.start,span.end)]
                sent = ' '.join(x for i,x in tokens)
                if self.verbose > 2:
                    print >> sys.stderr, "Concept '%s' Matched to span '%s' "%(cur_concept,' '.join(w for i,w in enumerate(sentence.split()) if i+1 in range(span[0],span[1])))
                    print sent
                    print alignment
                    
                    #raw_input('ENTER to continue')

        return alignment
    


    def align_single_concept(self,sent,tokens,cur_var,amr,alignment,unmatched_vars,triples,NEXT=False):
        '''align single concept'''
        
        if cur_var in amr.node_to_concepts:
            cur_concept = amr.node_to_concepts[cur_var]
        else:
            cur_concept = cur_var

        if cur_var in alignment and not NEXT and not isinstance(cur_var,(StrLiteral,Quantity,Polarity)) : # already aligned
            return True, sent,tokens

        match = self.concept_patterns.match(cur_concept)
        if match:
            rule_type = match.lastgroup
            span = None
            update = True
            if rule_type == "NameEntity":
                NE_items = [v[0] for k,v in amr[cur_var].items() if isinstance(v[0],StrLiteral)]
                nep = r'%s|%s'%(r'\s'.join(NE_items),r'\s'.join(n[:4] if len(n) > 3 else n for n in NE_items))
                NE_pattern = re.compile(nep,re.IGNORECASE)
                
                start,end = self._search_sent(NE_pattern,sent,tokens)
                assert end-start == len(NE_items)
                span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],NE_items)
                alignment[cur_var].append(span)
                for k,v in amr[cur_var].items():
                    if isinstance(v[0],StrLiteral):
                        self.remove_aligned_concepts(cur_var,k,v[0],unmatched_vars,triples)

            elif rule_type in ["DateEntity", "haveOrgRole91","RateEntity"]:
                EN_items = []
                EN_spans = []
                for k,v in amr[cur_var].items():                    
                    vconcept = amr.node_to_concepts[v[0]] if v[0] in amr.node_to_concepts else v[0]
                    EN_items.append(vconcept)
                    success, sent, tokens = self.align_single_concept(sent,tokens,v[0],amr,alignment,unmatched_vars,triples)

                    sp = alignment[v[0]][-1]
                    sp.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type])
                    EN_spans.append(sp)
                    self.remove_aligned_concepts(cur_var,k,v[0],unmatched_vars,triples)
                #print NE_spans,alignment
                start = EN_spans[0].start
                end = EN_spans[-1].end
                span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],EN_items)
                span.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type])
                alignment[cur_var].append(span)

            elif rule_type == "QuantityEntity":
                quantity = ''
                unit = ''
                unit_var = None
                q_success = False
                u_success = False
                                
                for k,v in amr[cur_var].items():
                    if k == 'quant':
                        quantity = v[0]
                        q_success, sent, tokens = self.align_single_concept(sent,tokens,quantity,amr,alignment,unmatched_vars,triples)
                    elif k == 'unit':
                        unit_var = v[0]
                        unit = amr.node_to_concepts[v[0]]
                        u_success, sent, tokens = self.align_single_concept(sent,tokens,unit_var,amr,alignment,unmatched_vars,triples)
                    else:
                        pass
                        
                if q_success and u_success:
                    #QTY_pattern = r'(%s|%s)\s+(%s)s?' % (quantity,english_number(int(quantity)),unit)
                    #QTY_items = [quantity,unit]
                    #start,end = self._search_sent(QTY_pattern,QTY_items,sent,tokens)
                    #assert end - start == len(QTY_items)
                    quantity_span = alignment[quantity][-1]
                    unit_span = alignment[unit_var][0]
                    start = quantity_span.start if quantity_span.start < unit_span.end else unit_span.start
                    end = unit_span.end if quantity_span.start < unit_span.end else quantity_span.end
                    while not (end - len(quantity_span.words)-len(unit_span.words) - start < 2): # wrong match more than one quantity to map in sentence
                        alignment[quantity].pop()
                        q_success, sent, tokens = self.align_single_concept(sent,tokens,quantity,amr,alignment,unmatched_vars,triples,NEXT=True) # redo it on updated sentence
                        quantity_span = alignment[quantity][-1]
                        start = quantity_span.start
                    #assert start == end - 2
                    span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],[quantity,unit])
                    self.remove_aligned_concepts(cur_var,'quant',quantity,unmatched_vars,triples)
                    alignment[cur_var].append(span)
                elif q_success and not u_success: # does not have unit or unit cannot be aligned
                    quantity_span =  alignment[quantity][0]
                    start = quantity_span.start
                    end = quantity_span.end
                    span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],[quantity])
                    self.remove_aligned_concepts(cur_var,'quant',quantity,unmatched_vars,triples)
                    alignment[cur_var].append(span)
                    #self.remove_aligned_concepts(unmatched_vars,amr[cur_var].items())
                elif not q_success and u_success:
                    unit_span = alignment[unit_var][0]
                    span = Span(unit_span.start,unit_span.end,Aligner.ENTITY_TAG_TABLE[rule_type],[unit])
                    self.remove_aligned_concepts(cur_var,'unit',unit_var,unmatched_vars,triples)
                    alignment[cur_var].append(span)
                else:
                    rule_type = 'SingleConcept'
            elif rule_type == "Number":
                '''
                aligned = False
                num = [cur_var]
                num.extend(english_number(int(cur_var)).split('|'))
                for i,token in tokens:
                    if token.lower() in num:
                        aligned = True
                        break
                if aligned:
                    span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE[rule_type],[token])
                    alignment[cur_var].append(span)
                else:
                    print >> sys.stderr, 'Variable/Concept %s/%s cannot be aligned'%(cur_var,cur_concept)
                    update = False
                '''
                if re.match('[0-9]+:[0-9]+',cur_concept):
                    num = [('time','(\\s|^)('+cur_concept+')(\\s|&)'),
                           ('english','(\\s|^)('+to_time(cur_concept)+')(\\s|&)')]
                else:
                    num = [('digit','(\\s|^)('+cur_concept+'|'+format_num(cur_concept)+')(\\s|&)'),
                           ('string','(\\s|^)('+english_number(int(cur_concept))+')(\\s|&)'),
                           ('order','(\\s|^)('+to_order(cur_concept)+')(\\s|&)'),
                           ('round','(\\s|^)('+to_round(int(cur_concept))+')(\\s|&)') 
                       ]
                NUM_pattern = self._compile_regex_rule(num)
                #print NUM_pattern.pattern
                try:
                    start,end = self._search_sent(NUM_pattern,sent,tokens)
                    span = Span(start,end,Aligner.ENTITY_TAG_TABLE[rule_type],[w for i,w in tokens if i in range(start,end)])
                    alignment[cur_var].append(span)                
                except Exception as e:
                    update = False
                    print >> sys.stderr,e
                    #raw_input('CONTINUE')
            
            elif rule_type == 'multiple':
                op1 = amr[cur_var]['op1'][0]

                success, sent, tokens = self.align_single_concept(sent,tokens,op1,amr,alignment,unmatched_vars,triples)
                if success:
                    span = alignment[op1][0]
                    alignment[cur_var].append(span)                                    
                    self.remove_aligned_concepts(cur_var,'op1',op1,unmatched_vars,triples)  
                else:
                    update = False
                
            elif rule_type in ["person","picture","country","state","city","desert","organization"]:
                if 'name' in amr[cur_var]:
                    k_var = amr[cur_var]['name'][0]
                    success, sent, tokens = self.align_single_concept(sent,tokens,k_var,amr,alignment,unmatched_vars,triples)
                    span = alignment[k_var][0]
                    span.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type+'-name'])
                    alignment[cur_var].append(span)
                else:
                    ind,span = self.try_align_as_single_concept(cur_var,cur_concept,amr,alignment,tokens,unmatched_vars,triples)
                    if ind:
                        pass
                    elif 'ARG0-of' in amr[cur_var]:
                        k_var = amr[cur_var]['ARG0-of'][0]
                        success, sent, tokens = self.align_single_concept(sent,tokens,k_var,amr,alignment,unmatched_vars,triples)
                        if success:
                            span = alignment[k_var][0]
                            span.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type])
                            alignment[cur_var].append(span)
                        else:
                            update = False

                    else:
                        update = False
               


            elif rule_type == "NegPolarity":
                aligned = False
                for i,token in tokens:
                    if token.lower() in Aligner.neg_polarity:
                        aligned = True
                        break
                if aligned:
                    span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE[rule_type],[token])
                    alignment[cur_var].append(span)
                else:
                    print >> sys.stderr, 'Variable/Concept %s/%s cannot be aligned'%(cur_var,cur_concept)
                    update = False

            elif rule_type == "thing":
                if 'ARG1-of' in amr[cur_var]:
                    k_var = amr[cur_var]['ARG1-of'][0]
                    success, sent, tokens = self.align_single_concept(sent,tokens,k_var,amr,alignment,unmatched_vars,triples)
                    if success:
                        span = alignment[k_var][0]
                        span.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type])
                        alignment[cur_var].append(span)
                    else:
                        update = False
                else:
                    rule_type = 'SingleConcept'

            elif rule_type == 'OrdinalEntity':
                val = amr[cur_var]['value'][0]
                success, sent, tokens = self.align_single_concept(sent,tokens,val,amr,alignment,unmatched_vars,triples)
                self.remove_aligned_concepts(cur_var,'value',val,unmatched_vars,triples)
                span = alignment[val][0]
                span.set_entity_tag(Aligner.ENTITY_TAG_TABLE[rule_type])
                alignment[cur_var].append(span)

            elif rule_type == 'RelativePosition':
                if 'direction' in amr[cur_var]:
                    dir_var = amr[cur_var]['direction'][0]
                    if amr.node_to_concepts[dir_var] == 'away':
                        aligned = False
                        for i,tok in tokens:
                            if tok.lower() == 'from':
                                aligned = True
                                break
                        if aligned:
                            span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE[rule_type],[tok])
                            alignment[cur_var].append(span)
                            alignment[dir_var].append(span)
                        else:
                            print >> sys.stderr, 'Variable/Concept %s/%s cannot be aligned'%(cur_var,cur_concept)
                            update = False
                    else:
                        rule_type = 'SingleConcept'
                else:
                    rule_type = 'SingleConcept'
                
            elif self.is_ago(cur_var,cur_concept,amr):
                k_var = amr[cur_var]['op1'][0]
                aligned = False
                for i,tok in tokens:
                    if tok.lower() == 'ago':
                        aligned = True
                        break
                if aligned:
                    span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE['ago'],[tok])
                    alignment[cur_var].append(span)
                    alignment[k_var].append(span)
                else:
                    print >> sys.stderr, '(%s/%s) :op1 (%s/%s) cannot be aligned'%(cur_var,cur_concept,k_var,amr.node_to_concepts[k_var])
                    update = False

            elif self.is_why_question(cur_var,amr):
                arg0_var = amr[cur_var]['ARG0'][0]
                aligned = False
                for i,tok in tokens:
                    if tok.lower() == 'why':
                        aligned = True
                        break
                if aligned:
                    span = Span(i,i+1,Aligner.ENTITY_TAG_TABLE['cause'],[tok])
                    alignment[cur_var].append(span)
                    alignment[arg0_var].append(span)
                else:
                    print >> sys.stderr, '(%s/%s) :op1 (%s/%s) cannot be aligned'%(cur_var,cur_concept,arg0_var,amr.node_to_concepts[arg0_var])
                    update = False
            else:
                pass

            if rule_type == "SingleConcept":
                update,span = self.try_align_as_single_concept(cur_var,cur_concept,amr,alignment,tokens,unmatched_vars,triples)
            elif cur_var in alignment:
                pass
            else:
                print >> sys.stderr, 'Can not find type of concept %s / %s'%(cur_var,cur_concept)

            # update
            #print cur_concept,rule_type
            if update:
                tokens = [(i,tk) for i,tk in tokens if i not in range(span.start,span.end)]
                sent = ' '.join(x for i,x in tokens)
                if self.verbose > 2:
                    print >> sys.stderr, "Concept '%s' Matched to span '%s' "%(cur_concept,' '.join(w for i,w in enumerate(sentence.split()) if i+1 in range(span[0],span[1])))
                    print sent
                    print alignment
                    
                    #raw_input('ENTER to continue')
            return update, sent, tokens

    def try_align_as_single_concept(self,cur_var,cur_concept,amr,alignment,tokens,unmatched_vars,triples):
        span = None
        update = True
        rule_type = 'SingleConcept'
        tmp = cur_concept.rsplit('-',1)
        sense = None 
        if not isinstance(cur_var,StrLiteral) and len(tmp) == 2 and re.match('[0-9]+',tmp[1]):
            sense = tmp[1]
            cur_concept = tmp[0].lower()
            
        for idx,token in tokens:
            t = token.lower()
            cur_concept = cur_concept.lower()
            if t == cur_concept:  # exact match
                span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                break
            elif self.fuzzy_match(t,cur_concept,Aligner.fuzzy_max_len):
                span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                break
            elif self.is_neg_form(t,cur_concept):
                #print cur_concept
                neg_var = None
                span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE['NegForm'],[t])
                if 'polarity' in amr[cur_var]:
                    neg_var = amr[cur_var]['polarity'][0]
                    self.remove_aligned_concepts(cur_var,'polarity',neg_var,unmatched_vars,triples)
                    alignment[neg_var].append(span)                    
                elif 'possible' in amr[cur_var]:
                    posb_var = amr[cur_var]['possible'][0]
                    neg_var = amr[posb_var]['polarity'][0]
                    alignment[posb_var].append(span)
                    alignment[neg_var].append(span)                    
                else:
                    pass

                break
            elif self.WN_lemma_match(t,cur_concept,sense):
                span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                break
            elif self.is_spec_form(t,cur_concept):
                span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                break
                #elif len(cur_concept) > 1 and self.is_plural(t,cur_concept):
                #    span = Span(idx,idx+1,Aligner.ENTITY_TAG_TABLE[rule_type],[t])
                #    break
            else:
                pass

        if span:
            alignment[cur_var].append(span)
        else:
            print >> sys.stderr, 'Variable/Concept %s/%s cannot be aligned'%(cur_var,cur_concept)
            #alignment[matched_variable].append(matched_variable)
            update = False                    
        return update,span
                
    def print_align_result(self,alignment,amr):

        output = ''

        for var in alignment:
            if var == 'root':
                continue
            spans = alignment[var]
            if var in amr.node_to_concepts:
                concept = amr.node_to_concepts[var]
            else:
                concept = var
            sntchunks = ','.join(str(span.start)+':'+' '.join(w for w in span.words) for span in spans)
            output += '%s/%s -> %s;'%(var,concept,sntchunks)
        return output
        
    def is_plural(self,token,concept):
        candidates = [concept+suffix for suffix in Aligner.plural_suffix]
        for c in candidates:
            if c == token:
                return True
        return False

    def is_neg_form(self,token,concept):
        neg_concepts = [prefix+concept for prefix in Aligner.neg_prefix]
        for neg in neg_concepts:
            if neg == token or self.fuzzy_match(token,neg,7):
                return True
        return False

    def is_ago(self,cur_var,cur_concept,amr):
        if cur_concept == 'before':
            op1 = amr[cur_var]['op1'][0] if 'op1' in amr[cur_var] else None
            if op1:
                return amr.node_to_concepts[op1] == 'now'
        return False

    def is_why_question(self,cur_var,amr):
        if 'ARG0' in amr[cur_var] and cur_var in amr.node_to_concepts:
            cur_concept = amr.node_to_concepts[cur_var]
            arg0 = amr[cur_var]['ARG0'][0]
            arg0_concept = amr.node_to_concepts[arg0]
            if cur_concept == 'cause-01' and arg0_concept == 'amr-unknown':
                return True
        return False

    def is_spec_form(self,token,concept):
        if concept in Aligner.special_morph_table and token in  Aligner.special_morph_table[concept]:
            return True
        else:
            return False
    def fuzzy_match(self,token,concept,max_len = 4):
        if len(token) < max_len:
            return False
        elif concept.startswith(token[:max_len]):
            return True
        else:
            return False
    
    def WN_lemma_match(self,token,concept,sense):
        '''wordnet lemma'''
        if sense:
            pos_tag = 'v'
        else:
            pos_tag = 'n'
        lemma = Aligner.lmtzr.lemmatize(token,pos_tag) 
        if lemma == concept:
            return True
        else:
            return False
        
    def _search_sent(self,pattern,sent,tokens):
        '''search for the first occurrence of pattern in sentence
           return its span
        '''
        #print pattern.pattern
        #print sent
        
        
        m = pattern.search(sent)
        if m:                        
            items = m.group().split()

            if m.group()[0] == ' ' and m.group()[-1] == ' ':
                spans = [(tokens[i][0],tokens[i][0]+len(items)) for i in range(len(tokens)) \
                         if (sum(map(lambda x:len(x[1])+1,tokens[:i]))-1,sum(map(lambda x:len(x[1])+1,tokens[:i]))-1+len(m.group())) == m.span()]
            else:
                spans = [(tokens[i][0],tokens[i][0]+len(items)) for i in range(len(tokens)) \
                         if (sum(map(lambda x:len(x[1])+1,tokens[:i])),sum(map(lambda x:len(x[1])+1,tokens[:i]))+len(m.group())) == m.span()]
            return spans[0]
        else:
            print pattern.pattern
            print sent
            raise SearchException("WARNING:Unable to find the matched span in sentence!")
            #return (-1,-1)
            
            
                
    def remove_aligned_concepts(self,cur_var,rel,child_var,var_list,triples):
        """remove childrens of aligned concepts"""
        #print triples
        #print var_list

        i = triples.index((rel,cur_var,child_var))
        var_list.pop(i)
        triples.pop(i)
