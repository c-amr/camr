#!/usr/bin/python

# transition-based (incremental) AMR parser
# author Chuan Wang
# March 28,2014

from common.util import *
from constants import *
from graphstate import GraphState
from newstate import Newstate
import optparse
import sys,copy,time,datetime
import numpy as np
from perceptron import Perceptron
import cPickle as pickle
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

DRAW_GRAPH = False
WRITE_FAKE_AMR = False
OUTPUT_PARSED_AMR = True

class Parser(object):
    """
    """
    State = None
    #model = None
    oracle = None
    #action_table = None
    cm = None # confusion matrix for error analysis
    rtx = None # array for store the rumtime data
    rty = None #
    
    def __init__(self,model=None,oracle_type=DETERMINE_TREE_TO_GRAPH_ORACLE_SC,action_type='basic',verbose=1,elog=sys.stdout):
        self.sent = ''
        self.oracle_type=oracle_type
        self.verbose = verbose
        self.elog = elog
        self.model = model
        if self.oracle_type == DETERMINE_TREE_TO_GRAPH_ORACLE:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table(ACTION_TYPE_TABLE[action_type])
            Parser.oracle = __import__("oracle").DetOracle(self.verbose)
        elif self.oracle_type == DETERMINE_TREE_TO_GRAPH_ORACLE_SC:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table(ACTION_TYPE_TABLE[action_type])
            Parser.oracle = __import__("oracle").DetOracleSC(self.verbose)
        elif self.oracle_type ==  DETERMINE_STRING_TO_GRAPH_ORACLE:
            Parser.State = __import__("newstate").Newstate
        else:
            pass
        self.perceptron = Perceptron(model)
        Parser.State.model = model

            
    def get_best_act(self,scores,actions):
        best_label_index = None
        best_act_ind = np.argmax(map(np.amax,scores))
        if actions[best_act_ind]['type'] in ACTION_WITH_EDGE:
            best_label_index = scores[best_act_ind].argmax()
        return best_act_ind, best_label_index

    def parse_corpus_train(self, instances, interval=500):
        start_time = time.time()
        n_correct_total = .0
        n_parsed_total = .0
        #n_gold_total = .0
        
        for i,inst in enumerate(instances,1):
            #per_start_time = time.time()
            _,state = self.parse(inst)
            #print 'Parsing %s instances takes %s'%(str(inst.sentID),datetime.timedelta(seconds=round(time.time()-per_start_time,0)))
            _,n_correct_arc,n_parsed_arc,_,_,_,_ = state.evaluate()
            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            #n_gold_total += n_gold_arc

            if i % interval == 0:
                p = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
                #r = n_correct_total/n_gold_total if n_gold_total != .0 else .0
                print >> self.elog,"Over "+str(i)+" sentences ","Accuracy:%s" % (p)

        print >> self.elog,"One pass on %s instances takes %s" % (str(i),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        pt = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
        #r = n_correct_total/n_gold_total
        #f = 2*p*r/(p+r)
        print >> self.elog,"Total Accuracy: %s" % (pt)

    def parse_corpus_test(self, instances):
        Parser.cm = np.zeros(shape=(len(GraphState.action_table),len(GraphState.action_table)))
        Parser.rtx = []
        Parser.rty = []
        Parser.steps = []

        start_time = time.time()
        parsed_amr = []
        n_correct_labeled_total = .0
        n_correct_total = .0
        n_parsed_total = .0
        n_gold_total = .0

        n_correct_tag_total = .0
        n_parsed_tag_total = .0
        brackets = defaultdict(set)
        results = []
        #n_gold_tag_total = .0
        #cm_total = np.zeros(shape=(len(GraphState.action_table),len(GraphState.action_table)))
        #if WRITE_FAKE_AMR: out_fake_amr = open('data/fake_amr_triples.txt','w')
         
        for i,inst in enumerate(instances,1):
            per_start_time = time.time()
            step,state = self.parse(inst,train=False)
            per_parse_time = round(time.time()-per_start_time,3)
            
            Parser.rtx.append(len(inst.tokens))
            Parser.rty.append(per_parse_time)
            Parser.steps.append(step)

            n_correct_labeled_arc,n_correct_arc,n_parsed_arc,n_gold_arc,n_correct_tag,n_parsed_tag,_ = state.evaluate()

            p = n_correct_arc/n_parsed_arc if n_parsed_arc else .0
            r = n_correct_arc/n_gold_arc if n_gold_arc else .0
            f = 2*p*r/(p+r) if p+r != .0 else .0
            '''
            results.append(f)

            if f <= 0.4 and f >= .0:
                brackets['0-40'].add(inst.sentID)
            elif f <= 0.6 and f > 0.4:
                brackets['40-60'].add(inst.sentID)
            else:
                brackets['60-100'].add(inst.sentID)
            '''
            n_correct_labeled_total += n_correct_labeled_arc
            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            n_gold_total += n_gold_arc

            n_correct_tag_total +=  n_correct_tag
            n_parsed_tag_total +=  n_parsed_tag
            ##########################
            #gold edge labeled amr; gold tag labeled amr ;for comparison
            #garc_graph = state.get_gold_edge_graph()                
            #parsed_amr.append(GraphState.get_parsed_amr(garc_graph))            
            #
            #gtag_graph = state.get_gold_tag_graph()
            #parsed_amr.append(GraphState.get_parsed_amr(gtag_graph))            
            
            #g_graph = state.get_gold_label_graph()
            #parsed_amr.append(GraphState.get_parsed_amr(g_graph))            
            ############################


            parsed_amr.append(GraphState.get_parsed_amr(state.A))
            
            
        print >> self.elog,"Parsing on %s instances takes %s" % (str(i),datetime.timedelta(seconds=round(time.time()-start_time,0)))
        p = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
        r = n_correct_total/n_gold_total
        f = 2*p*r/(p+r)
        print >> self.elog,"Unlabeled Precision:%s Recall:%s F1:%s" % (p,r,f)

        lp = n_correct_labeled_total/n_parsed_total
        lr = n_correct_labeled_total/n_gold_total
        lf = 2*lp*lr/(lp+lr)
        print >> self.elog,"Labeled Precision:%s Recall:%s F1:%s" % (lp,lr,lf)

        tp = n_correct_tag_total/n_parsed_tag_total
        print >> self.elog,"Tagging Precision:%s" % (tp)

        
        #pickle.dump((Parser.rtx,Parser.rty,Parser.steps),open('draw-graph/rt.pkl','wb'))
        #plt.plot(Parser.rtx,Parser.rty,'o')
        #plt.savefig('draw-graph/rt.png')
        #plt.plot(Parser.rtx,Parser.steps,'o')
        #plt.xlabel('Sentence length')
        #plt.ylabel('Actions')
        #plt.savefig('draw-graph/rt-act.png')

        print "Confusion matrix action class:"
        np.set_printoptions(suppress=True)
        print np.round(np.divide(Parser.cm,10))
        return parsed_amr

        ##############################
        #import random
        #print random.sample(brackets['0-40'],10)
        #print random.sample(brackets['40-60'],10)
        #print random.sample(brackets['60-100'],10)        
        
        #return results

    def _parse(self,instance):
        self.perceptron.no_update()
        return (True,Parser.State.init_state(instance,self.verbose))
    
    def parse(self,instance,train=True): 
        # no beam; pseudo deterministic oracle
        state = Parser.State.init_state(instance,self.verbose)
        ref_graph = instance.gold_graph
        step = 0
        pre_state = None
        
        while not state.is_terminal():
            if self.verbose > 2:
                print >> sys.stderr, state.print_config()
                
            #start_time = time.time()    
            actions = state.get_possible_actions(train)
            #print "Done getactions, %s"%(round(time.time()-start_time,2))
            if train:
                features = map(state.make_feat,actions)
                scores = map(state.get_score,(act['type'] for act in actions),features)
            
                best_act_ind, best_label_index = self.get_best_act(scores,actions)

                #print "Done argmax, %s"%(round(time.time()-start_time,2))
                #gold_act = getattr(self,self.oracle_type)(state,ref_graph)
                gold_act, gold_label = Parser.oracle.give_ref_action(state,ref_graph)
                #gold_act_type = dict([(t,v) for t,v in gold_act.items() if t in ['type','parent_to_add','parent_to_attach','tag']])
                #if 'edge_label' in gold_act:
                #    gold_label_index = Parser.State.model.rel_codebook.get_index(gold_act['edge_label'])
                #else:
                #    gold_label_index = None
                

                try:
                    gold_act_ind = actions.index(gold_act)
                except ValueError:
                    if self.verbose > 2:
                        print >> sys.stderr, 'WARNING: gold action %s not in possible action set %s'%(str(gold_act),str(actions))
                        #import pdb
                        #pdb.set_trace()
                    actions.append(gold_act)
                    gold_act_ind = len(actions)-1
                    features.append(state.make_feat(gold_act))
                gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)

                if self.verbose > 2:
                    print >> sys.stderr, "Step %s:take action %s gold action %s | State:sigma:%s beta:%s\n" % (step,actions[best_act_ind],actions[gold_act_ind],state.sigma,state.beta)
                    
                if gold_act_ind != best_act_ind or gold_label_index != best_label_index:
                    self.perceptron.update_weight_one_step(actions[gold_act_ind]['type'],features[gold_act_ind],gold_label_index,actions[best_act_ind]['type'],features[best_act_ind],best_label_index)
                    
                    best_act_ind = gold_act_ind
                    #best_tag_index = gold_tag_index
                    best_label_index = gold_label_index
                else:
                    self.perceptron.no_update()

                #print "Done update, %s"%(round(time.time()-start_time,2))
                #raw_input('ENTER TO CONTINUE')
            else:
                features = map(state.make_feat,actions)
                scores = map(state.get_score,(act['type'] for act in actions),features,[train]*len(actions))
            
                best_act_ind, best_label_index = self.get_best_act(scores,actions)
                self.evaluate_actions(actions[best_act_ind],best_label_index,state,ref_graph)
                gold_act, gold_label = Parser.oracle.give_ref_action(state,ref_graph)
                
                best_label = Parser.State.model.rel_codebook.get_label(best_label_index) if best_label_index is not None else None
                
                if self.verbose > 2:
                    print >> sys.stderr, "Step %s: (%s,%s) | take action %s, edge_label:%s | gold action %s,edge label:%s | State:sigma:%s beta:%s" % (step,actions[best_act_ind]['type'],gold_act['type'],actions[best_act_ind],best_label,gold_act,gold_label,state.sigma,state.beta)
                if self.verbose > 3:
                    # REATTACH NEXT pair error
                    if actions[best_act_ind]['type'] != NEXT1:
                        self.output_weight(best_act_ind,best_label_index,features,actions)
                        if gold_act['type'] == NEXT1 and gold_act in actions:
                            gold_act_ind = actions.index(gold_act)
                            gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                            self.output_weight(gold_act_ind,gold_label_index,features,actions)
                    # NEXT REATTACH pair error
                    if actions[best_act_ind]['type'] == NEXT1:
                        self.output_weight(best_act_ind,best_label_index,features,actions)
                        if gold_act['type'] == REATTACH and gold_act in actions:
                            gold_act_ind = actions.index(gold_act)
                            gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                            self.output_weight(gold_act_ind,gold_label_index,features,actions)

                    #if actions[best_act_ind]['type'] == REPLACEHEAD:
                    #    self.output_weight(best_act_ind,best_label_index,features,actions)
                    #    if gold_act['type'] == NEXT1 and gold_act in actions:
                    #        gold_act_ind = actions.index(gold_act)
                    #        gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                    #        self.output_weight(gold_act_ind,gold_label_index,features,actions)
                    #if actions[best_act_ind]['type'] == MERGE:
                    #    self.output_weight(best_act_ind,best_label_index,features,actions)
                    #    if gold_act['type'] == NEXT1 and gold_act in actions:
                    #        gold_act_ind = actions.index(gold_act)
                    #        gold_label_index = Parser.State.model.rel_codebook.get_index(gold_label)
                    #        self.output_weight(gold_act_ind,gold_label_index,features,actions)
                        
            act_to_apply = actions[best_act_ind]
            act_to_apply['edge_label'] = Parser.State.model.rel_codebook.get_label(best_label_index) if best_label_index is not None else None
            pre_state = state
            state = state.apply(act_to_apply)
            
            step += 1

        if self.verbose == 1:
            print >> sys.stderr, pre_state.print_config()

        return (step,state)

    def output_weight(self,act_ind,label_index,feats,actions):
        '''for debug '''
        label_ind = label_index if label_index is not None else 0
        feats_fired = feats[act_ind]
        act_idx = GraphState.model.class_codebook.get_index(actions[act_ind]['type']) 
        weight = GraphState.model.avg_weight[act_idx]
        feat_idx = map(GraphState.model.feature_codebook[act_idx].get_index,feats_fired)
        weight_sum = np.sum(weight[ [i for i in feat_idx if i is not None] ],axis = 0)
        #weight_fired = weight[[i for i in feat_idx if i is not None]]
        try:
            print >> sys.stderr, '\n'.join('%s,%f'%(f,weight[i][label_ind]) if i is not None else '%s,%f'%(f,0.0)  for f,i in zip(feats_fired,feat_idx))
            print >> sys.stderr, 'Sum: %f \n\n'%(weight_sum[label_ind])
        except TypeError:
            import pdb
            pdb.set_trace()
        #print >> sys.stderr,Parser.State.model.rel_codebook.get_label(0)
        
    def evaluate_actions(self,best_act,best_label_index,cur_state,ref_graph):
        gold_act,gold_label = Parser.oracle.give_ref_action(cur_state,ref_graph)
        Parser.cm[gold_act['type'],best_act['type']] += 1.0

    def testUserGuide(self,instance):
        """take user input actions as guide"""
        state = Parser.State.init_state(instance,self.verbose)
        #for action in user_actions:
        while True:
            if state.is_terminal():
                return state

            print state.print_config()
            print state.A.print_tuples()
            action_str = raw_input('input action:')
            if not action_str:
                break                    
            act_type = int(action_str.split()[0])            
            if len(action_str) == 2:
                child_to_add = int(action_str.split()[1]) 
                action = {'type':act_type,'child_to_add':child_to_add}
            else:
                action = {'type':act_type}

            if state.is_permissible(action):
                state = state.apply(action)

            else:
                raise Error('Impermissibe action: %s'%(action))
            
        return state

    def draw_graph(self,fname,gtext):
        """ draw a graph using latex tikz/pgf """
        template = open("draw-graph/graph-template.tex",'r').read()
        fout = open("draw-graph/"+fname+".tex",'w')

        fout.write(template%(gtext))
        fout.close()
        
    def testOracleGuide(self,instance,start_step=0):
        """simulate the oracle's action sequence"""
            
        state = Parser.State.init_state(instance,self.verbose)
        ref_graph = state.gold_graph
        if state.A.is_root(): # empty dependency tree
            print >> sys.stderr, "Empty sentence! "+instance.text
            state.A = copy.deepcopy(ref_graph)
        step = 1
        if self.verbose > 1:
            #print "Gold graph:\n"+ref_graph.print_tuples()
            if DRAW_GRAPH:
                fname = "graph"+str(state.sentID)+"_gold"
                self.draw_graph(fname,ref_graph.getPGStyleGraph())

        while not state.is_terminal():
            if self.verbose > 0:
                print >> sys.stderr, state.print_config()
                #print state.A.print_tuples()                                    
                if DRAW_GRAPH:
                    fname = "graph"+str(state.sentID)+"_s"+str(step)
                    self.draw_graph(fname,state.A.getPGStyleGraph((state.idx,state.cidx)))

            #action = getattr(self,self.oracle_type)(state,ref_graph)
            action,label = Parser.oracle.give_ref_action(state,ref_graph)

            if self.verbose > 0:
                #print "Step %s:take action %s"%(step,action)
                print >> sys.stderr, "Step %s:take action %s, edge label %s | State:sigma:%s beta:%s" % (step,action,label,state.sigma,state.beta)
                print >> sys.stderr, [state.A.get_edge_label(state.idx,child) for child in state.A.nodes[state.idx].children if state.A.get_edge_label(state.idx,child).startswith('ARG') and child != state.cidx]
                if action['type'] in [REATTACH]:
                    node_to_add = action['parent_to_add'] if 'parent_to_add' in action else action['parent_to_attach']
                    path = state.A.get_path(state.cidx,node_to_add)
                    path_str=[(state.sent[i]['pos'],state.sent[i]['rel']) for i in path[1:-1]]
                    path_str.insert(0,state.sent[path[0]]['rel'])
                    path_str.append(state.sent[path[-1]]['rel'])
                    print >> sys.stderr,'path for attachment', path, path_str #Parser.State.deptree.path(state.cidx),Parser.State.deptree.path(node_to_add),Parser.State.deptree.get_path(state.cidx,node_to_add)
                if action['type'] not in [NEXT2,DELETENODE]:
                    path = GraphState.deptree.get_path(state.cidx,state.idx)
                    if state.A.nodes[state.idx].end - state.A.nodes[state.idx].start > 1:
                        path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1] if i not in range(state.A.nodes[state.idx].start,state.A.nodes[state.idx].end)]
                    else:
                        path_pos_str = [(GraphState.sent[i]['pos'],GraphState.sent[i]['rel']) for i in path[1:-1]]
                    path_pos_str.insert(0,GraphState.sent[path[0]]['rel'])
                    path_pos_str.append(GraphState.sent[path[-1]]['rel'])
                    print >> sys.stderr,'path for current edge', path, path_pos_str
                    print >> sys.stderr,'Deleted children','b0',sorted([GraphState.sent[j]['form'].lower() for j in state.A.nodes[state.cidx].del_child]),'s0',sorted([GraphState.sent[j]['form'].lower() for j in state.A.nodes[state.idx].del_child])
                    
            if state.is_permissible(action):
                action['edge_label'] = label
                state = state.apply(action)
                step += 1
                #if self.verbose > 2 and step > start_step:
                #    raw_input('ENTER to continue')
            else:
                raise Error('Impermissibe action: %s'%(action))
            
        return state


                                  
    def record_actions(self,outfile):
        output = open(outfile,'w')
        for act in list(Parser.State.new_actions):
            output.write(str(act)+'\n')
        output.close()

    '''
    def det_graph_oracle(self,state,ref_graph):

        def isCorrectReplace(childIdx,node,rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False
            
        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentNode = state.get_current_node()
        currentChild = state.get_current_child()
        currentGraph = state.A
        goldNodeSet = ref_graph.nodes.keys()

        result_act_type = None
        result_act_label = None
        if currentIdx in goldNodeSet:
            goldNode = ref_graph.nodes[currentIdx]
            #for child in currentNode.children: 
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode) or goldNode.contains(currentChild):
                        return {'type':MERGE} # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in goldChild.children and \
                       currentChildIdx in goldNode.children:
                        print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in goldChild.children:
                        return {'type':SWAP} # swap
                        #result_act_type = {'type':SWAP}                        
                    elif currentChildIdx in goldNode.children: # correct
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents]
                        if parents_to_attach:
                            return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]}
                        else:
                            return {'type':REATTACH}
                else:
                    if goldNode.contains(currentChild):
                        return {'type':MERGE}
                        #result_act_type = {'type':MERGE}
                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        return {'type':NEXT1}
                        
            else:
                if set(currentNode.children) == set(goldNode.children):
                    gold_tag = goldNode.tag
                    return {'type':NEXT2, 'tag':gold_tag} # next: done with the current node move to next one
                    #result_act_type = {'type':NEXT2,'tag':gold_tag}
                elif len(currentNode.children) < len(goldNode.children):
                    nodes_to_add = [c for c in goldNode.children if c not in currentNode.children]
                    child_to_add = nodes_to_add[0]
                    if child_to_add != 0 and child_to_add != currentIdx \
                       and child_to_add not in currentNode.children and child_to_add in currentGraph.nodes: 

                        gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                        return {'type':ADDCHILD, 'child_to_add':child_to_add, 'edge_label':gold_edge} # add one child each action
                        #result_act_type = {'type':ADDCHILD, 'child_to_add':nodes_to_add[0]}
                        #result_act_label = gold_edge
                    else:
                        if self.verbose > 2:
                            print >> sys.stderr, "Not a correct link between %s and %s!"%(currentIdx,nodes_to_add[0])
                        return {'type':NEXT2}
                        #result_act_type = {'type':NEXT2}
                else:
                    if self.verbose > 2:
                        print >> sys.stderr, "Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                    pass
                    
        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type':MERGE}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents]
                        if parents_to_attach:
                            return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]}
                        else:
                            return {'type':REATTACH}
                else:
                    return {'type':NEXT1}                    
            else:
                return {'type':NEXT2}

        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if (isCorrectReplace(currentChildIdx,currentNode,ref_graph) or len(currentNode.children) == 1):
                        return {'type':REPLACEHEAD} # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents]
                        if parents_to_attach:
                            return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]}
                        else:
                            return {'type':REATTACH}
                else:
                    #return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}
                    return {'type':NEXT1}
            else:
                # here currentNode.children must be empty
                return {'type':DELETENODE} 
                #result_act_type = {'type':DELETENODE}


        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip}
        
                
    def det_oracle2(self,state,ref_graph):
        def is_complete(node,ref_node):
            if set(node.children) == set(ref_node.children) and set(node.parents) == set(ref_node.parents) \
                    and (node.start,node.end) == (ref_node.start,ref_node.end):
                return True
            return False
        def need_pass(lefttop,node,ref_node):
            leftmostidx = node.getLeftmostchildidx()
            gold_leftmostidx = ref_node.getLeftmostchildidx()
            if leftmostidx < lefttop or gold_leftmostidx < lefttop:
                return True
            else:
                return False

        DepArc,leftidx,rightidx = state.cur_arc()
        if DepArc != -1:
            head = leftidx if DepArc == 0 else rightidx
            dep = rightidx if DepArc == 0 else leftidx
            head_node = state.A.nodes[head]
            dep_node = state.A.nodes[dep]
            
            if leftidx in ref_graph.nodes and ref_graph.nodes[leftidx].contains(state.A.nodes[rightidx]):
                return 4 # merge
            if head in ref_graph.nodes and dep not in ref_graph.nodes:
                return 2 # del child
            elif head not in ref_graph.nodes and dep in ref_graph.nodes:
                return 3 # swap
            elif head in ref_graph.nodes and dep in ref_graph.nodes:
                if head in ref_graph.nodes[dep].children:
                    #if dep in ref_graph.nodes[head].children:
                    #    raise Error('Circle between %s and %s'%(head,dep))
                    return 3 # swap
                elif not dep in ref_graph.nodes[head].children:
                    if True and need_pass(leftidx,state.A.nodes[rightidx],ref_graph.nodes[righidx]): # the dependent wrong edge
                        return 8 # delete edge
    '''
            
        
if __name__ == "__main__":
    pass
    

