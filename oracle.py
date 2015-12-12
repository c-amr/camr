#!/usr/bin/python

# class for different oracles
from constants import *
import sys

class Oracle():

    def __init__(self,verbose=0):
        self.verbose = verbose
    
    def give_ref_action(self):
        raise NotImplementedError('Not implemented!')


class DynOracle(Oracle):
    '''Using dynamic programming find the optimal action sequence
       for a initial state given gold graph
       The score of one action sequence is defined as 
       parsed graph and gold graph
    '''
    
    def give_ref_action_seq(self,state,ref_graph):
        pass
        
    def give_ref_action(self):
        pass


class DetOracleABT(Oracle):
    '''
       deterministic oracle try to infer the unaligned concepts given gold span graph
    '''
    def give_ref_action(self,state,ref_graph):
        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentGraph = state.A

        return self.give_ref_action_aux(currentIdx,currentChildIdx,currentGraph,ref_graph)
        
    def give_ref_action_aux(self,currentIdx,currentChildIdx,currentGraph,ref_graph):
        
        def isCorrectReplace(childIdx,node,rgraph):
            for p in node.parents:
                if p in rgraph.nodes and childIdx in rgraph.nodes[p].children:
                    return True
            return False

        def getABTParent(gn):
            for p in gn.parents:
                if not isinstance(p,int):
                    return p
            return None
        def need_infer(cidx,gn,cn,gnset,cg,rg):
            for p in cn.parents:
                #if p not in gnset and rg.isContained(p):
                #    return False
                if p in gn.parents or gn.contains(cg.nodes[p]) or p in gn.children: # correct/merge/swap arc
                    return False
            return True
        def is_aligned(abtIdx,rg,cg):
            return abtIdx in rg.abt_node_table and rg.abt_node_table[abtIdx] in cg.nodes
            
        def need_swap(idx,idx_p,cg,gc,rg):
            cur_head_p = cg.find_true_head(idx_p)
            try:
                cur_head = cg.abt_node_table[cur_head_p] if not isinstance(cur_head_p,int) else cur_head_p
            except KeyError:
                print >> sys.stderr,"inferred abt node cur_head_p not in ref graph!"
                return None
                
            
            if cur_head in gc.children:
                return cur_head
            cur_grand_head = getABTParent(rg.nodes[cur_head])
            if (not isinstance(cur_head,int) and cur_grand_head in gc.children):
                return cur_grand_head
            return None            
            
        if currentIdx == START_ID:
            return {'type':NEXT2},None            

        currentNode = currentGraph.nodes[currentIdx] #state.get_current_node()
        currentChild = currentGraph.nodes[currentChildIdx] if currentChildIdx in currentGraph.nodes else None #state.get_current_child()
        goldNodeSet = [ref_graph.abt_node_table[k] if k in ref_graph.abt_node_table else k for k in ref_graph.nodes] 
        result_act_type = None
        result_act_label = None


        
        if currentIdx in goldNodeSet:
            currentIdx_p = currentIdx
            if currentIdx not in ref_graph.nodes:
                currentIdx = currentGraph.abt_node_table[currentIdx]            
            goldNode = ref_graph.nodes[currentIdx]                
            
            #for child in currentNode.children: 
            if currentChildIdx:
                if currentChildIdx == START_ID:
                    abtParentIdx = getABTParent(goldNode)
                    #if state.action_history and state.action_history[-1]['type'] in [REPLACEHEAD,NEXT2,DELETENODE] 
                    if need_infer(currentIdx,goldNode,currentNode,goldNodeSet,currentGraph,ref_graph) \
                            and abtParentIdx and not is_aligned(abtParentIdx,ref_graph,currentGraph) \
                            and (len(ref_graph.nodes[abtParentIdx].children) == 1 or not isinstance(currentIdx_p,int) or ((goldNode.words[0] in NOMLIST or currentGraph.sent[currentIdx_p]['lemma'].lower() in NOMLIST) and len(goldNode.words) == 1)):
                        gold_tag = ref_graph.nodes[abtParentIdx].tag
                        abt_node_index = ABT_PREFIX+str(currentGraph.abt_node_num)
                        ref_graph.add_abt_mapping(abtParentIdx,abt_node_index)
                        currentGraph.add_abt_mapping(abt_node_index,abtParentIdx)
                        return {'type':INFER},gold_tag
                    else: 
                        return {'type':NEXT1},START_EDGE

                if currentChildIdx in goldNodeSet:
                    currentChildIdx_p = currentChildIdx
                    if currentChildIdx not in ref_graph.nodes:
                        currentChildIdx = currentGraph.abt_node_table[currentChildIdx]
                    try:
                        goldChild = ref_graph.nodes[currentChildIdx] 
                    except KeyError:
                        import pdb
                        pdb.set_trace()

                    if goldChild.contains(currentNode) or goldNode.contains(currentChild):
                        return {'type':MERGE}, None # merge
                        #result_act_type = {'type':MERGE}
                    #if currentIdx in goldChild.children and \
                    #   currentChildIdx in goldNode.children:
                    '''
                    if currentChildIdx in goldNode.children and ref_graph.is_cycle(currentIdx):
                        print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1}, gold_edge # next
                    '''
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    if currentChildIdx in goldNode.children: # correct
                        #parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_reentrance_constrained(currentIdx_p,currentChildIdx_p)]
                        parents_to_add = [p for p in goldChild.parents if p not in [currentGraph.abt_node_table[cp] if cp in currentGraph.abt_node_table else cp for cp in currentChild.parents]]
                        
                        if parents_to_add:
                            pta = parents_to_add[0]
                            if pta in ref_graph.abt_node_table: pta = ref_graph.abt_node_table[pta]
                            if pta in currentGraph.get_possible_parent_unconstrained(currentIdx_p,currentChildIdx_p):
                                gold_edge = ref_graph.get_edge_label(parents_to_add[0],currentChildIdx)
                                return {'type':REENTRANCE,'parent_to_add':pta},gold_edge
                            else:
                                gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                                return {'type':NEXT1}, gold_edge # next
                        else:
                            gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                            return {'type':NEXT1}, gold_edge # next

                    thead = need_swap(currentIdx,currentIdx_p,currentGraph,goldChild,ref_graph)
                    if thead is not None:#in goldChild.children:
                        gold_edge = ref_graph.get_edge_label(currentChildIdx,thead)
                        return {'type':SWAP}, gold_edge # swap
                        #result_act_type = {'type':SWAP}                        

                    parents_to_attach = [p for p in goldChild.parents if p not in [currentGraph.abt_node_table[cp] if cp in currentGraph.abt_node_table else cp for cp in currentChild.parents]]
                    if parents_to_attach:
                        pta = parents_to_attach[0]
                        if pta in ref_graph.abt_node_table: pta = ref_graph.abt_node_table[pta]
                        if pta in currentGraph.get_possible_parent_unconstrained(currentIdx_p,currentChildIdx_p):
                            gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                            return {'type':REATTACH,'parent_to_attach':pta},gold_edge
                        #elif not isinstance(pta,int): # abstract (or unaligned) nodes
                        #    abt_node_index = ABT_PREFIX+str(currentGraph.abt_node_num)
                        #    if pta == parents_to_attach[0]:
                        #        ref_graph.add_abt_mapping(pta,abt_node_index)
                        #        currentGraph.add_abt_mapping(abt_node_index,pta)
                        #    else:
                        #        currentGraph.add_abt_mapping(abt_node_index,parents_to_attach[0])
                        #    return {'type':INFER},None
                        else:
                            return {'type':NEXT1},None # violates the attachment constraints

                    else:
                        if self.verbose > 1: print >> sys.stderr, "Current child node %s doesn't have parents in gold span graph!"%(currentChildIdx)
                        return {'type':NEXT1},None
                        #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    if goldNode.contains(currentChild):
                        return {'type':MERGE},None
                        #result_act_type = {'type':MERGE}
                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        #k = ref_graph.isContained(currentChildIdx)
                        #if k:
                        #    return {'type':REATTACH,'parent_to_attach':k}
                        #else:    

                        return {'type':NEXT1},None
                        #return {'type':REATTACH,'parent_to_attach':None},None
                        
            else:
                #if len(currentNode.children) <= len(goldNode.children) and set(currentNode.children).issubset(set(goldNode.children)):
                #children_to_add = [c for c in goldNode.children if c not in currentNode.children and c in currentGraph.get_possible_children_constrained(currentIdx)]

                #if children_to_add:
                #    child_to_add = children_to_add[0]
                #    gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                #    return {'type':ADDCHILD,'child_to_add':child_to_add,'edge_label':gold_edge}
                #else:
                gold_tag = goldNode.tag
                return {'type':NEXT2, 'tag':gold_tag},None # next: done with the current node move to next one
                #else:
                #    if self.verbose > 2:
                #        print >> sys.stderr, "ERROR: Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                #    pass            
                
                                    
        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    #goldChild = ref_graph.nodes[currentChildIdx] if currentChildIdx in ref_graph.nodes else ref_graph.nodes[currentGraph.abt_node_table[currentChildIdx]]
                    currentChildIdx_p = currentChildIdx
                    if currentChildIdx not in ref_graph.nodes:
                        currentChildIdx = currentGraph.abt_node_table[currentChildIdx]
                    goldChild = ref_graph.nodes[currentChildIdx] 
                    if goldChild.contains(currentNode):
                        return {'type':MERGE},None
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx_p)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if ref_graph.nodes[parents_to_attach[0]].contains(currentNode):                                
                                return {'type':NEXT1},None # delay action for future merge
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None                                                                                                     
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None                    
            else:                
                return {'type':NEXT2},None

        else:
            '''
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if (isCorrectReplace(currentChildIdx,currentNode,ref_graph,state.beta) or len(currentNode.children) == 1):
                        return {'type':REPLACEHEAD},None # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if isCorrectReplace(parents_to_attach[0],currentNode,ref_graph,state.beta):
                                return {'type':NEXT1},None # delay action for future replace head
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    #return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}

                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None
                    '''
            if currentChildIdx:
                if currentChildIdx == START_ID:
                    return {'type':NEXT1},None
                if currentChildIdx in goldNodeSet:
                    #goldChild = ref_graph.nodes[currentChildIdx] if currentChildIdx in ref_graph.nodes else ref_graph.nodes[currentGraph.abt_node_table[currentChildIdx]]
                    currentChildIdx_p = currentChildIdx
                    if currentChildIdx not in ref_graph.nodes:
                        currentChildIdx = currentGraph.abt_node_table[currentChildIdx]
                    goldChild = ref_graph.nodes[currentChildIdx] 
                    if isCorrectReplace(currentChildIdx,currentNode,ref_graph) or len(currentNode.children) == 1: #current node's parents are already aligned
                        return {'type':REPLACEHEAD},None # replace head
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in [currentGraph.abt_node_table[cp] if cp in currentGraph.abt_node_table else cp for cp in currentChild.parents]]
                        if parents_to_attach:
                            pta = parents_to_attach[0]
                            if pta in ref_graph.abt_node_table: pta = ref_graph.abt_node_table[pta]
                            if pta in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx_p):
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                return {'type':REATTACH,'parent_to_attach':pta},gold_edge
                            #elif not isinstance(pta,int) and pta not in currentGraph.nodes: # abstract (or unaligned) nodes
                                
                                '''
                                #heuristic 1: just assign the abstract node index according to its first gold child  
                                #assert pta != currentIdx
                                if pta == parents_to_attach[0] and state.is_possible_align(currentIdx,parents_to_attach[0],ref_graph):
                                    ref_graph.add_abt_mapping(pta,currentIdx)
                                    currentGraph.add_abt_mapping(currentIdx,pta)
                                    gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                    return {'type':NEXT1},gold_edge           
                                else:
                                    return {'type':REPLACEHEAD},None
                                '''
                            else:
                                return {'type':NEXT1},None # violates the attachment constraints
                        else:
                            if self.verbose > 1: print >> sys.stderr, "Current child node %s doesn't have parents in gold span graph!"%(currentChildIdx)
                            return {'type':NEXT1},None
                else:
                    if self.verbose > 1: print >> sys.stderr, "Current child node %s should have been deleted!"%(currentChildIdx)
                    return {'type':NEXT1},None 
            else:
                # here currentNode.children must be empty
                if currentNode.children and self.verbose > 1: print >> sys.stderr, "Unaligned node %s has children"%(currentNode.start)
                # avoid delete name entity; will delete that in further reentrance action
                #if currentGraph.sent[currentIdx]['ne'] == 'O' or currentGraph.sent[currentIdx]['pos'] in FUNCTION_TAG:
                return {'type':DELETENODE},None
                #else:
                #    return {'type':NEXT2},None



        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip},None

class DetOracleSC(Oracle):
    '''
       deterministic oracle keeps strong connectivity of the graph
       1) using reattach rather than delete edge
       2) discard ADDCHILD
    '''

    def give_ref_action(self,state,ref_graph):

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
                        return {'type':MERGE}, None # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in goldChild.children and \
                       currentChildIdx in goldNode.children:
                        if self.verbose > 1: print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1}, gold_edge # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in goldChild.children:
                        gold_edge = ref_graph.get_edge_label(currentChildIdx,currentIdx)
                        return {'type':SWAP}, gold_edge # swap
                        #result_act_type = {'type':SWAP}                        
                    elif currentChildIdx in goldNode.children: # correct
                        parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_reentrance_constrained(currentIdx,currentChildIdx)]
                        #parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_add:
                            gold_edge = ref_graph.get_edge_label(parents_to_add[0],currentChildIdx)
                            return {'type':REENTRANCE,'parent_to_add':parents_to_add[0]},gold_edge
                        else:
                            gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                            return {'type':NEXT1}, gold_edge # next

                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                            #if gold_edge == 'x': # not actually gold edge, skip this 
                            #    return {'type':NEXT1},None
                            #else:
                            return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    if goldNode.contains(currentChild):
                        return {'type':MERGE},None
                        #result_act_type = {'type':MERGE}
                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        #k = ref_graph.isContained(currentChildIdx)
                        #if k:
                        #    return {'type':REATTACH,'parent_to_attach':k}
                        #else:    

                        return {'type':NEXT1},None
                        #return {'type':REATTACH,'parent_to_attach':None},None
                        
            else:
                #if len(currentNode.children) <= len(goldNode.children) and set(currentNode.children).issubset(set(goldNode.children)):
                #children_to_add = [c for c in goldNode.children if c not in currentNode.children and c in currentGraph.get_possible_children_constrained(currentIdx)]

                #if children_to_add:
                #    child_to_add = children_to_add[0]
                #    gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                #    return {'type':ADDCHILD,'child_to_add':child_to_add,'edge_label':gold_edge}
                #else:
                gold_tag = goldNode.tag
                return {'type':NEXT2, 'tag':gold_tag},None # next: done with the current node move to next one
                #else:
                #    if self.verbose > 2:
                #        print >> sys.stderr, "ERROR: Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                #    pass            
                
                                    
        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type':MERGE},None
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if ref_graph.nodes[parents_to_attach[0]].contains(currentNode):                                
                                return {'type':NEXT1},None # delay action for future merge
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None                                                                                                     
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None                    
            else:                
                return {'type':NEXT2},None

        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if isCorrectReplace(currentChildIdx,currentNode,ref_graph):# or len(currentNode.children) == 1:
                        return {'type':REPLACEHEAD},None # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if isCorrectReplace(parents_to_attach[0],currentNode,ref_graph):
                                return {'type':NEXT1},None # delay action for future replace head
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    #return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}

                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None
            else:
                # here currentNode.children must be empty
                return {'type':DELETENODE},None
                #result_act_type = {'type':DELETENODE}


        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip},None



        
class DetOracle(Oracle):
    '''
       deterministic oracle keeps weak connectivity of the graph
       1) using delete node rather than reattach
       2) discard ADDCHILD
    '''

    def give_ref_action(self,state,ref_graph):

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
                        return {'type':MERGE}, None # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in goldChild.children and \
                       currentChildIdx in goldNode.children:
                        if self.verbose > 1: print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1}, gold_edge # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in goldChild.children:
                        gold_edge = ref_graph.get_edge_label(currentChildIdx,currentIdx)
                        return {'type':SWAP}, gold_edge # swap
                        #result_act_type = {'type':SWAP}                        
                    elif currentChildIdx in goldNode.children: # correct
                        parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_reentrance_constrained(currentIdx,currentChildIdx)]
                        #parents_to_add = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_add:
                            gold_edge = ref_graph.get_edge_label(parents_to_add[0],currentChildIdx)
                            return {'type':REENTRANCE,'parent_to_add':parents_to_add[0]},gold_edge
                        else:
                            gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                            return {'type':NEXT1}, gold_edge # next

                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                            #if gold_edge == 'x': # not actually gold edge, skip this 
                            #    return {'type':NEXT1},None
                            #else:
                            return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    if goldNode.contains(currentChild):
                        return {'type':MERGE},None
                        #result_act_type = {'type':MERGE}
                    else:
                        #return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                        #k = ref_graph.isContained(currentChildIdx)
                        #if k:
                        #    return {'type':REATTACH,'parent_to_attach':k}
                        #else:    

                        return {'type':NEXT1},None
                        #return {'type':REATTACH,'parent_to_attach':None},None
                        
            else:
                #if len(currentNode.children) <= len(goldNode.children) and set(currentNode.children).issubset(set(goldNode.children)):
                #children_to_add = [c for c in goldNode.children if c not in currentNode.children and c in currentGraph.get_possible_children_constrained(currentIdx)]

                #if children_to_add:
                #    child_to_add = children_to_add[0]
                #    gold_edge = ref_graph.get_edge_label(currentIdx,child_to_add)
                #    return {'type':ADDCHILD,'child_to_add':child_to_add,'edge_label':gold_edge}
                #else:
                gold_tag = goldNode.tag
                return {'type':NEXT2, 'tag':gold_tag},None # next: done with the current node move to next one
                #else:
                #    if self.verbose > 2:
                #        print >> sys.stderr, "ERROR: Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
                #    pass            
                
                                    
        elif ref_graph.isContained(currentIdx):
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if goldChild.contains(currentNode):
                        return {'type':MERGE},None
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if ref_graph.nodes[parents_to_attach[0]].contains(currentNode):                                
                                return {'type':NEXT1},None # delay action for future merge
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None                                                                                                     
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None                    
            else:                
                return {'type':NEXT2},None

        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    goldChild = ref_graph.nodes[currentChildIdx]
                    if isCorrectReplace(currentChildIdx,currentNode,ref_graph):# or len(currentNode.children) == 1:
                        return {'type':REPLACEHEAD},None # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_constrained(currentIdx,currentChildIdx)]
                        #parents_to_attach = [p for p in goldChild.parents if p not in currentChild.parents and p in currentGraph.get_possible_parent_unconstrained(currentIdx,currentChildIdx)]
                        if parents_to_attach:
                            if isCorrectReplace(parents_to_attach[0],currentNode,ref_graph):
                                return {'type':NEXT1},None # delay action for future replace head
                            else:
                                gold_edge = ref_graph.get_edge_label(parents_to_attach[0],currentChildIdx)
                                #if gold_edge == 'x': # not actually gold edge, skip this
                                #    return {'type':NEXT1},None
                                #else:
                                return {'type':REATTACH,'parent_to_attach':parents_to_attach[0]},gold_edge
                        else:
                            return {'type':NEXT1},None
                            #return {'type':REATTACH,'parent_to_attach':None},None
                else:
                    #return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}

                    return {'type':NEXT1},None
                    #return {'type':REATTACH,'parent_to_attach':None},None
            else:
                # here currentNode.children must be empty
                return {'type':DELETENODE},None
                #result_act_type = {'type':DELETENODE}


        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip},None

    ''' 
    give local optimal action based on current state and the gold graph
    
    def give_ref_action_seq(self,state):
        pass

    def give_ref_action(self,state,ref_graph):

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
                    if ref_graph.nodes[currentChildIdx].contains(currentNode) or ref_graph.nodes[currentIdx].contains(currentChild):
                        return {'type':MERGE} # merge
                        #result_act_type = {'type':MERGE}
                    if currentIdx in ref_graph.nodes[currentChildIdx].children and \
                       currentChildIdx in ref_graph.nodes[currentIdx].children:
                        print >> sys.stderr, "Circle detected in gold graph!"
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    elif currentIdx in ref_graph.nodes[currentChildIdx].children:
                        return {'type':SWAP} # swap
                        #result_act_type = {'type':SWAP}                        
                    elif currentChildIdx in ref_graph.nodes[currentIdx].children: # correct
                        gold_edge = ref_graph.get_edge_label(currentIdx,currentChildIdx)
                        return {'type':NEXT1, 'edge_label':gold_edge} # next
                        #result_act_type = {'type':NEXT1}
                        #result_act_label = gold_edge
                    else:
                        return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
                else:
                    if ref_graph.nodes[currentIdx].contains(currentChild):
                        return {'type':MERGE}
                        #result_act_type = {'type':MERGE}
                    else:
                        return {'type':DELETEEDGE} # delete edge
                        #result_act_type = {'type':DELETEEDGE}
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
            if currentChildIdx and currentChildIdx in goldNodeSet and \
               ref_graph.nodes[currentChildIdx].contains(currentNode):
                return {'type':MERGE}
                #result_act_type = {'type':MERGE}
            else:
                if currentChildIdx:
                    return {'type':NEXT1}
                    #result_act_type = {'type':NEXT1}
                else:
                    return {'type':NEXT2}
                    #result_act_type = {'type':NEXT2}
        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    if (isCorrectReplace(currentChildIdx,currentNode,ref_graph) or len(currentNode.children) == 1):
                        return {'type':REPLACEHEAD} # replace head
                        #result_act_type = {'type':REPLACEHEAD}
                    else:
                        return {'type':NEXT1} #
                        #result_act_type = {'type':NEXT1}
                else:
                    return {'type':DELETEEDGE}
                    #result_act_type = {'type':DELETEEDGE}
            else:
                # here currentNode.children must be empty
                return {'type':DELETENODE} 
                #result_act_type = {'type':DELETENODE}


        skip = NEXT1 if currentChildIdx else NEXT2
        return {'type':skip}
        '''
