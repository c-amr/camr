#!/usr/bin/python

# transition-based (incremental) AMR parser
# author Chuan Wang
# March 28,2014

from common.util import *
from graphstate import GraphState
from newstate import Newstate
import optparse
import sys

        
class Parser(object):
    """
    """
    State = None
    #action_table = None

    def __init__(self,oracle_type=1,verbose=1):
        self.sent = ''
        self.oracle_type=oracle_type
        self.verbose = verbose
        if self.oracle_type == 1:
            Parser.State = __import__("graphstate").GraphState
            Parser.State.init_action_table()
        elif self.oracle_type == 2:
            Parser.State = __import__("newstate").Newstate
 

    def testUserGuide(self,user_actions,depGraph,sent):
        """take user input actions as guide"""
        state = self.State.init_state(depGraph,sent)
        for action in user_actions:
            if state.is_terminal():
                return state
            if state.is_permissible(action):
                state = state.apply(action)
                print state.print_config()
                print state.A.print_tuples()
                raw_input('ENTER to continue')
            else:
                raise Error('Impermissibe action: %s'%(action))
            
        return state

    def testOracleGuide(self,oracle,ref_graph,depGraph,sent,start_step=0):
        """test the oracle's output actions"""
        state = self.State.init_state(depGraph,sent)
        step = 1
        if self.verbose > 1:
            print "Gold graph:\n"+ref_graph.print_tuples()
        while not state.is_terminal():
            #if step == 20:
            #    import pdb
            #    pdb.set_trace()
            if self.verbose > 1:
                print state.print_config()
                print state.A.print_tuples()                                    

            action = getattr(self,oracle)(state,ref_graph)
            if isinstance(action,tuple):
                action,parameter = action
            else:
                action = getattr(self,oracle)(state,ref_graph)
                parameter = None

            if self.verbose > 1:
                print "Step %s:take action %s"%(step,action)

            if state.is_permissible(action):
                state = state.apply(action,parameter)
                step += 1
                if step > start_step:
                    raw_input('ENTER to continue')
            else:
                raise Error('Impermissibe action: %s'%(action))

    def record_actions(self,outfile):
        output = open(outfile,'w')
        for act in list(self.State.new_actions):
            output.write(str(act)+'\n')
        output.close()

    def det_oracle(self,state,ref_graph):
        """deterministic oracle: by using some rules to make the action unique 
        (one of the gold standard transition sequence)
        """
        def is_complete(node,ref_node):
            if set(node.children) == set(ref_node.children) and set(node.parents) == set(ref_node.parents) \
                    and (node.start,node.end) == (ref_node.start,ref_node.end):
                return True
            return False
        # rules 
        DepArc, leftidx, rightidx = state.cur_arc()
        
        if DepArc != -1: # exist dependency arc between stack buffer top
            head = leftidx if DepArc == 0 else rightidx
            dep = rightidx if DepArc == 0 else leftidx
            head_node = state.A.nodes[head]
            dep_node = state.A.nodes[dep]

            if leftidx in ref_graph.nodes and ref_graph.nodes[leftidx].contains(state.A.nodes[rightidx]):
                return 7 # merge
            if head in ref_graph.nodes and dep not in ref_graph.nodes:
                return 2 # del child
            elif head not in ref_graph.nodes and dep in ref_graph.nodes:
                return 3 # swap 
            elif head in ref_graph.nodes and dep in ref_graph.nodes:
                if head in ref_graph.nodes[dep].children:
                    #if dep in ref_graph.nodes[head].children:
                    #    raise Error('Circle between %s and %s'%(head,dep))
                    return 3 # swap
                elif not dep in ref_graph.nodes[head].children: # the dependent wrong edge
                    return 8 # delete edge
                elif not is_complete(state.A.nodes[leftidx],ref_graph.nodes[leftidx]) and leftidx != 0:
                    return 9 # attach
                    
        else:
            if leftidx in ref_graph.nodes and rightidx in ref_graph.nodes:
                if rightidx in ref_graph.nodes[leftidx].children: # additional arc
                    return 5 # rightarc
                elif leftidx in ref_graph.nodes[rightidx].children:
                    return 4 # leftarc
                elif not is_complete(state.A.nodes[leftidx],ref_graph.nodes[leftidx]) and leftidx != 0:
                    return 9 # attach

        if leftidx in ref_graph.nodes and is_complete(state.A.nodes[leftidx],ref_graph.nodes[leftidx]) and len(state.sigma) > 1:
            return 6 # reduce
        if len(state.beta) >= 1:
            return 1 # shift
                
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

    def det_tree_oracle(self,state,ref_graph):
        currentIdx = state.idx
        currentChildIdx = state.cidx
        currentNode = state.getCurrentNode()
        currentChild = state.getCurrentChild()
        currentGraph = state.A
        goldNodeSet = ref_graph.nodes.keys()
        
        if currentIdx in goldNodeSet:
            goldNode = ref_graph.nodes[currentIdx]
            #for child in currentNode.children: 
            if currentChildIdx:
                if currentChildIdx in goldNodeSet:
                    if ref_graph.nodes[currentChildIdx].contains(currentNode) or ref_graph.nodes[currentIdx].contains(currentChild):
                        return 6 # merge   
                    if currentIdx in ref_graph.nodes[currentChildIdx].children and \
                       currentChildIdx in ref_graph.nodes[currentIdx].children:
                        print >> sys.stderr, "Circle detected!"
                        return 1 # next
                    elif currentIdx in ref_graph.nodes[currentChildIdx].children:
                        return 3 # swap
                    elif currentChildIdx in ref_graph.nodes[currentIdx].children: # correct
                        return 1 # next
                    else:
                        return 2 # delete edge                
                else:
                    if ref_graph.nodes[currentIdx].contains(currentChild):
                        return 6 
                    else:
                        return 2 # delete edge
            else:
                if set(currentNode.children) == set(goldNode.children):
                    return 1 # next: done with the current node move to next one
                elif len(currentNode.children) < len(goldNode.children):
                    nodes_to_add = [c for c in goldNode.children if c not in currentNode.children]
                    if nodes_to_add[0] not in currentNode.parents:  # should not exist circle
                        return (4, nodes_to_add[0]) # add one child each action
                    else:
                        print >> sys.stderr, "Producing circle between %s and %s!"%(currentIdx,nodes_to_add[0])
                        return 1
                else:
                    print >> sys.stderr, "Missing actions, current node's and gold node's children:%s  %s"%(str(currentNode.children), str(goldNode.children))
        else:
            if currentChildIdx:
                #assert len(currentNode.children) == 1
                if currentChildIdx in goldNodeSet:
                    if ref_graph.nodes[currentChildIdx].contains(currentNode):
                        return 6
                    else:
                        return 5 # replace head
                else:
                    return 1
            else:
                return 1 # next
            
            
        
if __name__ == "__main__":
    pass
    
