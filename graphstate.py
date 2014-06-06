#!/usr/bin/python

# parsing state representing a subgraph
# initialized with dependency graph
#

import copy
import cPickle
from parser import *

class ActionError(Exception):
    pass    
    
class ActionTable(dict):
    '''to do'''
    def add_action(self,action_name):
        key =  len(self.keys())+1
        self[key] = action_name
    
class GraphState(object):
    """
    Starting from dependency graph, each state represents subgraph in parsing process
    Indexed by current node being handled
    """
    
    sent = None
    deptree = None
    action_table = None
    new_actions = None
    stateID = 0
    
    def __init__(self,sigma,A):
        self.sigma = sigma
        self.idx = self.sigma.top()
        self.beta = Buffer(A.nodes[self.idx].children[:]) if self.idx != -1 else None
        if self.beta:
            self.cidx = self.beta.top()
        else:
            self.cidx = None
        self.A = A


    @staticmethod
    def init_state(depGraph,sent):        
        seq = depGraph.postorder()
        seq.append(-1)
        sigma = Buffer(seq)
        print sigma
        GraphState.sent = "ROOT "+sent
        GraphState.deptree = depGraph
        GraphState.stateID += 1
        return GraphState(sigma,copy.deepcopy(depGraph))

    @staticmethod
    def init_action_table():
        actionTable = ActionTable()
        actionTable[1] = 'next'
        actionTable[2] = 'delete_edge'
        actionTable[3] = 'swap' 
        #actionTable[4] = 'change_head'
        actionTable[4] = 'add_child'
        actionTable[5] = 'replace_head'
        actionTable[6] = 'merge'
        GraphState.new_actions = set()
        GraphState.action_table = actionTable

    def pcopy(self):
        return cPickle.loads(cPickle.dumps(self,-1))
    
    def is_terminal(self):
        """done traverse the graph"""
        return self.idx == -1
    
    def is_permissible(self,action):
        #TODO
        return True

    def getCurrentNode(self):
        return self.A.nodes[self.idx]

    def getCurrentChild(self):
        if self.cidx:
            return self.A.nodes[self.cidx]
        else:
            return None

    def apply(self,action,parameter):
        if parameter:
            return getattr(self,GraphState.action_table[action])(parameter)
        return getattr(self,GraphState.action_table[action])()

    def next(self):
        newstate = self.pcopy()
        if len(newstate.beta) >= 1:
            newstate.beta.pop()            
            newstate.cidx = newstate.beta.top() if newstate.beta else None
        else:
            newstate.sigma.pop()
            newstate.idx = newstate.sigma.top()
            newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children) if newstate.idx != -1 else None
            newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate
        
    def delete_edge(self):
        newstate = self.pcopy()
        newstate.A.remove_edge(newstate.idx,newstate.cidx)
        newstate.beta.pop()
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate
    
    def swap(self):
        newstate = self.pcopy()
        newstate.A.swap_head2(newstate.idx,newstate.cidx)
        #newstate.idx = newstate.cidx
        tmp = newstate.sigma.pop()
        newstate.sigma.push(newstate.cidx)
        newstate.sigma.push(tmp)
        newstate.beta.pop()
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate
    '''
    def change_head(self,goldParent):
        newstate = self.pcopy()
        newstate.A.remove_edge(newstate.idx,newstate.cidx)
        newstate.A.addEdge(goldParent,newstate.cidx)
        newstate.A.relativePos(newstate.cidx,goldParent)
    '''
    
    def add_child(self,node_to_add):
        newstate = self.pcopy()
        newstate.A.addEdge(newstate.idx,node_to_add)
#        hoffset,voffset = GraphState.deptree.relativePos(newstate.idx,node_to_add)
        atype = GraphState.deptree.relativePos2(newstate.idx,node_to_add)
        #self.new_actions.add('add_child_('+str(hoffset)+')_('+str(voffset)+')_'+str(self.stateID))
        self.new_actions.add('add_child_%s_%s'%(atype,str(self.stateID)))
        return newstate

    def replace_head(self):
        """
        Use current child to replace current node
        """
        newstate = self.pcopy()
        newstate.A.swap_head(newstate.idx,newstate.cidx)
        newstate.A.remove_edge(newstate.cidx,newstate.idx)
        newstate.sigma.pop()
        newstate.idx = newstate.sigma.top()
        newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children)
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate

    def merge(self):
        """
        merge nodes to form entity 
        """
        newstate = self.pcopy()
        tmp = newstate.idx
        newstate.idx = newstate.idx if newstate.idx < newstate.cidx else newstate.cidx
        newstate.cidx = newstate.cidx if tmp < newstate.cidx else tmp
        newstate.A.merge_node(newstate.idx,newstate.cidx)
        newstate.beta = Buffer(newstate.A.nodes[newstate.idx].children[:])
        newstate.cidx = newstate.beta.top() if newstate.beta else None
        return newstate
        
    def print_config(self):
        if self.cidx:
            return 'ID:%s Parent:(%s) Child:(%s)'%(str(self.stateID),','.join(GraphState.sent.split()[self.idx:self.A.nodes[self.idx].end]),\
                                             ','.join(GraphState.sent.split()[self.cidx:self.A.nodes[self.cidx].end]))
        else:
            return 'ID:%s Parent:(%s) children:%s'%(str(self.stateID),','.join(GraphState.sent.split()[self.idx:self.A.nodes[self.idx].end]),\
                                              ['('+','.join(GraphState.sent.split()[c:self.A.nodes[c].end])+')' for c in self.A.nodes[self.idx].children])

