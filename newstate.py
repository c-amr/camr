#!/usr/bin/python

# Implementation of the configuration in transition-based AMR parsing
# author Chuan Wang
# March 26, 2014

from common.util import *
from collections import deque
import copy
import cPickle

class ActionError(Exception):
    pass



class Action:
    NOSHIFT = 1
    DELETEC = 2
    SWAPHEAD = 3
    MERGE = 4
    #NOARC = 4
    LEFTREDUCE = 5
    LEFTPASS = 6
    RIGHTSHIFT = 7
    RIGHTPASS = 8
    NOREDUCE = 9
    DELETEARC = 10
    #DELETEPASS = 11
    NOPASS = 12

class Newstate(object):
    """
    configurations in transition system,
    1) sigma: a stack of dependency nodes (DNode) processed
    2) delta: a deque of nodes popped out of sigma but will be put back in sigma later to address coreference
    3) beta: a buffer of unprocessed nodes keep the sentence order
    4) A: A graph representation contains a set of triples
    """
    action_table = {Action.NOSHIFT:'noshift',
                    Action.DELETEC:'del_child',
                    Action.SWAPHEAD:'swap',
                    Action.MERGE:'merge',
                    #Action.NOARC:'no_arc',
                    Action.LEFTREDUCE:'leftreduce',
                    Action.LEFTPASS:'leftpass',
                    Action.RIGHTSHIFT:'rightshift',
                    Action.RIGHTPASS:'rightpass',
                    Action.NOREDUCE:'noreduce',
                    Action.DELETEARC:'deletearc',                
                    #Action.DELETEPASS:'deletepass',
                    Action.NOPASS:'nopass'
                }
    sent = None
    
    def __init__(self,sigma,delta,beta,A):
        self.sigma = sigma
        self.delta = delta
        self.beta = beta
        self.A = A

    @staticmethod
    def init_state(depGraph,sent):
        sigma = Stack([0])
        delta = deque()
        beta = Buffer(depGraph.nodes_list()[1:])
        A = depGraph
        State.sent = "ROOT "+sent
        return Newstate(sigma,delta,beta,A)        
    
    def pcopy(self):
        return cPickle.loads(cPickle.dumps(self,-1))

    def is_terminal(self):
        return self.beta.isEmpty()
    
    def is_permissible(self,action):
        #TODO
        return True
    
    def _getBufferStackPair(self):
        return tuple((self.sigma.top(),self.beta.top()))
    
    def cur_arc(self):
        """return current dependency arc corresponding to the buffer stack pair"""
        i,j = self._getBufferStackPair()
        lr = self.A.get_direction(i,j)
        return tuple((lr,i,j))

        '''
        if lr == 0:
            return tuple((0,i,j))
        elif lr == 1:
            return tuple((1,i,j))
        else:
            # no dependency arc between i,j
            return tuple((-1,i,j))
        '''
    def apply(self,action):
        return getattr(self,Newtate.action_table[action])()

    def noshift(self):
        newstate = self.pcopy()
        newstate.sigma.extend(newstate.delta)
        newstate.delta.clear()
        newstate.sigma.append(newstate.beta.pop())
        return newstate
    
    def del_child(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        lr = self.A.get_direction(i,j)
        if lr == 0:
            newstate.beta.pop()
            newstate.beta.push(newstate.sigma.pop())
            newstate.A.remove_node(j)
        elif lr == 1:
            newstate.sigma.pop()
            newstate.A.remove_node(i)
        else:
            raise ActionError('Impermissible action: DELETEC!')
        return newstate

    def swap(self):
        """switch head of the subtree formed by stack buffer top"""
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        tmp = newstate.sigma.pop()
        newstate.sigma.push(newstate.beta.pop())
        newstate.beta.push(tmp)
        lr = self.A.get_direction(i,j)
        if lr == 0:
            newstate.A.swap_head(i,j)
        elif lr == 1:
            newstate.A.swap_head(j,i)
        else:
            raise ActionError('Impermissible action: SWAPHEAD!')
        return newstate
    '''
    def no_arc(self):
        """
        put current stack top in delta list so that it can be 
        compared to nodes in beta later; 
        """
        newstate = self.pcopy()
        newstate.delta.appendleft(newstate.sigma.pop())
        return newstate
    '''
    def leftreduce(self):
        """additional semantic relation arc which forms a graph here"""
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        if i != 0:
            newstate.sigma.pop()
        newstate.A.addEdge(j,i)
        return newstate

    def leftpass(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        if i != 0:
            newstate.delta.appendleft(newstate.sigma.pop())
        newstate.A.addEdge(j,i)
        return newstate

    def rightshift(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        newstate.sigma.extend(newstate.delta)
        newstate.delta.clear()
        newstate.sigma.append(newstate.beta.pop())
        newstate.A.addEdge(i,j)
        return newstate
        
    def rightpass(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        newstate.delta.appendleft(newstate.sigma.pop())
        newstate.A.addEdge(i,j)
        return newstate
    
    def noreduce(self):
        """
        reduce completed node
        """
        newstate = self.pcopy()
        newstate.sigma.pop()
        return newstate
        
    def merge(self):
        """merge span"""
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        newstate.A.merge_node(i,j)
        newstate.beta.pop()
        return newstate
    '''
    def del_arc(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        lr = self.A.get_direction(i,j)
#        tmp = newstate.sigma.pop()
        if i != 0:
            newstate.beta.push(newstate.sigma.pop())
#        newstate.beta.push(tmp)
        if lr == 0:
            #newstate.beta.pop()
            #newstate.beta.push(newstate.sigma.pop())
            newstate.A.remove_edge(i,j)
        elif lr == 1:
            #newstate.sigma.pop()
            newstate.A.remove_edge(j,i)
        else:
            raise ActionError('Impermissible action: DELETEC!')
        return newstate
    '''
    def deletearc(self):
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        lr = self.A.get_direction(i,j)
        if lr == 0:
            #newstate.sigma.extend(newstate.delta)
            #newstate.delta.clear()
            #newstate.sigma.append(newstate.beta.pop())
            newstate.A.remove_edge(i,j)
        elif lr == 1:
            #newstate.sigma.pop()
            newstate.A.remove_edge(j,i)
        else:
            raise ActionError('Impermissible action: DELETEC!')
        return newstate

    def deletepass(self):
        newstate = self.copy()
        i,j = newstate._getBufferStackPair()
        lr = self.A.get_direction(i,j)
        newstate.delta.appendleft(newstate.sigma.pop())
        if lr == 0:
            newstate.A.remove_edge(i,j)
        elif lr == 1:
            newstate.A.remove_edge(j,i)
        else:
            raise ActionError('Impermissible action: DELETEC!')
        return newstate

    def nopass(self):
        """rather than reduce the stack top, save it"""
        newstate = self.pcopy()
        i,j = newstate._getBufferStackPair()
        newstate.delta.appendleft(newstate.sigma.pop())
        return newstate

    def print_config(self):
        return '[%s] [%s] [%s]'%(' '.join('('+' '.join(State.sent.split()[i:self.A.nodes[i].end])+')' for i in self.sigma),\
                                 ' '.join('('+' '.join(State.sent.split()[i:self.A.nodes[i].end])+')' for i in self.delta),\
                            ' '.join('('+' '.join(State.sent.split()[j:self.A.nodes[j].end])+')' for j in self.beta))
