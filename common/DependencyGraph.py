#!/usr/bin/python

"""
@date april 10
@author Chuan Wang
"""
import codecs
from util import ListMap,Stack

class DNode(object):
    def __init__(self,idx,str):
        self.index = idx
        self.str = str
        self.children = []
        self.parents = [] 
    def addChildren(self,c):
        if isinstance(c,list):
            self.children.extend(c)
        else:
            self.children.append(c)

    def addParent(self,parent):
        self.parents.append(parent)

    def removeChild(self,child):
        self.children.remove(child)
        
    def removeParent(self,parent):
        self.parents.remove(parent)
    def __str__(self):
        return 'Node: %s Children: %s'%(self.index,self.children)

class DepGraph(object):
    """
    Dependency graph adjacent link list implementation
    """
    LABELED = False

    def __init__(self):
        self.root = 0
        self.nodes = {}

        

    @staticmethod
    def init_graph(stp_deps):
        """Instantiate graph from dependency tuples"""
        dpg = DepGraph()
        #DepGraph.LABELED = LABELED

        for line in stp_deps:
            line = line.strip()
            label = line.split('(')[0]
            gov_str, gov_idx = line.split('(')[1].split(',')[0].split('-')
            gov_str = gov_str.strip()
            gov_idx = int(gov_idx)
            if not gov_idx in dpg.nodes.keys():
                gov_node = DNode(gov_idx,gov_str)
                dpg.addNode(gov_node)
            dep_str, dep_idx = line.strip().split('(')[1].split(',')[1][:-1].split('-')
            dep_str = dep_str.strip()
            dep_idx = int(dep_idx)
            if not dep_idx in dpg.nodes.keys():
                dep_node = DNode(dep_idx, dep_str)
                dpg.addNode(dep_node)
            dpg.addEdge(gov_idx,dep_idx)
        return dpg
                
    def is_empty(self):
        return len(self.nodes.keys()) == 0
    
    def numNodes(self):
        return len(self.nodes.keys())

    def nodes_list(self):
        return self.nodes.keys()

    def addNode(self,node):
        self.nodes[node.index] = node

    def addEdge(self,gov_index,dep_index):
        self.nodes[gov_index].addChildren(dep_index)
        self.nodes[dep_index].addParent(gov_index)
        
    def get_direction(self,i,j):
        if j in self.nodes[i].children:
            return 0
        elif i in self.nodes[j].children:
            return 1
        else:
            return -1
        
    # ignore the multiedge between same nodes, which does not exist in unlabled mode
    def remove_edge(self,gov_index,dep_index):
        self.nodes[gov_index].removeChild(dep_index)
        self.nodes[dep_index].removeParent(gov_index)

    def swap_head(self,gov_index,dep_index):
        """
        make dep head of gov and all current gov's dependents
        """
        self.nodes[dep_index].addChildren(gov_index)
        self.nodes[gov_index].children.remove(dep_index)
        children = self.nodes[gov_index].children
        self.nodes[dep_index].addChildren(children)
        self.nodes[gov_index].children = []
        for c in children:
            self.nodes[c].parents.remove(gov_index)
            self.nodes[c].parents.append(dep_index)

        parents = self.nodes[gov_index].parents
        self.nodes[dep_index].parents = parents
        self.nodes[gov_index].parents = [dep_index]
        for p in parents:
            self.nodes[p].children.remove(gov_index)
            self.nodes[p].children.append(dep_index)

    
    def bfs(self):
        from collections import deque
        visited_nodes = set()
        dep_tuples = []
        
        queue = deque([self.root])
        while queue:
            next = queue.popleft()
            if next in visited_nodes:
                continue
            visited_nodes.add(next)
            for child in self.nodes[next].children:
                if not (next,child) in dep_tuples:
                    if not child in visited_nodes:
                        queue.append(child)
                    dep_tuples.append((next,child))
        return dep_tuples

    def postorder(self,root,seq=[]):
        if self.nodes[root].children == []:
            seq.append(root)
        else:
            for child in self.nodes[root].children:
                self.postorder(child,seq)
            seq.append(root)
        return seq
            
        
    def print_tuples(self):
        """print the dependency graph as tuples"""
        return '\n'.join("(%s,%s)"%(self.nodes[g].str,self.nodes[d].str) for g,d in self.bfs())
    
            
            
