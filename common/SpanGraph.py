#!/usr/bin/python


# Implementation for graph of which nodes are spans of sentence 
# author Chuan Wang

import copy
import sys,re
from util import StrLiteral,Polarity,Quantity

class SpanNode(object):
    def __init__(self,start,end,words,entity_tag=None):
        self.start = start
        self.end = end
        self.entity_tag = entity_tag
        self.words = words
        self.children = []
        self.parents = [] 
    @staticmethod
    def from_span(span):
        """initialize from span object"""
        return SpanNode(span.start,span.end,span.words,span.entity_tag)

    def addChildren(self,c):
        if isinstance(c,list):
            self.children.extend(c)
        else:
            self.children.append(c)

    def contains(self,other_node):
        if other_node.start >= self.start and other_node.end <= self.end:
            return True
        else:
            return False

    def addParent(self,parent):
        self.parents.append(parent)

    def removeChild(self,child):
        self.children.remove(child)
        
    def removeParent(self,parent):
        self.parents.remove(parent)

    def __str__(self):
        return 'Span node:(%s,%s) Children: %s'%(self.start,self.end,self.children)
    def __repr__(self):
        return 'Span node:(%s,%s) Children: %s'%(self.start,self.end,self.children)

class SpanGraph(object):
    """
    Graph of span nodes
    """
    LABELED = False

    def __init__(self):
        self.root = 0
        self.nodes = {} # refer to spans by start index

    @staticmethod
    def init_ref_graph(amr,alignment):
        """Instantiate graph from AMR graph and alignment"""
        spgraph = SpanGraph()
        span_alignment = copy.deepcopy(alignment)
        #DepGraph.LABELED = LABELED
        vars = [node.node_label for node in amr.dfs()[0] if not isinstance(node.node_label,StrLiteral)]

        for h in vars:
            hconcept = amr.node_to_concepts[h] if h in amr.node_to_concepts else h
            if not h in span_alignment:
                continue
            hspans = span_alignment[h]
            if len(hspans) < 2:
                hspan = hspans[0]
            else:
                hspan = hspans.pop(0)
            if not hspan.start in spgraph.nodes:
                h_node = SpanNode.from_span(hspan)
                spgraph.addNode(h_node)
            for ds in amr[h].values():
                d = ds[0]
                dconcept = amr.node_to_concepts[d] if d in amr.node_to_concepts else d
                if not d in span_alignment:
                    continue
                dspans = span_alignment[d]
                dspan = dspans[0]
                if hspan.contains(dspan):
                    continue
                if not dspan.start in spgraph.nodes:
                    d_node = SpanNode.from_span(dspan)
                    spgraph.addNode(d_node)
                spgraph.addEdge(hspan.start,dspan.start)
        root = SpanNode(0,1,['root'],'O')
        spgraph.addNode(root)
        if amr.roots[0] in span_alignment:
            spgraph.addEdge(0,span_alignment[amr.roots[0]][0].start)
        else:
            print >> sys.stderr, "WARNING:root not aligned!"
        #spgraph.clean_replicate_nodes()
        return spgraph

    @staticmethod
    def init_dep_graph(stp_deps):
        """instantiate graph from dependency tuples"""
        dpg = SpanGraph()
        for line in stp_deps:
            line = line.strip()
            label = line.split('(')[0]
            
            gov_str,gov_idx = line.split('(')[1].split(',',1)[0].rsplit('-',1)
            gov_str = gov_str.strip()
            gov_idx = int(gov_idx)
            if not gov_idx in dpg.nodes:
                gov_node = SpanNode(gov_idx,gov_idx+1,[gov_str])
                dpg.addNode(gov_node)
            
            dep_str,dep_idx = line.split('(')[1].split(',',1)[1][:-1].rsplit('-',1)
            dep_str = dep_str.strip()
            dep_idx = int(dep_idx)
            if not dep_idx in dpg.nodes:
                dep_node = SpanNode(dep_idx,dep_idx+1,[dep_str])
                dpg.addNode(dep_node)
            dpg.addEdge(gov_idx,dep_idx)

        if not dpg.nodes:
            root = SpanNode(0,1,['root'],'O')
            dpg.addNode(root)
        elif 0 not in dpg.nodes:
            dpg.make_root()
        return dpg
    
    def make_root(self):
        first = sorted(self.nodes.keys())[0]
        root = SpanNode(0,1,['root'],'O')
        self.addNode(root)
        for c in self.nodes[first].children:
            self.remove_edge(first,c)
            self.addEdge(0,c)

    def is_empty(self):
        return len(self.nodes.keys()) == 0
    
    def numNodes(self):
        return len(self.nodes.keys())

    def nodes_list(self):
        return self.nodes.keys()

    def addNode(self,node):
        self.nodes[node.start] = node

    def addEdge(self,gov_index,dep_index):
        self.nodes[gov_index].addChildren(dep_index)
        self.nodes[dep_index].addParent(gov_index)
        
    def get_direction(self,i,j):
        """left or right or no arc"""
        if j in self.nodes[i].children:
            return 0
        elif i in self.nodes[j].children:
            return 1
        else:
            return -1
    def remove_node(self,idx):
        for p in self.nodes[idx].parents:
            self.remove_edge(p,idx)
        for c in self.nodes[idx].children:
            self.remove_edge(idx,c)
        
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

    def swap_head2(self,gov_index,dep_index):
        """
        keep dep and gov's dependents unchanged, only switch the dependency edge 
        direction 
        """
        # 
        for p in self.nodes[gov_index].parents:
            self.remove_edge(p,gov_index)
            self.addEdge(p,dep_index)

        self.nodes[gov_index].removeChild(dep_index)
        self.nodes[gov_index].addParent(dep_index)

        self.nodes[dep_index].removeParent(gov_index)
        self.nodes[dep_index].addChildren(gov_index)

    def merge_node(self,idx1,idx2):
        lr = self.get_direction(idx1,idx2)
        if lr == 0:
            self.remove_edge(idx1,idx2)
        elif lr == 1:
            self.remove_edge(idx2,idx1)
        else:
            print >> sys.stderr, "WARNING: no edge between merged nodes!"
        
        self.nodes[idx1].end = self.nodes[idx2].end
        self.nodes[idx1].words.extend(self.nodes[idx2].words)
        self.nodes[idx1].parents.extend(self.nodes[idx2].parents)
        for p in self.nodes[idx2].parents:
            self.nodes[p].children.remove(idx2)
            self.nodes[p].children.append(idx1)
        self.nodes[idx1].children.extend(self.nodes[idx2].children)
        for c in self.nodes[idx2].children:
            self.nodes[c].parents.remove(idx2)
            self.nodes[c].parents.append(idx1)
        
        
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

    def postorder(self,root=0,seq=None):
        if seq is None:
            seq = []
        if self.nodes[root].children == []:
            #seq.append(root)
            pass
        else:
            for child in self.nodes[root].children:
                self.postorder(child,seq)
            seq.append(root)
        return seq
    
    def leaves(self):
        """return all the leaves ordered by their indexes in the sentence"""
        leaves = []
        for nidx in self.nodes:
            if self.nodes[nidx].children == []:
                leaves.append(nidx)
        return sorted(leaves)
    
    def locInTree(self,idx):        
        depth = 0 
        candidates = self.leaves()
        while idx not in candidates:
            candidates = sorted(list(set([self.nodes[l].parents[0] for l in candidates if self.nodes[l].parents])))
            depth +=1 
        assert idx in candidates
        return (candidates.index(idx),depth)
        
    def path(self,idx):
        path = []
        cur = self.nodes[idx]
        while cur.parents:            
            path.insert(0,self.nodes[cur.parents[0]].children.index(cur.start))
            cur = self.nodes[cur.parents[0]]
        path.insert(0,0)
        return path
        
    def relativePos(self,currentIdx,otherIdx):
        cindex,cdepth = self.locInTree(currentIdx)
        oindex,odepth = self.locInTree(otherIdx)
        return (cindex-oindex,cdepth-odepth)  
    
    def relativePos2(self,currentIdx,otherIdx):
        cpath = self.path(currentIdx)
        opath = self.path(otherIdx)
        
        if len(cpath) > 1 and len(opath) > 1:
            if cpath[-2] == opath[-2]: # same parent
                return 'SP'

        if len(cpath) > 2 and len(opath) > 2:
            if cpath[-3] == opath[-3] and cpath[-2] != opath[-2]: # same grand parent
                return 'SGP'
            if cpath[-3] == opath[-2]: # current Parent's brother
                return 'PB'
            if opath[-3] == cpath[-2]:
                return 'rPB'
        if len(cpath) > 0 and len(opath) > 0:
            if opath[-1] in cpath: # otherIdx on currentIdx's path
                #return 'P'+str(len(cpath)-1-cpath.index(opath[-1]))
                return 'P'
            if cpath[-1] in opath: # currentIdx on otherIdx's path
                #return 'rP'+str(len(opath)-1-opath.index(cpath[-1]))
                return 'rP'

        return 'O'
                
            
    def print_tuples(self):
        """print the dependency graph as tuples"""
        return '\n'.join("((%s),(%s))"%(','.join(w for w in self.nodes[g].words),','.join(t for t in self.nodes[d].words)) for g,d in self.bfs())
    
            
            

