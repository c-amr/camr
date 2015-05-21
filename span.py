
"""
sentence span mapping to special concepts in amr like name,date-entity,etc.
"""


class Span(object):
    
    def __init__(self,start,end,words,entity_tag):
        self.start = start
        self.end = end
        self.entity_tag = entity_tag
        self.words = words

    def set_entity_tag(self,entity_tag):
        self.entity_tag = entity_tag
    
    def __str__(self):
        return '%s: start: %s, end: %s , tag:%s'%(self.__class__.__name__,self.start,self.end,self.entity_tag)
    
    def __repr__(self):
        return '%s: start: %s, end: %s , tag:%s'%(self.__class__.__name__,self.start,self.end,self.entity_tag)

    def __eq__(self,other):
        return other.start == self.start and other.end == self.end
        
    def contains(self,other_span):
        if other_span.start >= self.start and other_span.end <= self.end and not (other_span.start == self.start and other_span.end == self.end):
            return True
        else:
            return False
