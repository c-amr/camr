# -*- coding:utf-8 -*-

"""
amr graph representaion and parsing util

@author: Chuan Wang
@since: 2013-11-20
"""

from collections import defaultdict
import re,string
import numpy as np
from constants import START_ID

# feature window placeholder

__AUX_PUNCTUATIONS = ["''"]

prep_list = ["aboard" ,"about" ,"above" ,"across" ,"after" ,"against" ,"along" ,"amid" ,"among" ,"anti" ,"around" ,"as" ,"at" ,"before" ,"behind" ,"below" ,"beneath" ,"beside" ,"besides" ,"beyond" ,"by" ,"concerning" ,"considering" ,"despite" ,"down" ,"during" ,"except" ,"excepting" ,"excluding" ,"following" ,"for" ,"from" ,"in" ,"inside" ,"into" ,"like" ,"minus" ,"near" ,"of" ,"off" ,"on" ,"onto" ,"opposite" ,"outside" ,"over" ,"past" ,"per" ,"plus" ,"regarding" ,"round" ,"save" ,"since" ,"than" ,"through" ,"to" ,"toward" ,"towards" ,"under" ,"underneath" ,"unlike" ,"until" ,"up" ,"upon" ,"versus" ,"via" ,"with" ,"within" ,"without"]

def ispunctuation(s):
    return s in string.punctuation or s in __AUX_PUNCTUATIONS

to_19 = ( 'zero',  '(one|a|an)',   'two',  'three', 'four',   'five',   'six',
          'seven', 'eight', 'nine', 'ten',   'eleven', 'twelve', 'thirteen',
          'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen' )
tens  = ( 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
denom = ( '',
          'thousands?',     'millions?',         'billion?',       'trillion',       'quadrillion',
          'quintillion',  'sextillion',      'septillion',    'octillion',      'nonillion',
          'decillion',    'undecillion',     'duodecillion',  'tredecillion',   'quattuordecillion',
          'sexdecillion', 'septendecillion', 'octodecillion', 'novemdecillion', 'vigintillion' )

def uniqify(seq):
    seen = {}
    result = []
    for item in seq:
        if item in seen: continue
        seen[item] = 1
        result.append(item)

    return result
    
def trim_concepts(line):
    """
    quote all the string literals
    """
    pattern = re.compile('(:name\s*\(n / name\s*:op\d)\s*\(([^:)]+)\)\)')
    def quote(match):
        return match.group(1)+' "'+ match.group(2) + '")' 
    return pattern.sub(quote,line)


# convert a value < 100 to English.
def _convert_nn(val):
    if val < 20:
        return to_19[val]
    for (dcap, dval) in ((k, 20 + (10 * v)) for (v, k) in enumerate(tens)):
        if dval + 10 > val:
            if val % 10:
                return dcap + '\s?-\s?' + to_19[val % 10]
            return dcap

# convert a value < 1000 to english, special cased because it is the level that kicks 
# off the < 100 special case.  The rest are more general.  This also allows you to
# get strings in the form of 'forty-five hundred' if called directly.
def _convert_nnn(val):
    word = ''
    (mod, rem) = (val % 100, val // 100)
    if rem > 0:
        word = to_19[rem] + '(\s?-\s?|\s)hundred'
        if mod > 0:
            word = word + '(\s?-\s?|\s)'
    if mod > 0:
        word = word +'(and)?(\s?-\s?|\s)?'+ _convert_nn(mod)
    return word

def english_number(val):
    if val < 100:
        return _convert_nn(val)
    if val < 1000:
        return _convert_nnn(val)
    for (didx, dval) in ((v - 1, 1000 ** v) for v in range(len(denom))):
        if dval > val:
            mod = 1000 ** didx
            l = val // mod
            r = val - (l * mod)
            ret = _convert_nnn(l) + '(\s?-\s?|\s)' + denom[didx]
            if r > 0:
                ret = ret + ' , ' + english_number(r)
            return ret

def to_order(val):
    order_dict = {'1':'first|once|every',
                  '2':'second|twice',
                  '3':'third',
                  '5':'fifth',
                  '8':'eighth',
                  '12':'twelfth'
                  }
    if val in order_dict:
        return order_dict[val]
    elif int(val) < 20:
        return to_19[int(val)] + 'th'
    else:
        return 'a^'
def format_num(val):
    length = len(val)
    result = val
    if length > 3:
        result = result[:length-3]+','+result[length-3:]
    if length > 6:
        result = result[:length-6]+','+result[length-6:]
    if length > 9:
        result = result[:length-9]+','+result[length-9:]
    return result

time_table = {'12:00':'noon'}
def to_time(time):
    if time in time_table:
        return time_table[time]
    else:
        hour = int(time.split(':')[0])
        return english_number(hour)
    
def to_round(val):
    if val < 100:
        return 'a^'
    if val < 1000:
        return 'hundreds'
    for (didx, dval) in ((v - 1, 1000 ** v) for v in range(len(denom))):
        if dval > val:
            return denom[didx]

class StrLiteral(unicode):
    def __str__(self):
        return '"%s"' % "".join(self)

    def __repr__(self):
            return "".join(self)

class SpecialValue(str):
    pass

class Quantity(str):
    pass

class Polarity(str):
    pass

class Interrogative(str):
    pass

class Literal(str):
    def __str__(self):
        return "'%s" % "".join(self)

    def __repr__(self):
            return "".join(self)

# entity class wrap around concept, distinguish between normal concept and abstract concept
class ETag(str):
    pass
# constant variable like quantity 
class ConstTag(str):
    pass
    
class ListMap(defaultdict):
    '''
    Here we use Nathan Schneider (nschneid)'s nice ListMap implementation
    for bolinas.

    A  map that can contain several values for the same key.
    @author: Nathan Schneider (nschneid)
    @since: 2012-06-18

    >>> x = ListMap()
    >>> x.append('mykey', 3)
    >>> x.append('key2', 'val')
    >>> x.append('mykey', 8)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [3, 8]})
    >>> x['mykey']
    3
    >>> x.getall('mykey')
    [3, 8]
    >>> x.items()
    [('key2', 'val'), ('mykey', 3), ('mykey', 8)]
    >>> x.itemsfor('mykey')
    [('mykey', 3), ('mykey', 8)]
    >>> x.replace('mykey', 0)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [0]})
    '''
    
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, list, *args, **kwargs)
        self._keys = []
        self._key_value = []
    
    def __setitem__(self, k, v):
        if k in self:
            raise KeyError('Cannot assign to ListMap entry; use replace() or append()')
        else:
            self._keys.append(k)
            self._key_value.extend([(k,vv) for vv in v])
        return defaultdict.__setitem__(self, k, v)
    
    def __getitem__(self, k):
        '''Returns the *first* list entry for the key.'''
        return dict.__getitem__(self, k)[0]

    def getall(self, k):
        return dict.__getitem__(self, k)
        
    def items(self):
        #return [(k,v) for k in self._keys for v in self.getall(k)]
        return [(k,v) for k,v in self._key_value]
    
    def values(self):
        return [v for k,v in self.items()]
    
    def itemsfor(self, k):
        return [(k,v) for v in self.getall(k)]
    
    def replace(self, k, v):
        defaultdict.__setitem__(self, k, [v])
        for i,(m,n) in enumerate(self._key_value):
            if m == k:
                self._key_value[i] = (k,v)
        
    def append(self, k, v):
        defaultdict.__getitem__(self, k).append(v)
        self._key_value.append((k,v))
    
    def remove(self, k, v):
        defaultdict.__getitem__(self, k).remove(v)
        self._key_value.remove((k,v))
        if not dict.__getitem__(self,k):
            del self[k]
            self._keys.remove(k)

    def removeall(self,v):
        i = 0
        while i < len(self._keys):
            k = self._keys[i]
            if v in self.getall(k):
                self.remove(k,v)
                self._keys.remove(k)
            i+=1


    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) + t[3:]

from collections import deque

class Stack(deque):
    def __init__(self,alist=[]):
        deque.__init__(self,alist)
    
    def top(self):
        return self[-1]

    def push(self,v):
        self.append(v)
    
    def isEmpty(self):
        return len(self) == 0

    def __reduce__(self):
        t = deque.__reduce__(self)
        return (t[0],(t[1][0],)) + t[2:]

class Buffer(deque):
    '''TODO: actually a stack need change name'''
    def __init__(self,alist=[]):
        deque.__init__(self,alist)
        #self.appendleft(START_ID)
        
    def top(self):
        return self[0]
    
    def push(self,v):
        self.appendleft(v)
    
    def pop(self):
        return self.popleft()
    
    def isEmpty(self):
        return len(self) == 0
    
    def __reduce__(self):
        t = deque.__reduce__(self)
        return (t[0],(t[1][0],)) + t[2:]
        
class Alphabet(object):
    """Two way map for label/feature and label/feature index

    It is an essentially a code book for labels or features
    This class makes it convenient for us to use numpy.array
    instead of dictionary because it allows us to use index instead of
    label string. The implemention of classifiers uses label index space
    instead of label string space.
    """
    def __init__(self):
        self._index_to_label = {}
        self._label_to_index = {}
        self.num_labels = 0

    def indexes(self):
        return self._index_to_label.keys()
        
    def labels(self):
        return self._label_to_index.keys()
        
    def size(self):
        return self.num_labels
    
    def has_label(self, label):
        return label in self._label_to_index
    
    def get_label(self, index):
        """Get label from index"""
        if index >= self.num_labels:
            raise KeyError("There are %d labels but the index is %d" % (self.num_labels, index))
        return self._index_to_label[index]

    def get_index(self, label):
        """Get index from label"""
        return self._label_to_index[label] if label in self._label_to_index else None
        
    def get_default_index(self,label):
        """get index for label, if label is not in the alphabet, we add it"""
        if label in self._label_to_index:
            return self._label_to_index[label]
        else:
            self.add(label)
            return self._label_to_index[label]
    
    def add(self,label):
        """Add an index for the label if it's a new label"""
        if label not in self._label_to_index:
            self._label_to_index[label] = self.num_labels
            self._index_to_label[self.num_labels] = label
            self.num_labels += 1

    def json_dumps(self):
        return json.dumps(self.to_dict())

    @classmethod
    def json_loads(cls, json_string):
        json_dict = json.loads(json_string)
        return Alphabet.from_dict(json_dict)

    def to_dict(self,index_to_label=False):
        if not index_to_label:
            new_table = dict([(str(key),value) for key,value in self._label_to_index.items()])
        else:
            new_table = dict([(key,str(value)) for key,value in self._index_to_label.items()])
        return new_table


    @classmethod
    def from_dict(cls, dictionary, index_to_label=False):
        """
        Create an Alphabet from dictionary
        """
        alphabet = cls()
        if not index_to_label:
            alphabet._label_to_index = dictionary
            #dict([(eval(key),value) if key[0] =='(' else (key,value) for key,value in alphabet_dictionary['_label_to_index'].items()])
            alphabet._index_to_label = {}
            for label, index in alphabet._label_to_index.items():
                alphabet._index_to_label[index] = label
        else:
            alphabet._index_to_label = dictionary
            alphabet._label_to_index = {}
            for index, label in alphabet._index_to_label.items():
                alphabet._label_to_index[label] = index
        # making sure that the dimension agrees
        assert(len(alphabet._index_to_label) == len(alphabet._label_to_index))
        alphabet.num_labels = len(alphabet._index_to_label)
        return alphabet

    def __len__(self):
        return self.size()
    
    def __eq__(self, other):
        return self._index_to_label == other._index_to_label and \
            self._label_to_index == other._label_to_index and \
            self.num_labels == other.num_labels
