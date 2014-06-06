# -*- coding:utf-8 -*-

"""
amr graph representaion util

@author: Chuan Wang
@since: 2013-11-20
"""

from collections import defaultdict
import re

def trim_concepts(line):
    """
    quote all the string literals
    """
    pattern = re.compile('(:name\s*\(n / name\s*:op\d)\s*\(([^:)]+)\)\)')
    def quote(match):
        return match.group(1)+' "'+ match.group(2) + '")' 
    return pattern.sub(quote,line)


to_19 = ( 'zero',  '(one|a|an)',   'two',  'three', 'four',   'five',   'six',
          'seven', 'eight', 'nine', 'ten',   'eleven', 'twelve', 'thirteen',
          'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen' )
tens  = ( 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
denom = ( '',
          'thousands?',     'millions?',         'billion?',       'trillion',       'quadrillion',
          'quintillion',  'sextillion',      'septillion',    'octillion',      'nonillion',
          'decillion',    'undecillion',     'duodecillion',  'tredecillion',   'quattuordecillion',
          'sexdecillion', 'septendecillion', 'octodecillion', 'novemdecillion', 'vigintillion' )

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

class Literal(str):
    def __str__(self):
        return "'%s" % "".join(self)

    def __repr__(self):
            return "".join(self)

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
    
    def __setitem__(self, k, v):
        if k in self:
            raise KeyError('Cannot assign to ListMap entry; use replace() or append()')
        else:
            self._keys.append(k)
        return defaultdict.__setitem__(self, k, v)
    
    def __getitem__(self, k):
        '''Returns the *first* list entry for the key.'''
        return dict.__getitem__(self, k)[0]

    def getall(self, k):
        return dict.__getitem__(self, k)
        
    def items(self):
        return [(k,v) for k in self._keys for v in self.getall(k)]
    
    def values(self):
        return [v for k,v in self.items()]
    
    def itemsfor(self, k):
        return [(k,v) for v in self.getall(k)]
    
    def replace(self, k, v):
        defaultdict.__setitem__(self, k, [v])
        
    def append(self, k, v):
        defaultdict.__getitem__(self, k).append(v) 
    
    def remove(self, k, v):
        defaultdict.__getitem__(self, k).remove(v)        
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
        return (t[0], ()) + t[2:]

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
    def __init__(self,alist=[]):
        deque.__init__(self,alist)
        
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
        
