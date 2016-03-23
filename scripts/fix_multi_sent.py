
'''
fix charniak parse file with multiple sentences 
'''
from nltk.tree import Tree
import sys
import re
import codecs

def fix_multi_sent(line):
    tree = Tree.fromstring(line)
    if len(tree) > 1:
        newtree = Tree('S1',[Tree('S',tree[:])])
    else:
        newtree = tree
    return re.sub('\n\s*',' ',newtree.__str__())


if __name__ == '__main__':
    old_parse_file = sys.argv[1]
    new_parse_file = old_parse_file.rsplit('.',1)[0]
    print '%s >> %s' % (old_parse_file, new_parse_file)
    result = []
    with codecs.open(old_parse_file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            new = fix_multi_sent(line)
            result.append(new)

    with codecs.open(new_parse_file, 'w', encoding='utf-8') as wf:
        wf.write('\n'.join(result))
        wf.write('\n')
    

