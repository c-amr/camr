from common.util import ConstTag,ETag


#print os.path.relpath(__file__)
#print os.path.exists('data.py')

#print open('data.py','r').read()
#print open('data.py','r').readlines()

def delta_func(tag_to_predict,tok_form):
    if isinstance(tag_to_predict,(ConstTag,ETag)):
        return 'ECTag'
    else:
        tok_form = tok_form.lower()
        tag_lemma = tag_to_predict.split('-')[0]
        if tag_lemma == tok_form:
            return '1'
        i=0
        slength = len(tag_lemma) if len(tag_lemma) < len(tok_form) else len(tok_form)
        while i < slength and tag_lemma[i] == tok_form[i]:
            i += 1
        if i == 0:
            return '0'
        elif tok_form[i:]:
            return tok_form[i:]
        elif tag_lemma[i:]:
            return tag_lemma[i:]
        else:
            assert False