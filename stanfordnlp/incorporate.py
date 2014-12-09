import incorporate_util as iu
import re
from pprint import pprint
from util.py import find_abr_fullname, find_query_fullname
source_port = '8000'
source_tv_handle = 'tvrh'
dump_port = '8001'
knowledge_base_port = '8002'
kb_tv_handle = 'tvrh'

def get_full_form (query):
    ## First parse the source text to get the full form of the query
    return query['name'].replace(' ','_')

def isabbr(query_string):
    ## is abbreviation
    return query_string.isupper()

def get_all_related (query_string , source_doc ):
    ### Get all named entities that include a token of query_string
    ### For example, if the query is American Express
    ### Get all named entities such as: American News, Fun Express etc.
    if isabbr(query_string):
        allforms = find_abr_fullname(source_doc,query_string,2)
    else:
        allforms = find_query_fullname(source_doc,query_string)
    return allforms

def get_all_in_doc ( source_doc ):
### Get all named entities inside the document
    pass

def wiki_standard (string):
    return string.replace(' ','_')

def get_candidate (query):
    query_string = query['name']

    full_query_string = get_full_form (query)
    candidates = {}
    
    ## Connect to knowledge_base solr server to get back some candidates
    ## Three different kind of queries directly into knowledge base

    ## Exact match, it might work with full_query
    ## http://localhost:8002/solr/tvrh/?q=wiki_title:Mike_Quigley_(footballer)&fl=text&tv.fl=text&tv.tf_idf=true
    ## exac_link = ('http://localhost:8002/solr/' + kb_tv_handle +
    ## '/?q=wiki_title:' + full_query_string + '&fl=text&tv.fl=text&tv.tf_idf=true' )
##    exac_link = ('http://localhost:8002/solr/select/?q=wiki_title:' + wiki_standard(query_string) + '&fl=wiki_title' )
    exac_link = ('http://localhost:8002/solr/' + kb_tv_handle +
                 '/?q=wiki_title:' + wiki_standard(query_string) + '&fl=wiki_title&tv.fl=text&tv.tf_idf=true' )
    exact_parsed = iu.crawl_and_parse(exac_link)
    e_items = iu.parse_source_result ( exact_parsed )
    
    candidates['exact'] = []
    for item in e_items:
        candidates['exact'].append( {'title': item['title'], 'tv': item['term_vector']} )
    
    e_flag = False
    if len(e_items) > 0:
        e_flag = True

    ## Token match, tokenize the title of the knowledge base
    ## http://localhost:8002/solr/tvrh/?q=wiki_title_token:Quigley&fl=text&tv.fl=text&tv.tf_idf=true
##    token_link = ('http://localhost:8002/solr/' + kb_tv_handle +
##    '/?q=wiki_title_token:' + query + '&fl=text&tv.fl=text&tv.tf_idf=true' )
    if e_flag:
        rows = 5
    else:
        rows = 20
##    token_link = ('http://localhost:8002/solr/select/?q=wiki_title_token:' + wiki_standard(query_string) + '&fl=wiki_title&rows=' + str(rows) )
    token_link = ('http://localhost:8002/solr/' + kb_tv_handle +
    '/?q=wiki_title_token:' + wiki_standard(query_string) +
    '&fl=wiki_title&tv.fl=text&tv.tf_idf=true&rows=' + str(rows) )
    
    token_parsed = iu.crawl_and_parse(token_link)
    t_items = iu.parse_source_result ( token_parsed )
    candidates['token'] = []
    for item in t_items:
        candidates['token'].append( {'title': item['title'], 'tv': item['term_vector']} )
    
    ## For example, query is Quigley, and one document has name Mike_Quigley_(footballer)
    ## Ngram match
    ## http://localhost:8002/solr/tvrh/?q=wiki_title_ngram:Quig&fl=text&tv.fl=text&tv.tf_idf=true
    ## ngram_link = ('http://localhost:8002/solr/' + kb_tv_handle +
    ## '/?q=wiki_title_ngram:' + query + '&fl=text&tv.fl=text&tv.tf_idf=true' )
##    ngram_link = ('http://localhost:8002/solr/select/?q=wiki_title_ngram:' + wiki_standard(query_string) + '&fl=wiki_title&rows=' + str(rows) )
    ngram_link = ('http://localhost:8002/solr/' + kb_tv_handle +
    '/?q=wiki_title_ngram:' + wiki_standard(query_string) +
    '&fl=wiki_title&tv.fl=text&tv.tf_idf=true&rows=' + str(rows) )
    ngram_parsed = iu.crawl_and_parse(ngram_link)
    n_items = iu.parse_source_result ( ngram_parsed )
    
    candidates['ngram'] = []
    for item in n_items:
        candidates['ngram'].append( {'title': item['title'], 'tv': item['term_vector']} )

    ## We should also check through the redirect page and disambiguation page
    
    ## redirect_link = ('http://localhost:8001/solr/select/?q=title:' + full_query_string + '&fl=wiki_title' )

    disambiguation_link = ('http://localhost:8003/solr/select/?q=title:' + wiki_standard(query_string) )
    disam_parsed = iu.crawl_and_parse(disambiguation_link)
    if disam_parsed.find(attrs={"name": "title"}) != None:
        candidates['disam'] = []
        cds = disam_parsed.find_all(attrs={"name":re.compile("candidate_")})
        aux_texts = disam_parsed.find_all(attrs={"name":re.compile("aux_text_")})
        
        candidates_dict = {}
        aux_dict = {}
        for cd in cds:
            candidates_dict[cd['name'][10:]] = cd.text
        for aux in aux_texts:
            aux_dict[aux['name'][9:]] = aux.text

        for t in candidates_dict:
            dis_exac_link = ('http://localhost:8002/solr/' + kb_tv_handle +
                 '/?q=wiki_title:' + wiki_standard(candidates_dict[t]) + '&fl=wiki_title&tv.fl=text&tv.tf_idf=true' )
            
            dis_exac_parsed = iu.crawl_and_parse(dis_exac_link)
            dis_e_items = iu.parse_source_result ( dis_exac_parsed )
            
            if len(dis_e_items) > 0:
                print dis_exac_link
                print wiki_standard(candidates_dict[t])
            for item in dis_e_items:
                pprint(item['title'])
                candidates['disam'].append({'title': item['title'], 'tv': item['term_vector'], 'support' : aux_dict[t]} )

    ## that could be a redirect page
    dump_link = ('http://localhost:8001/solr/select/?q=title:' + full_query_string )
    dump_parsed = iu.crawl_and_parse(dump_link)
    if dump_parsed.find(attrs={"name": "text"}) != None:
        dump_text = dump_parsed.find(attrs={"name": "text"}).text
        if dump_text[:10] == '#REDIRECT ':
            match = re.search ('\[\[(?P<redirect>.+)\]\]', dump_text)
            candidate = wiki_standard(match.group('redirect'))

            rdr_exac_link = ('http://localhost:8002/solr/' + kb_tv_handle +
                 '/?q=wiki_title:' + candidate + '&fl=wiki_title&tv.fl=text&tv.tf_idf=true' )
            rdr_exac_parsed = iu.crawl_and_parse( rdr_exac_link )
            rdr_e_items = iu.parse_source_result ( rdr_exac_parsed )

            candidates['redirect'] = []
            for item in rdr_e_items:
                candidates['redirect'].append({'title': item['title'], 'tv': item['term_vector']} )
            
        
    return candidates
##    ## Using disambiguation and dump solr server to provide more candidates
    
##query = {'node': u'E0361097', 'doc': u'eng-NG-31-127442-12066646', 'type': u'ORG', 'name': u'UN Security Council'}
##get_candidate (query)
##query = {'doc':'eng-NG-31-128712-9340034'}
##get_source_data(query)
