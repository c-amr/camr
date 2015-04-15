#!/usr/bin/python
import sys,argparse
from stanfordnlp.corenlp import *
from common.AMRGraph import *
from pprint import pprint
import cPickle as pickle
from Aligner import Aligner
from common.SpanGraph import SpanGraph

log = sys.stderr

def readAMR(amrfile_path):
    amrfile = codecs.open(amrfile_path,'r',encoding='utf-8')
    comment_list = []
    comment = {}
    amr_list = []
    amr_string = ''
    for line in amrfile.readlines():
        if line.startswith('#'):
            for m in re.finditer("::([^:\s]+)\s(((?!::).)*)",line):
                #print m.group(1),m.group(2)
                comment[m.group(1)] = m.group(2)
        elif not line.strip():
            if amr_string and comment:
                comment_list.append(comment)
                amr_list.append(amr_string)
                amr_string = ''
                comment = {}
        else:
            amr_string += line.strip()+' '
    if amr_string and comment:
        comment_list.append(comment)
        amr_list.append(amr_string)

    return (comment_list,amr_list)

def _write_sentences(file_path,sentences):
    """
    write out the sentences to file
    """
    output = codecs.open(file_path,'w',encoding='utf-8')
    for sent in sentences:
        output.write(sent+'\n')
    output.close()


def _write_tok_amr(file_path,amr_file,instances):
    output_tok = open(file_path,'w')
    origin_comment_string = ''
    origin_amr_string = ''
    comment_list = []
    amr_list = []
    for line in open(amr_file,'r').readlines():
        if line.startswith('#'):
            origin_comment_string += line 
        elif not line.strip():
            if origin_amr_string and origin_comment_string:
                comment_list.append(origin_comment_string)
                amr_list.append(origin_amr_string)

                origin_amr_string = ''
                origin_comment_string = ''
        else:
            origin_amr_string += line
    if origin_amr_string and origin_comment_string:
        comment_list.append(origin_comment_string)
        amr_list.append(origin_amr_string)

    for i in xrange(len(instances)):
        output_tok.write(comment_list[i])
        output_tok.write("# ::tok %s\n" % (' '.join(instances[i].get_tokenized_sent())))
        output_tok.write(amr_list[i])
        output_tok.write('\n')

    output_tok.close()

def _add_amr(instances,amr_strings):
    assert len(instances) == len(amr_strings)
    
    for i in range(len(instances)):
        instances[i].addAMR(AMR.parse_string(amr_strings[i]))

def preprocess(amr_file,START_SNLP=True):
    '''nasty function'''
    aligned_amr_file = amr_file + '.aligned'
    if os.path.exists(aligned_amr_file):
        comments,amr_strings = readAMR(aligned_amr_file)
    else:
        comments,amr_strings = readAMR(amr_file)
    comments,amr_strings = readAMR(aligned_amr_file)
    sentences = [c['tok'] for c in comments]
    tmp_sentence_file = amr_file+'.sent'
    _write_sentences(tmp_sentence_file,sentences)

    print >> log, "pos, ner and dependency..."
    proc = StanfordCoreNLP()
    if START_SNLP: proc.setup()
    instances = proc.parse(tmp_sentence_file)

    tok_amr_filename = amr_file + '.tok'
    if not os.path.exists(tok_amr_filename):
        _write_tok_amr(tok_amr_filename,amr_file,instances)
    
    SpanGraph.graphID = 0
    for i in range(len(instances)):

        amr = AMR.parse_string(amr_strings[i])
        
        alignment = Aligner.readJAMRAlignment(amr,comments[i]['alignments'])
        ggraph = SpanGraph.init_ref_graph(amr,alignment,instances[i].tokens)
        #ggraph.pre_merge_netag(instances[i])
        #print >> log, "Graph ID:%s\n%s\n"%(ggraph.graphID,ggraph.print_tuples())
        instances[i].addAMR(amr)
        instances[i].addGoldGraph(ggraph)
        
    #print >> log, "adding amr"
    #_add_amr(instances,amr_strings)
    #if writeToFile:
    #    output_file = amr_file.rsplit('.',1)[0]+'_dataInst.p'
    #    pickle.dump(instances,open(output_file,'wb'),pickle.HIGHEST_PROTOCOL)
        
    return instances

def _init_instances(sent_file,amr_strings,comments):
    print >> log, "Preprocess 1:pos, ner and dependency using stanford parser..."
    proc = StanfordCoreNLP()
    instances = proc.parse(sent_file)
    
    
    print >> log, "Preprocess 2:adding amr and generating gold graph"
    assert len(instances) == len(amr_strings)
    for i in range(len(instances)):
        amr = AMR.parse_string(amr_strings[i])
        instances[i].addAMR(amr)
        alignment = Aligner.readJAMRAlignment(amr,comments[i]['alignments'])
        ggraph = SpanGraph.init_ref_graph(amr,alignment,comments[i]['snt'])
        ggraph.pre_merge_netag(instances[i])
        instances[i].addGoldGraph(ggraph)

    return instances


def add_JAMR_align(instances,aligned_amr_file):
    comments,amr_strings = readAMR(aligned_amr_file)
    for i in range(len(instances)):
        amr = AMR.parse_string(amr_strings[i])
        alignment = Aligner.readJAMRAlignment(amr,comments[i]['alignments'])
        ggraph = SpanGraph.init_ref_graph(amr,alignment,instances[i].tokens)
        ggraph.pre_merge_netag(instances[i])
        #print >> log, "Graph ID:%s\n%s\n"%(ggraph.graphID,ggraph.print_tuples())
        instances[i].addAMR(amr)
        instances[i].addGoldGraph(ggraph)

    #output_file = aligned_amr_file.rsplit('.',1)[0]+'_dataInst.p'
    #pickle.dump(instances,open(output_file,'wb'),pickle.HIGHEST_PROTOCOL)

def preprocess_aligned(aligned_amr_file,writeToFile=True):
    comments,amr_strings = readAMR(aligned_amr_file)
    sentences = [c['tok'] for c in comments]
    tmp_sentence_file = aligned_amr_file.rsplit('.',1)[0]+'_sent.txt'
    _write_sentences(tmp_sentence_file,sentences)
    
    instances = _init_instances(tmp_sentence_file,amr_strings,comments)
    if writeToFile:
        output_file = aligned_amr_file.rsplit('.',1)[0]+'_dataInst.p'
        pickle.dump(instances,open(output_file,'wb'),pickle.HIGHEST_PROTOCOL)
        
    return instances


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="preprocessing for training/testing data")
    arg_parser.add_argument('-v','--verbose',action='store_true',default=False)
    #arg_parser.add_argument('-m','--mode',choices=['train','parse'])
    arg_parser.add_argument('-w','--writeToFile',action='store_true',help='write preprocessed sentences to file')
    arg_parser.add_argument('amr_file',help='amr bank file')
    
    args = arg_parser.parse_args()    

    instances = preprocess(args.amr_file)
    pprint(instances[1].toJSON())
    
