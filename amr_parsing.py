#!/usr/bin/python

import sys,codecs
from optparse import OptionParser
import re
import cPickle as pickle
from common.SpanGraph import *
from common.AMRGraph import *
import subprocess
from Aligner import *
from parser import *

def readAMR(amrfile_path):
    amrfile = codecs.open(amrfile_path,'r',encoding='utf-8')
    comment_list = []
    comment = {}
    amr_list = []
    amr_string = ''
    for line in amrfile.readlines():
        if line.startswith('#'):
            m = re.match("#\s::(\w+)\s(.+)",line)
            if m:
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


def get_dependency_graph(stp_dep,FROMFILE=False):
    if FROMFILE: 
        depfile = codecs.open(stp_dep,'r',encoding='utf-8')
        inputlines = depfile.readlines()
    else:
        inputlines = stp_dep.split('\n')

    dpg_list = []
    dep_lines = []
    i = 0
    
    for line in inputlines:
        if line.strip():
            dep_lines.append(line)
            #label = line.split('(')[0]
            #gov_node = DNode(line.split('(')[1].split(',')[0])
            #dep_node = DNode(line.strip().split('(')[1].split(',')[1][:-1])
            #dpg.add_edge(gov_node,dep_node,label)
        else:            
            dpg = SpanGraph.init_dep_graph(dep_lines)
            dep_lines = []
            dpg_list.append(dpg)

    if not dpg.is_empty():
        dpg_list.append(dpg)

    return dpg_list
                
    
def write_sentences(file_path,comments):
    """
    write out the sentences to file
    """
    output = codecs.open(file_path,'w',encoding='utf-8')
    for c in comments:
        sent = c['snt']
        output.write(sent+'\n')
    output.close()
    
    
def main():
    usage = "Usage:%prog [options] amr_file"
    opt = OptionParser(usage=usage)
    opt.add_option("-v",action="store",dest="verbose",type='int',
                   default=0,help="set up verbose level")
    opt.add_option("-a",action="store_true",dest="align",
                   default=False,help="do alignment between sentence and amr")
    opt.add_option("-b",action="store",dest="begin",type='int',
                   default=0,help="for debugging"
                   "When do alignment, where the alignment begins"
                   "When test oracle, where to begin")
    opt.add_option("-s",action="store",dest="start_step",type='int',
                   default=0,help="where the step begins,for testing oracle")
    opt.add_option("-o",action="store",dest="sentfilep",
                   help="output sentences to file and parse the sentence into dependency graph")
    opt.add_option("-i",action="store",dest="parsedfilep",
                   help="read parsed dependency graph from file")
    opt.add_option("-g",action="store",dest="userActfile",
                   help="read user input action sequences as guide")
    opt.add_option("-d",action="store",dest="oracle",type='int',default=0,\
                   help="test the output actions of deterministic oracle: "
                         "1: tree oracle 2: list-based oracle")

    (options,args) = opt.parse_args()
    if len(args) != 1:
        opt.print_help()
        opt.error("Incorrect number of arguments!")
    if options.sentfilep and options.parsedfilep:
        opt.error("Options -i and -o are mutually exclusive!")
    if not (options.sentfilep or options.parsedfilep) and not options.align:
        opt.error("Should at least choose one of the option -i or -o")
    
    log = sys.stderr

    amrfilep = args[0]
    comments,amr_strings = readAMR(amrfilep)
    
    if options.sentfilep:
        write_sentences(options.sentfilep,comments)
        process = subprocess.Popen(['./runDep.sh',options.sentfilep],shell=False,stdout=subprocess.PIPE)
        stp_dep = process.communicate()[0] # stdout
        #print stp_dep
        #stp_filep = sent_filep+'.stp'
        #dgs = get_dependency_graph(stp_dep)


    if options.parsedfilep:
        dgs = get_dependency_graph(options.parsedfilep,FROMFILE=True)
        #print dgs[5].postorder(dgs[5].root)
        sample = dgs[388]
        print sample.locInTree(4),sample.locInTree(9)
        print sample.relativePos(4,9)
        
    
    #print 'AMR strings:\n'+'\n'.join(amr_str for amr_str in amr_strings)
    snts = [com['snt'] for com in comments]
    amrs = [AMR.parse_string(amr_string) for amr_string in amr_strings]
        
    # test user guide actions
    if options.userActfile:
       user_actions = [int(n) for n in open(options.userActfile).readline().split()] 
       amr_parser = Parser()
       amr_parser.testUserGuide(user_actions,dgs[4],snts[4])

    # do alignment
    if options.align:
        if options.verbose > 0:
            print >> log, "Do alignment..."
        amr_aligner = Aligner(align_type=3,verbose=options.verbose)
        ref_graphs = []
        begin = options.begin 
        counter = 1
        for snt, amr in zip(snts[begin:],amrs[begin:]):
            if options.verbose > 1:
                print >> log, counter
                print >> log, "Sentences:"
                print >> log, snt+'\n'
                
                print >> log, "AMR:"                
                print >> log, amr.to_amr_string()

            alresult = amr_aligner.apply_align(snt,amr)
            ref_amr_graph = SpanGraph.init_ref_graph(amr,alresult)
            ref_graphs.append(ref_amr_graph)
            if options.verbose > 1:
                #print >> log, "Reference tuples:"
                #print >> log, ref_depGraph.print_tuples()
                print amr_aligner.print_align_result(alresult,amr)
                raw_input('ENTER to continue')
            counter += 1
        pickle.dump(ref_graphs,open('ref_graph.p','wb'))

    # test deterministic oracle 
    if options.oracle:
        oracle_type = options.oracle
        start_step = options.start_step
        begin = options.begin
        amr_parser = Parser(oracle_type,options.verbose)
        #amr_aligner = Aligner()
        #alignment = amr_aligner.apply_align(snts[4],amrs[4])
        ref_graphs = pickle.load(open('ref_graph1.p','rb'))

        for ref_graph,dep_graph,snt in zip(ref_graphs,dgs,snts)[begin:]:
            amr_parser.testOracleGuide('det_tree_oracle',ref_graph,dep_graph,snt,start_step)

        amr_parser.record_actions('data/action_set.txt')
        
    #stack = dgs[4].tree_reordering(dgs[4].root)
    #print '\n'.join(dgs[4].get_node(x).get_strform() for x in stack)
    
if __name__ == "__main__":
    main()

