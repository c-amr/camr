#!/usr/bin/python

"""
Interface for the parser:
parse command line 
read in corpus
"""

import sys,codecs,time,string
#from optparse import OptionParser
import re
import random
import cPickle as pickle
from common.SpanGraph import *
from common.AMRGraph import *
import subprocess
from Aligner import *
from parser import *
from model import Model
import argparse
from preprocessing import *
import constants
from graphstate import GraphState

#import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding('utf-8')

log = sys.stderr
LOGGED= False
#experiment_log = open('log/experiment.log','a')
experiment_log = sys.stdout


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

def write_parsed_amr(parsed_amr,instances,amr_file,suffix='parsed'):
    output = open(amr_file+'.'+suffix,'w')
    for pamr,inst in zip(parsed_amr,instances):
        output.write('# ::id %s\n'%(inst.sentID))
        output.write('# ::snt %s\n'%(inst.text))
        try:
            output.write(pamr.to_amr_string())
        except TypeError:
            import pdb
            pdb.set_trace()
        output.write('\n\n')
    output.close()
    
def main():
    '''
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
    '''
    arg_parser = argparse.ArgumentParser(description="Brandeis transition-based AMR parser 1.0")
    
    arg_parser.add_argument('-v','--verbose',type=int,default=0,help='set up verbose level for debug')
    arg_parser.add_argument('-b','--begin',type=int,default=0,help='specify which sentence to begin the alignment or oracle testing for debug')
    arg_parser.add_argument('-s','--start_step',type=int,default=0,help='specify which step to begin oracle testing for debug')
    #arg_parser.add_argument('-i','--input_file',help='the input: preprocessed data instances file for aligner or training')
    arg_parser.add_argument('-d','--dev',help='development file')
    arg_parser.add_argument('-as','--actionset',choices=['basic'],default='basic',help='choose different action set')
    arg_parser.add_argument('-m','--mode',choices=['preprocess','test_gold_graph','align','userGuide','oracleGuide','train','parse'],help="preprocess:generate pos tag, dependency tree, ner\n" "align:do alignment between AMR graph and sentence")
    arg_parser.add_argument('-dp','--depparser',choices=['stanford','turbo','mate','malt','stdconv+charniak'],default='stanford',help='choose the dependency parser, default:{stanford}')
    arg_parser.add_argument('--model',help='specify the model file')
    arg_parser.add_argument('--feat',help='feature template file')
    arg_parser.add_argument('-iter','--iterations',type=int,help='training iterations')
    arg_parser.add_argument('amr_file',nargs='?',help='amr bank file for preprocessing')
    

    args = arg_parser.parse_args()

    amr_file = args.amr_file
    instances = None
    train_instance = None

    constants.FLAG_DEPPARSER=args.depparser

    # using corenlp to preprocess the sentences 
    if args.mode == 'preprocess':
        instances = preprocess(amr_file)
        print >> experiment_log, "Done preprocessing!"
    # preprocess the JAMR aligned amr
    elif args.mode == 'test_gold_graph':     
        instances = preprocess(amr_file,False)
        #instances = pickle.load(open('data/gold_edge_graph.pkl','rb'))
        pseudo_gold_amr = []
        for inst in instances:
            GraphState.sent = inst.tokens
            pseudo_gold_amr.append(GraphState.get_parsed_amr(inst.gold_graph))
        #pseudo_gold_amr = [GraphState.get_parsed_amr(inst.gold_graph) for inst in instances]
        write_parsed_amr(pseudo_gold_amr,instances,amr_file,'gold')
        #instances = preprocess_aligned(amr_file)
        print "Done output AMR!"
    # do alignment
    elif args.mode == 'align':

        if args.input_file:
            instances = pickle.load(open(args.input_file,'rb'))
        else:
            raise ValueError("Missing data file! specify it using --input or using preprocessing!")
        gold_instances_file = args.input_file.split('.')[0]+'_gold.p'

        print >> log, "Doing alignment..."

        if LOGGED:
            saveerr = sys.stderr
            sys.stderr = open('./log/alignment.log','w')

        amr_aligner = Aligner(verbose=args.verbose)
        ref_graphs = []
        begin = args.begin 
        counter = 1
        #for snt, amr in zip(snts[begin:],amrs[begin:]):
        for i in range(len(instances)):
            snt = instances[i].text
            amr = instances[i].amr
            if args.verbose > 1:
                print >> log, counter
                print >> log, "Sentence:"
                print >> log, snt+'\n'
                
                print >> log, "AMR:"                
                print >> log, amr.to_amr_string()

            alresult = amr_aligner.apply_align(snt,amr)
            ref_amr_graph = SpanGraph.init_ref_graph(amr,alresult)
            #ref_graphs.append(ref_amr_graph)
            instances[i].addGoldGraph(ref_amr_graph)
            if args.verbose > 1:
                #print >> log, "Reference tuples:"
                #print >> log, ref_depGraph.print_tuples()
                print >> log, amr_aligner.print_align_result(alresult,amr)
                #raw_input('ENTER to continue')
            counter += 1

        pickle.dump(instances,open(gold_instances_file,'wb'),pickle.HIGHEST_PROTOCOL)
        #pickle.dump(ref_graphs,open('./data/ref_graph.p','wb'),pickle.HIGHEST_PROTOCOL)
        if LOGGED:
            sys.stderr.close() 
            sys.stderr = saveerr
        print >> log, "Done alignment and gold graph generation."
        sys.exit()
        
    # test user guide actions
    elif args.mode == 'userGuide':
        print 'Read in training instances...'
        train_instances = preprocess(amr_file,False)

        sentID = int(raw_input("Input the sent ID:"))
        amr_parser = Parser()
        amr_parser.testUserGuide(train_instances[sentID])

        sys.exit()

    # test deterministic oracle 
    elif args.mode == 'oracleGuide':
        
        train_instances = preprocess(amr_file,False)

        start_step = args.start_step
        begin = args.begin
        amr_parser = Parser(oracle_type=DETERMINE_TREE_TO_GRAPH_ORACLE_SC,verbose=args.verbose)
        #ref_graphs = pickle.load(open('./data/ref_graph.p','rb'))
        n_correct_total = .0
        n_parsed_total = .0
        n_gold_total = .0
        pseudo_gold_amr = []
        for instance in train_instances[begin:]:
            state = amr_parser.testOracleGuide(instance,start_step)
            n_correct_arc,n1,n_parsed_arc, n_gold_arc,_,_,_ = state.evaluate()
            assert n_correct_arc == n1
            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            n_gold_total += n_gold_arc
            p = n_correct_arc/n_parsed_arc if n_parsed_arc else .0
            indicator = 'PROBLEM!' if p < 0.5 else ''
            if args.dev > 2: print >> sys.stderr, "Accuracy: %s  %s\n" % (p,indicator)
            #if instance.sentID == 704:
            #    import pdb
            #    pdb.set_trace()
            pseudo_gold_amr.append(GraphState.get_parsed_amr(state.A))
            #assert set(state.A.tuples()) == set(instance.gold_graph.tuples())
        pt = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
        rt = n_correct_total/n_gold_total if n_gold_total !=.0 else .0
        ft = 2*pt*rt/(pt+rt) if pt+rt != .0 else .0
        write_parsed_amr(pseudo_gold_amr,train_instances,amr_file,'pseudo-gold')
        print "Total Accuracy: %s, Recall: %s, F-1: %s" % (pt,rt,ft)

        #amr_parser.record_actions('data/action_set.txt')
    elif args.mode == 'train': # actual parsing
        train_instances = preprocess(amr_file,False)
        if args.dev: dev_instances = preprocess(args.dev,False)
        feat_template = args.feat if args.feat else None
        model = Model(elog=experiment_log)
        model.setup(action_type=args.actionset,instances=train_instances,feature_templates_file=feat_template)
        #model.output_feature_generator()
        parser = Parser(model=model,action_type=args.actionset,verbose=args.verbose,elog=experiment_log)
        
        print >> experiment_log, "BEGIN TRAINING!"
        for iter in xrange(1,args.iterations+1):
            print >> experiment_log, "shuffling training instances"
            random.shuffle(train_instances)
            
            print >> experiment_log, "Iteration:",iter
            begin_updates = parser.perceptron.get_num_updates()
            parser.parse_corpus_train(train_instances)
            parser.perceptron.average_weight()
            #model.save_model(args.model+'-iter'+str(iter)+'-'+str(int(time.time()))+'.m')
            model.save_model(args.model+'-iter'+str(iter)+'.m')
            if args.dev:
                print >> experiment_log ,"Result on develop set:"                
                parsed_amr = parser.parse_corpus_test(dev_instances)
                write_parsed_amr(parsed_amr,dev_instances,args.dev)

        print >> experiment_log ,"DONE TRAINING!"
        
    elif args.mode == 'parse':        
        test_instances = preprocess(amr_file,False)

        model = Model.load_model(args.model)
        parser = Parser(model=model,action_type=args.actionset,verbose=args.verbose,elog=experiment_log)
        print >> experiment_log ,"BEGIN PARSING"
        results = parser.parse_corpus_test(test_instances)
        write_parsed_amr(results,test_instances,amr_file)
        print >> experiment_log ,"DONE PARSING"
        #pickle.dump(results,open('data/gold_edge_graph.pkl','wb'),pickle.HIGHEST_PROTOCOL)
        #plt.hist(results)
        #plt.savefig('result.png')
    else:
        arg_parser.print_help()
    
if __name__ == "__main__":
    main()

