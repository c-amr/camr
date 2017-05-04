#!/usr/bin/python

"""
Interface for the parser:
parse command line 
read in corpus
"""
from __future__ import absolute_import
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

def write_parsed_amr(parsed_amr,instances,amr_file,suffix='parsed',hand_alignments=None):
    output = open(amr_file+'.'+suffix,'w')
    for pamr,inst in zip(parsed_amr,instances):
        if inst.comment:
            output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['id','date','snt-type','annotator'])))
            output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['snt','tok'])))
            if hand_alignments:
                output.write('# ::alignments %s ::gold\n' % (hand_alignments[inst.comment['id']]))
            #output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['alignments'])))
        else:
            output.write('# ::id %s\n'%(inst.sentID))
            output.write('# ::snt %s\n'%(inst.text))

        try:
            output.write(pamr.to_amr_string())
        except TypeError:
            import pdb
            pdb.set_trace()
        output.write('\n\n')
    output.close()

def write_span_graph(span_graph_pairs,instances,amr_file,suffix='spg'):
    output_d = open(amr_file+'.'+suffix+'.dep', 'w')
    output_p = open(amr_file+'.'+suffix+'.parsed','w')
    output_g = open(amr_file+'.'+suffix+'.gold','w')

    for i in xrange(len(instances)):
        output_d.write('# id:%s\n%s' % (instances[i].comment['id'],instances[i].printDep()))
        output_p.write('# id:%s\n%s' % (instances[i].comment['id'],span_graph_pairs[i][0].print_dep_style_graph()))
        output_g.write('# id:%s\n%s' % (instances[i].comment['id'],span_graph_pairs[i][1].print_dep_style_graph()))
        output_p.write('# eval:Unlabeled Precision:%s Recall:%s F1:%s\n' % (span_graph_pairs[i][2][0],span_graph_pairs[i][2][1],span_graph_pairs[i][2][2]))
        output_p.write('# eval:Labeled Precision:%s Recall:%s F1:%s\n' % (span_graph_pairs[i][2][3],span_graph_pairs[i][2][4],span_graph_pairs[i][2][5]))
        output_p.write('# eval:Tagging Precision:%s Recall:%s\n' % (span_graph_pairs[i][2][6],span_graph_pairs[i][2][7]))
        output_d.write('\n')
        output_p.write('\n')
        output_g.write('\n')

    output_d.close()
    output_p.close()
    output_g.close()
        
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
    arg_parser.add_argument('-s','--start_step',type=int,default=0,help='specify which step to begin oracle testing;for debug')
    #arg_parser.add_argument('-i','--input_file',help='the input: preprocessed data instances file for aligner or training')
    arg_parser.add_argument('-d','--dev',help='development file')
    arg_parser.add_argument('-a','--add',help='additional training file')
    arg_parser.add_argument('-as','--actionset',choices=['basic'],default='basic',help='choose different action set')
    arg_parser.add_argument('-m','--mode',choices=['preprocess','test_gold_graph','align','userGuide','oracleGuide','train','parse','eval'],help="preprocess:generate pos tag, dependency tree, ner\n" "align:do alignment between AMR graph and sentence string")
    arg_parser.add_argument('-dp','--depparser',choices=['stanford','stanfordConvert','stdconv+charniak','clear','mate','turbo'],default='stdconv+charniak',help='choose the dependency parser')
    arg_parser.add_argument('--coref',action='store_true',help='flag to enable coreference information')
    arg_parser.add_argument('--prop',action='store_true',help='flag to enable semantic role labeling information')
    arg_parser.add_argument('--rne',action='store_true',help='flag to enable rich name entity')
    arg_parser.add_argument('--verblist',action='store_true',help='flag to enable verbalization list')
    #arg_parser.add_argument('--onto',action='store_true',help='flag to enable charniak parse result trained on ontonotes')
    arg_parser.add_argument('--onto',choices=['onto','onto+bolt','wsj'],default='wsj',help='choose which charniak parse result trained on ontonotes')
    arg_parser.add_argument('--model',help='specify the model file')
    arg_parser.add_argument('--feat',help='feature template file')
    arg_parser.add_argument('-iter','--iterations',default=1,type=int,help='training iterations')
    arg_parser.add_argument('amr_file',nargs='?',help='amr annotation file/input sentence file for parsing')
    arg_parser.add_argument('--prpfmt',choices=['xml','plain'],default='plain',help='preprocessed file format')
    arg_parser.add_argument('--amrfmt',choices=['sent','amr','amreval'],default='sent',help='specifying the input file format')
    arg_parser.add_argument('--smatcheval',action='store_true',help='give evaluation score using smatch')
    arg_parser.add_argument('-e','--eval',nargs=2,help='Error Analysis: give parsed AMR file and gold AMR file')
    arg_parser.add_argument('--section',choices=['proxy','all'],default='all',help='choose section of the corpus. Only works for LDC2014T12 dataset.')

    args = arg_parser.parse_args()

    amr_file = args.amr_file
    instances = None
    train_instance = None
    constants.FLAG_COREF=args.coref
    constants.FLAG_PROP=args.prop
    constants.FLAG_RNE=args.rne
    constants.FLAG_VERB=args.verblist
    constants.FLAG_ONTO=args.onto
    constants.FLAG_DEPPARSER=args.depparser

    # using corenlp to preprocess the sentences 
    if args.mode == 'preprocess':
        instances = preprocess(amr_file,START_SNLP=True,INPUT_AMR=args.amrfmt, PRP_FORMAT=args.prpfmt)
        print "Done preprocessing!"
    # preprocess the JAMR aligned amr
    elif args.mode == 'test_gold_graph':     
        instances = preprocess(amr_file,START_SNLP=False,INPUT_AMR=args.amrfmt, PRP_FORMAT=args.prpfmt)
        #instances = pickle.load(open('data/gold_edge_graph.pkl','rb'))
        gold_amr = []
        for inst in instances:
            GraphState.sent = inst.tokens
            gold_amr.append(GraphState.get_parsed_amr(inst.gold_graph))
        #pseudo_gold_amr = [GraphState.get_parsed_amr(inst.gold_graph) for inst in instances]
        write_parsed_amr(gold_amr,instances,amr_file,'abt.gold')
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
        
        train_instances = preprocess(amr_file,START_SNLP=False,INPUT_AMR=args.amrfmt, PRP_FORMAT=args.prpfmt)
        try:
            hand_alignments = load_hand_alignments(amr_file+str('.hand_aligned'))
        except IOError:
            hand_alignments = []


        start_step = args.start_step
        begin = args.begin
        amr_parser = Parser(oracle_type=DET_T2G_ORACLE_ABT,verbose=args.verbose)
        #ref_graphs = pickle.load(open('./data/ref_graph.p','rb'))
        n_correct_total = .0
        n_parsed_total = .0
        n_gold_total = .0
        pseudo_gold_amr = []
        n_correct_tag_total = .0
        n_parsed_tag_total = 0.
        n_gold_tag_total = .0

        
        gold_amr = []
        aligned_instances = []
        #print "shuffling training instances"
        #random.shuffle(train_instances)
        for instance in train_instances[begin:]:
            
            if hand_alignments and instance.comment['id'] not in hand_alignments: continue
            state = amr_parser.testOracleGuide(instance,start_step)
            n_correct_arc,n1,n_parsed_arc, n_gold_arc,n_correct_tag,n_parsed_tag,n_gold_tag = state.evaluate()
            #assert n_correct_arc == n1
            if n_correct_arc != n1:
                import pdb
                pdb.set_trace()
            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            n_gold_total += n_gold_arc
            p = n_correct_arc/n_parsed_arc if n_parsed_arc else .0
            r = n_correct_arc/n_gold_arc if n_gold_arc else .0
            indicator = 'PROBLEM!' if p < 0.5 else ''
            if args.verbose > 2: print >> sys.stderr, "Precision: %s Recall: %s  %s\n" % (p,r,indicator)
            n_correct_tag_total +=  n_correct_tag
            n_parsed_tag_total +=  n_parsed_tag
            n_gold_tag_total += n_gold_tag
            p1 = n_correct_tag/n_parsed_tag if n_parsed_tag else .0
            r1 = n_correct_tag/n_gold_tag if n_gold_tag else .0
            if args.verbose > 2: print >> sys.stderr,"Tagging Precision:%s Recall:%s" % (p1,r1)

            instance.comment['alignments'] += ''.join(' %s-%s|%s'%(idx-1,idx,instance.amr.get_pid(state.A.abt_node_table[idx])) for idx in state.A.abt_node_table if isinstance(idx,int))

            aligned_instances.append(instance)
            pseudo_gold_amr.append(GraphState.get_parsed_amr(state.A))
            #gold_amr.append(instance.amr)
            #assert set(state.A.tuples()) == set(instance.gold_graph.tuples())
        pt = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
        rt = n_correct_total/n_gold_total if n_gold_total !=.0 else .0
        ft = 2*pt*rt/(pt+rt) if pt+rt != .0 else .0
        write_parsed_amr(pseudo_gold_amr,aligned_instances,amr_file,'pseudo-gold',hand_alignments)
        print "Total Accuracy: %s, Recall: %s, F-1: %s" % (pt,rt,ft)

        tp = n_correct_tag_total/n_parsed_tag_total if n_parsed_tag_total != .0 else .0
        tr = n_correct_tag_total/n_gold_tag_total if n_gold_tag_total != .0 else .0
        print "Tagging Precision:%s Recall:%s" % (tp,tr)

        #amr_parser.record_actions('data/action_set.txt')
    elif args.mode == 'train': # training
        print "Parser Config:"
        print "Incorporate Coref Information: %s"%(constants.FLAG_COREF)
        print "Incorporate SRL Information: %s"%(constants.FLAG_PROP)
        print "Substitue the normal name entity tag with rich name entity tag: %s"%(constants.FLAG_RNE)
        print "Using verbalization list: %s"%(constants.FLAG_VERB)
        print "Using charniak parser trained on ontonotes: %s"%(constants.FLAG_ONTO)
        print "Dependency parser used: %s"%(constants.FLAG_DEPPARSER)
        train_instances = preprocess(amr_file,START_SNLP=False,INPUT_AMR=args.amrfmt,PRP_FORMAT=args.prpfmt)
        if args.add: train_instances = train_instances + preprocess(args.add,START_SNLP=True,INPUT_AMR=args.amrfmt,PRP_FORMAT=args.prpfmt)
        if args.dev: dev_instances = preprocess(args.dev,START_SNLP=False,INPUT_AMR=args.amrfmt,PRP_FORMAT=args.prpfmt)


        if args.section != 'all':
            print "Choosing corpus section: %s"%(args.section)
            tcr = constants.get_corpus_range(args.section,'train')
            train_instances = train_instances[tcr[0]:tcr[1]]
            if args.dev:
                dcr = constants.get_corpus_range(args.section,'dev')
                dev_instances = dev_instances[dcr[0]:dcr[1]]

        
        feat_template = args.feat if args.feat else None
        model = Model(elog=experiment_log)
        #model.output_feature_generator()
        parser = Parser(model=model,oracle_type=DET_T2G_ORACLE_ABT,action_type=args.actionset,verbose=args.verbose,elog=experiment_log)
        model.setup(action_type=args.actionset,instances=train_instances,parser=parser,feature_templates_file=feat_template)
        
        print >> experiment_log, "BEGIN TRAINING!"
        best_fscore = 0.0
        best_pscore = 0.0
        best_rscore = 0.0
        best_model = None
        best_iter = 1
        for iter in xrange(1,args.iterations+1):
            print >> experiment_log, "shuffling training instances"
            random.shuffle(train_instances)
            
            print >> experiment_log, "Iteration:",iter
            begin_updates = parser.perceptron.get_num_updates()
            parser.parse_corpus_train(train_instances)
            parser.perceptron.average_weight()
            
            if args.dev:
                print >> experiment_log ,"Result on develop set:"                
                _,parsed_amr = parser.parse_corpus_test(dev_instances)
                parsed_suffix = args.section+'.'+args.model.split('.')[-1]+'.'+str(iter)+'.parsed'
                write_parsed_amr(parsed_amr,dev_instances,args.dev,parsed_suffix)
                if args.smatcheval:
                    smatch_path = "./smatch_2.0.2/smatch.py"
                    python_path = 'python'
                    options = '--pr -f'
                    parsed_filename = args.dev+'.'+parsed_suffix
                    command = '%s %s %s %s %s' % (python_path, smatch_path, options, parsed_filename, args.dev)
                    
                    print 'Evaluation using command: ' + (command)
                    #print subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                    eval_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
                    print eval_output
                    pscore = float(eval_output.split('\n')[0].split(':')[1].rstrip())
                    rscore = float(eval_output.split('\n')[1].split(':')[1].rstrip())
                    fscore = float(eval_output.split('\n')[2].split(':')[1].rstrip())
                    if fscore > best_fscore:
                        best_model = model
                        best_iter = iter
                        best_fscore = fscore
                        best_pscore = pscore
                        best_rscore = rscore

        if best_model is not None:
            print >> experiment_log, "Best result on iteration %d:\n Precision: %f\n Recall: %f\n F-score: %f" % (best_iter, best_pscore, best_rscore, best_fscore)
            best_model.save_model(args.model+'.m')
        print >> experiment_log ,"DONE TRAINING!"
        
    elif args.mode == 'parse': # actual parsing
        test_instances = preprocess(amr_file,START_SNLP=False,INPUT_AMR=args.amrfmt,PRP_FORMAT=args.prpfmt)
        if args.section != 'all':
            print "Choosing corpus section: %s"%(args.section)
            tcr = constants.get_corpus_range(args.section,'test')
            test_instances = test_instances[tcr[0]:tcr[1]]
            
        #random.shuffle(test_instances)
        print >> experiment_log, "Loading model: ", args.model 
        model = Model.load_model(args.model)
        parser = Parser(model=model,oracle_type=DET_T2G_ORACLE_ABT,action_type=args.actionset,verbose=args.verbose,elog=experiment_log)
        print >> experiment_log ,"BEGIN PARSING"
        span_graph_pairs,results = parser.parse_corpus_test(test_instances)
        parsed_suffix = '%s.%s.parsed'%(args.section,args.model.split('.')[-2])
        write_parsed_amr(results,test_instances,amr_file,suffix=parsed_suffix)
        #write_span_graph(span_graph_pairs,test_instances,amr_file,suffix='spg.50')
        ################
        # for eval     #
        ################
        #pickle.dump(span_graph_pairs,open('data/eval/%s_spg_pair.pkl'%(amr_file),'wb'),pickle.HIGHEST_PROTOCOL)
        #pickle.dump(test_instances,open('data/eval/%s_instances.pkl'%(amr_file),'wb'),pickle.HIGHEST_PROTOCOL)
        print >> experiment_log ,"DONE PARSING"
        if args.smatcheval:
            smatch_path = "./smatch_2.0.2/smatch.py"
            python_path = 'python'
            options = '--pr -f'
            parsed_filename = amr_file+'.'+parsed_suffix
            command = '%s %s %s %s %s' % (python_path,smatch_path,options,parsed_filename, amr_file)
                    
            print 'Evaluation using command: ' + (command)
            print subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

            
        #plt.hist(results)
        #plt.savefig('result.png')

    elif args.mode == 'eval':
        '''break down error analysis'''
        # TODO: here use pickled file, replace it with parsed AMR and gold AMR
        span_graph_pairs = pickle.load(open(args.eval[0],'rb'))
        instances = pickle.load(open(args.eval[1],'rb'))
        
        amr_parser = Parser(oracle_type=DET_T2G_ORACLE_ABT,verbose=args.verbose)
        error_stat = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for spg_pair,instance in zip(span_graph_pairs,instances):
            amr_parser.errorAnalyze(spg_pair[0],spg_pair[1],instance,error_stat)

    else:
        arg_parser.print_help()

if __name__ == "__main__":
    main()

