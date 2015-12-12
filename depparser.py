#!/usr/bin/env python
import os,subprocess
import sys

VERBOSE = True 
logs = sys.stdout

class DepParser(object):
    
    def __init__(self):
        pass

    def parse(self,sent_filename):
        '''the input should be tokenized sentences (tokenized by stanford CoreNLP) '''
        raise NotImplemented("Must implement setup method!")


class CharniakParser(DepParser):
    
    def parse(self,sent_filename):
        """
        use Charniak parser to parse sentences then convert results to Stanford Dependency
        """
        from bllipparser.ModelFetcher import download_and_install_model
        from bllipparser import RerankingParser
        #path_to_model = './bllip-parser/models/WSJ+Gigaword'
        #if not.path.exists(path_to_model):
        model_type = 'WSJ+Gigaword'
        path_to_model = download_and_install_model(model_type,'./bllip-parser/models')
        print "Loading Charniak parser model: %s ..." % (model_type)
        rrp = RerankingParser.from_unified_model_dir(path_to_model)
        print "Begin Charniak parsing ..."
        parsed_filename = sent_filename+'.charniak.parse'
        parsed_trees = ''
        lineno = 0
        with open(sent_filename,'r') as f, open(parsed_filename,'w') as of:
            for l in f:
                lineno += 1
                print >> logs, 'lineno %s, %s'% (lineno, l)
                parsed_trees = rrp.simple_parse(l.strip().split())
                parsed_trees += '\n'
                of.write(parsed_trees)

        
        # convert parse tree to dependency tree
        print "Convert Charniak parse tree to Stanford Dependency tree ..."
        subprocess.call('./scripts/stdconvert.sh '+parsed_filename,shell=True)
        

class StanfordDepParser(DepParser):
    
    def parse(self,sent_filename):
        """
        separate dependency parser
        """

        jars = ["stanford-parser-3.3.1-models.jar",
                "stanford-parser.jar"]
       
        # if CoreNLP libraries are in a different directory,
        # change the corenlp_path variable to point to them
        stanford_path = "/home/j/llc/cwang24/R_D/AMRParsing/stanfordnlp/stanford-parser/"
        
        java_path = "java"
        classname = "edu.stanford.nlp.parser.lexparser.LexicalizedParser"
        # include the properties file, so you can change defaults
        # but any changes in output format will break parse_parser_results()
        #props = "-props default.properties"
        flags = "-tokenized -sentences newline -outputFormat typedDependencies -outputFormatOptions basicDependencies,markHeadNodes"
        # add and check classpaths
        model = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
        jars = [stanford_path + jar for jar in jars]
        for jar in jars:
            if not os.path.exists(jar):
                print "Error! Cannot locate %s" % jar
                sys.exit(1)

        #Change from ':' to ';'
        # spawn the server
        start_depparser = "%s -Xmx2500m -cp %s %s %s %s %s" % (java_path, ':'.join(jars), classname, flags, model, sent_filename)
        if VERBOSE: print start_depparser
        #incoming = pexpect.run(start_depparser)    
        process = subprocess.Popen(start_depparser.split(),shell=False,stdout=subprocess.PIPE)
        incoming = process.communicate()[0]
        print 'Incoming',incoming
        
        return incoming


class ClearDepParser(DepParser):

    def parse(self,sent_filename):
        subprocess.call(["cp",sent_filename,sent_filename+'.tmp'])
        subprocess.call(["sed","-i",r':a;N;$!ba;s/\n/\n\n/g',sent_filename])
        subprocess.call(["sed","-i",r':a;N;$!ba;s/\s/\n/g',sent_filename])

        clear_path="/home/j/llc/cwang24/Tools/clearnlp"
        extension = "clear.dep"
        
        start_depparser = "%s/clearnlp-parse %s %s" % (clear_path,sent_filename,extension)
        print start_depparser
        extcode = subprocess.call(start_depparser,shell=True)
        dep_result = open(sent_filename+'.'+extension,'r').read()
        subprocess.call(["mv",sent_filename+'.tmp',sent_filename])
        return dep_result

class TurboDepParser(DepParser):
    
    def parse(self,sent_filename):
        turbo_path="/home/j/llc/cwang24/Tools/TurboParser"
        extension = "turbo.dep"

        start_depparser = "%s/scripts/parse-tok.sh %s %s" % (turbo_path,sent_filename,sent_filename+'.'+extension)
        print start_depparser
        subprocess.call(start_depparser,shell=True)
        dep_result = open(sent_filename+'.'+extension,'r').read()
        return dep_result


class MateDepParser(DepParser):
    
    def parse(self,sent_filename):
        mate_path="/home/j/llc/cwang24/Tools/MateParser"
        extension = "mate.dep"

        start_depparser = "%s/parse-eng %s %s" % (mate_path,sent_filename,sent_filename+'.'+extension)
        print start_depparser
        subprocess.call(start_depparser,shell=True)
        dep_result = open(sent_filename+'.'+extension,'r').read()
        return dep_result


if __name__ == "__main__":
    parser = CharniakParser()
    sent_fn = 'data/LDC2015E86_DEFT_Phase_2_AMR_Annotation_R1/data/amrs/split/training/preprocess/deft-p2-amr-r1-amrs-training-dfa.txt.sent.tok.part2'
    parser.parse(sent_fn)