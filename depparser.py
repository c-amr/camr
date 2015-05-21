#!/usr/bin/env python
import os,subprocess

VERBOSE = True

class DepParser(object):
    
    def __init__(self):
        pass

    def parse(self,sent_filename):
        '''the input should be tokenized sentences (tokenized by stanford CoreNLP) '''
        raise NotImplemented("Must implement setup method!")


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
