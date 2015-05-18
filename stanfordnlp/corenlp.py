#!/usr/bin/env python
#
# corenlp  - Python interface to Stanford Core NLP tools
# Copyright (c) 2012 Dustin Smith
#   https://github.com/dasmith/stanford-corenlp-python
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import json, optparse, os, re, sys, time, traceback, subprocess
import jsonrpc, pexpect
import subprocess
from progressbar import ProgressBar, Fraction
from unidecode import unidecode
from nltk.tree import Tree
import re
from data import Data
import constants


VERBOSE = True
STATE_START, STATE_TEXT, STATE_WORDS, STATE_TREE, STATE_DEPENDENCY, STATE_COREFERENCE = 0, 1, 2, 3, 4, 5
WORD_PATTERN = re.compile('\[([^\]]+)\]')
CR_PATTERN = re.compile(r"\((\d*),(\d)*,\[(\d*),(\d*)\)\) -> \((\d*),(\d)*,\[(\d*),(\d*)\)\), that is: \"(.*)\" -> \"(.*)\"")

def parse_bracketed(s):
    '''Parse word features [abc=... def = ...]
    Also manages to parse out features that have XML within them
    '''
    word = None
    attrs = {}
    temp = {}
    # Substitute XML tags, to replace them later
    for i, tag in enumerate(re.findall(r"(<[^<>]+>.*<\/[^<>]+>)", s)):
        temp["^^^%d^^^" % i] = tag
        s = s.replace(tag, "^^^%d^^^" % i)
    # Load key-value pairs, substituting as necessary
    for attr, val in re.findall(r"([^=\s]*)=([^=\s]*)", s):
        if val in temp:
            val = temp[val]
        if attr == 'Text':
            word = val
        else:
            attrs[attr.strip()] = val
    return (word, attrs)
   
        
def parse_parser_results(text):
    """ This is the nasty bit of code to interact with the command-line
    interface of the CoreNLP tools.  Takes a string of the parser results
    and then returns a Python list of dictionaries, one for each parsed
    sentence.
    """

    data = Data()
    
    state = STATE_START
    for line in re.split("\n(?!=)",text):
        line = line.strip()
        if line == 'NLP>':
            break
        if line.startswith("Sentence #"):
            state = STATE_TEXT
        
        elif state == STATE_TEXT:
            Data.newSen()
            data.addText(line)
            state = STATE_WORDS
        
        elif state == STATE_WORDS:
            if not line.startswith("[Text="):
                raise Exception('Parse error. Could not find "[Text=" in: %s' % line)
            for s in WORD_PATTERN.findall(line):
                t = parse_bracketed(s)
                data.addToken(t[0], t[1][u'CharacterOffsetBegin'], t[1][u'CharacterOffsetEnd'],
                              t[1][u'Lemma'],t[1][u'PartOfSpeech'],t[1][u'NamedEntityTag'])
            state = STATE_TREE
            parsed = []
        
        elif state == STATE_TREE:
            if len(line) == 0:
                state = STATE_DEPENDENCY
                parsed = " ".join(parsed)
                #data.addTree(Tree.parse(parsed))
            else:
                parsed.append(line)
        
        elif state == STATE_DEPENDENCY:
            if len(line) == 0:
                state = STATE_COREFERENCE
            else:
                pass
                '''
                split_entry = re.split("\(|, ", line[:-1])
                if len(split_entry) == 3:
                    rel, l_lemma, r_lemma = split_entry
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', l_lemma)
                    l_lemma, l_index = m.group('lemma'), m.group('index')
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', r_lemma)
                    r_lemma, r_index = m.group('lemma'), m.group('index')

                    data.addDependency( rel, l_lemma, r_lemma, l_index, r_index)
                '''
        elif state == STATE_COREFERENCE:
            if "Coreference set" in line:
##                if 'coref' not in results:
##                    results['coref'] = []
                coref_set = []
                data.addCoref(coref_set)
            else:
                for src_i, src_pos, src_l, src_r, sink_i, sink_pos, sink_l, sink_r, src_word, sink_word in CR_PATTERN.findall(line):
                    src_i, src_pos, src_l, src_r = int(src_i)-1, int(src_pos)-1, int(src_l)-1, int(src_r)-1
                    sink_i, sink_pos, sink_l, sink_r = int(sink_i)-1, int(sink_pos)-1, int(sink_l)-1, int(sink_r)-1
                    coref_set.append(((src_word, src_i, src_pos, src_l, src_r), (sink_word, sink_i, sink_pos, sink_l, sink_r)))
    
    return data

def add_sep_dependency(instances,result):
    if constants.FLAG_DEPPARSER == 'stanford':
        i = 0
        for line in result.split('\n'):
            if line.strip():
                split_entry = re.split("\(|, ", line[:-1])

                if len(split_entry) == 3:
                    rel, l_lemma, r_lemma = split_entry
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', l_lemma)
                    l_lemma, l_index = m.group('lemma'), m.group('index')
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', r_lemma)
                    r_lemma, r_index = m.group('lemma'), m.group('index')

                    instances[i].addDependency( rel, l_index, r_index )

            else:
                i += 1

    elif constants.FLAG_DEPPARSER in ["turbo","malt"]:
        i = 0
        for line in result.split('\n'):
            if line.strip():
                line = line.split()
                instances[i].addDependency( line[7], line[6], line[0])
            else:
                i += 1

    elif constants.FLAG_DEPPARSER == "mate":
        i = 0
        for line in result.split('\n'):
            if line.strip():
                line = line.split()
                instances[i].addDependency( line[11], line[9], line[0])
            else:
                i += 1
    elif constants.FLAG_DEPPARSER == 'stdconv+charniak':
        i = 0
        for line in result.split('\n'):
            if line.strip():
                split_entry = re.split("\(|, ", line[:-1])
                
                if len(split_entry) == 3:
                    rel, l_lemma, r_lemma = split_entry
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', l_lemma)
                    l_lemma, l_index = m.group('lemma'), m.group('index')
                    m = re.match(r'(?P<lemma>[^\^]+)(\^(?P<trace>[^-]+))?-(?P<index>[^-]+)', r_lemma)
                    r_lemma,r_trace, r_index = m.group('lemma'), m.group('trace'), m.group('index')

                    if r_index != 'null':
                        #print >> sys.stderr, line                        
                        instances[i].addDependency( rel, l_index, r_index )
                    #if r_trace is not None:
                    #    instances[i].addTrace( rel, l_index, r_trace )                      
                
            else:
                i += 1
    else:
        raise ValueError("Unknown dependency format!")

class StanfordCoreNLP(object):
    """
    Command-line interaction with Stanford's CoreNLP java utilities.
    Can be run as a JSON-RPC server or imported as a module.
    """


    def __init__(self):
        Data.current_sen = 1
        pass
        
    def setup(self):
        """
        Checks the location of the jar files.
        Spawns the server as a process.
        """
        jars = ["stanford-corenlp-3.2.0.jar",
                "stanford-corenlp-3.2.0-models.jar",
                "joda-time.jar",
                "xom.jar",
                "jollyday.jar"]
       
        # if CoreNLP libraries are in a different directory,
        # change the corenlp_path variable to point to them
        corenlp_path = os.path.relpath(__file__).split('/')[0]+"/stanford-corenlp-full-2013-06-20/"
        #corenlp_path = "stanford-corenlp-full-2013-06-20/"
        
        java_path = "java"
        classname = "edu.stanford.nlp.pipeline.StanfordCoreNLP"
        # include the properties file, so you can change defaults
        # but any changes in output format will break parse_parser_results()
        props = "-props "+ os.path.relpath(__file__).split('/')[0]+"/default.properties"
        
        # add and check classpaths
        jars = [corenlp_path + jar for jar in jars]
        for jar in jars:
            if not os.path.exists(jar):
                print "Error! Cannot locate %s" % jar
                sys.exit(1)

        #Change from ':' to ';'
        # spawn the server
        start_corenlp = "%s -Xmx1800m -cp %s %s %s" % (java_path, ':'.join(jars), classname, props)
        if VERBOSE: print start_corenlp
        self.corenlp = pexpect.spawn(start_corenlp)
        
        # show progress bar while loading the models
        widgets = ['Loading Models: ', Fraction()]
        pbar = ProgressBar(widgets=widgets, maxval=5, force_update=True).start()
        self.corenlp.expect("done.", timeout=20) # Load pos tagger model (~5sec)
        pbar.update(1)
        self.corenlp.expect("done.", timeout=200) # Load NER-all classifier (~33sec)
        pbar.update(2)
        self.corenlp.expect("done.", timeout=600) # Load NER-muc classifier (~60sec)
        pbar.update(3)
        self.corenlp.expect("done.", timeout=600) # Load CoNLL classifier (~50sec)
        pbar.update(4)
        #self.corenlp.expect("done.", timeout=200) # Loading PCFG (~3sec)
        #pbar.update(5)
        self.corenlp.expect("Entering interactive shell.")
        pbar.finish()
    
    def _parse(self, text):
        """
        This is the core interaction with the parser.
               
        """
        # clean up anything leftover
        while True:
            try:
                self.corenlp.read_nonblocking (4000, 0.3)
            except pexpect.TIMEOUT:
                break
        
        self.corenlp.sendline(text)
        
        # How much time should we give the parser to parse it?
        # the idea here is that you increase the timeout as a 
        # function of the text's length.
        # anything longer than 5 seconds requires that you also
        # increase timeout=5 in jsonrpc.py
        max_expected_time = min(10, 3 + len(text) / 20.0)
        end_time = time.time() + max_expected_time
        
        incoming = ""
        while True:
            # Time left, read more data
            try:
                incoming += self.corenlp.read_nonblocking(2000, 1)
                if "\nNLP>" in incoming: break
                time.sleep(0.0001)
            except pexpect.TIMEOUT:
                if end_time - time.time() < 0:
                    print "[ERROR] Timeout"
                    return {'error': "timed out after %f seconds" % max_expected_time,
                            'input': text,
                            'output': incoming}
                else:
                    continue
            except pexpect.EOF:
                break
        
        if VERBOSE: print "%s\n%s" % ('='*40, repr(incoming))
        return incoming


    def sep_depparsing(self,sent_filename):
        """
        separate dependency parser
        """

        jars = ["stanford-parser-3.3.1-models.jar",
                "stanford-parser.jar"]
       
        # if CoreNLP libraries are in a different directory,
        # change the corenlp_path variable to point to them
        corenlp_path = os.path.relpath(__file__).split('/')[0]+"/stanford-parser-full-2014-01-04/"
        
        java_path = "java"
        classname = "edu.stanford.nlp.parser.lexparser.LexicalizedParser"
        # include the properties file, so you can change defaults
        # but any changes in output format will break parse_parser_results()
        #props = "-props default.properties"
        flags = "-sentences newline -outputFormat typedDependencies -outputFormatOptions basicDependencies,markHeadNodes"
        # add and check classpaths
        model = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
        jars = [corenlp_path + jar for jar in jars]
        for jar in jars:
            if not os.path.exists(jar):
                print "Error! Cannot locate %s" % jar
                sys.exit(1)

        #Change from ':' to ';'
        # spawn the server
        start_depparser = "%s -Xmx1800m -cp %s %s %s %s %s" % (java_path, ':'.join(jars), classname, flags, model, sent_filename)
        if VERBOSE: print start_depparser
        #incoming = pexpect.run(start_depparser)    
        process = subprocess.Popen(start_depparser.split(),shell=False,stdout=subprocess.PIPE)
        incoming = process.communicate()[0]
        print 'Incoming',incoming
        
        return incoming

    def parse(self, sent_filename):
        """ 
        This function takes a text string, sends it to the Stanford parser,
        reads in the result, parses the results and returns a list
        of data instances for each parsed sentence. Dependency parsing may operate 
        seperately for easy changing dependency parser.
        """
        
        instances = []
        prp_filename = sent_filename.rsplit('.',1)[0]+'.prp' # preprocessing file
        if os.path.exists(prp_filename):
            print 'Read token,lemma,name entity file %s...' % (prp_filename)
            prp_result = open(prp_filename,'r').read()

            for result in prp_result.split('-'*40)[1:]:                
                try:
                    data = parse_parser_results(result)
                except Exception, e:
                    if VERBOSE: print traceback.format_exc()
                    raise e
                instances.append(data)
        else:
            output_prp = open(prp_filename,'w')            
            for i,line in enumerate(open(sent_filename,'r').readlines()):
                result = self._parse(line)
                output_prp.write("%s\n%s"%('-'*40,result))
                try:
                    data = parse_parser_results(result)
                except Exception, e:
                    if VERBOSE: print traceback.format_exc()
                    raise e
                instances.append(data)
            output_prp.close()
        
        #if seq_depparsing:
        if constants.FLAG_DEPPARSER == 'stanford':
            dep_filename = sent_filename +'.tok.stanford.dep'
            if os.path.exists(dep_filename):
                print 'Read dependency file %s...' % (dep_filename)
                dep_result = open(dep_filename,'r').read()
            else:
                print 'run dependency parsing seperately.'
                dep_result = self.sep_depparsing(sent_filename)
                output_dep = open(dep_filename,'w')
                output_dep.write(dep_result)
                output_dep.close()
            add_sep_dependency(instances,dep_result)
        else:
            dep_filename = None
            if constants.FLAG_DEPPARSER == 'stdconv+charniak':
                dep_filename = sent_filename+'.tok.charniak.parse.dep'
            elif constants.FLAG_DEPPARSER == 'turbo':
                dep_filename = sent_filename +'.tok.turbo.dep'
            elif constants.FLAG_DEPPARSER == 'mate':
                dep_filename = sent_filename + '.tok.mate.dep'
            elif constants.FLAG_DEPPARSER == 'malt':
                dep_filename =sent_filename + '.tok.malt.dep'
            else:
                raise ValueError('Invalid Dependency Format!')                
            if os.path.exists(dep_filename):
                print 'Read dependency file %s...' % (dep_filename)
                dep_result = open(dep_filename,'r').read()
            else:
                raise FileNotFoundError('charniak parse dependency file %s not found'%(dep_filename))
            add_sep_dependency(instances,dep_result)

        return instances


if __name__ == '__main__':
    """
    The code below starts an JSONRPC server
    """
    parser = optparse.OptionParser(usage="%prog [OPTIONS]")
    parser.add_option('-t', '--type', default='serve',
                      help='Choose between serve or kill')
    parser.add_option('-p', '--port', default='2346',
                      help='Port to serve or kill on (default 2346)')
    parser.add_option('-H', '--host', default='127.0.0.1',
                      help='Host to serve on (default localhost; 0.0.0.0 to make public)')
    
    options, args = parser.parse_args()

    if options.type == 'serve':
        server = jsonrpc.Server(jsonrpc.JsonRpc20(),
                                jsonrpc.TransportTcpIp(addr=(options.host, int(options.port))))
        
        nlp = StanfordCoreNLP()
        server.register_function(nlp.parse)
        
        print 'Serving on http://%s:%s' % (options.host, options.port)
        server.serve()
    else:
        popen = subprocess.Popen(['netstat', '-nao'],
                             shell=False,
                             stdout=subprocess.PIPE)
        (data, err) = popen.communicate()

##        pattern = "^\s+TCP.*" + options.port + ".*(?P<pid>[0-9]*)\s+$"
        pattern = "^\s+TCP.*"+ options.port + ".*\s(?P<pid>\d+)\s*$"
        prog = re.compile(pattern)
        print pattern
        for line in data.split('\n'):
            match = re.match(prog, line)
            if match:
                pid = match.group('pid')
                print pid
                subprocess.Popen(['taskkill', '/PID', pid, '/F'])
