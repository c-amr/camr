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

VERBOSE = True
STATE_START, STATE_TEXT, STATE_WORDS, STATE_TREE, STATE_DEPENDENCY, STATE_COREFERENCE, STATE_SENT_ERROR, STATE_TEXT_ERROR, STATE_WORD_ERROR = 0, 1, 2, 3, 4, 5, 6, 7, 8
WORD_PATTERN = re.compile('\[([^\]]+)\]')
WORD_ERROR_PATTERN = re.compile('\[([^\]]+)')
CR_PATTERN = re.compile(r"\((\d*),(\d*),\[(\d*),(\d*)\)\) -> \((\d*),(\d*),\[(\d*),(\d*)\)\), that is: \"(.*)\" -> \"(.*)\"")
SENTENCE_NO_PATTERN = re.compile(r"^Sentence\s*#\d+\s*\(\d+\s*tokens\):")

def parse_bracketed(s):
    '''Parse word features [abc=... def = ...]
    Also manages to parse out features that have XML within them
    '''
    word = None
    attrs = {}
    temp = {}
    s = s.replace('\r\n','')
    # Substitute XML tags, to replace them later
    for i, tag in enumerate(re.findall(r"(<[^<>]+>.*<\/[^<>]+>)", s)):
        temp["^^^%d^^^" % i] = tag
        s = s.replace(tag, "^^^%d^^^" % i)
    # Load key-value pairs, substituting as necessary
    for attr, _, val in re.findall(r"([^=\s]*)(\r\n)?=([^=\s]*)", s):
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
    #for line in re.split("\r\n(?![^\[]*\])",text):
    for line in re.split("\r\n", text):
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
            if len(line) == 0:
                continue
            if not line.startswith("[Text="):
                raise Exception('Parse error. Could not find "[Text=" in: %s' % line)
            for s in WORD_PATTERN.findall(line):
                t = parse_bracketed(s)
                if t[0] == '': continue
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
                # don't need here
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
                #if 'coref' not in results:
                #    results['coref'] = []
                coref_set = []
                data.addCoref(coref_set)
            else:
                for src_i, src_pos, src_l, src_r, sink_i, sink_pos, sink_l, sink_r, src_word, sink_word in CR_PATTERN.findall(line):
                    src_i, src_pos, src_l, src_r = int(src_i), int(src_pos), int(src_l), int(src_r)
                    sink_i, sink_pos, sink_l, sink_r = int(sink_i), int(sink_pos), int(sink_l), int(sink_r)
                    coref_set.append(((src_word, src_i, src_pos, src_l, src_r), (sink_word, sink_i, sink_pos, sink_l, sink_r)))
    
    return data


def parse_parser_results_new(text):
    """ This is the nasty bit of code to interact with the command-line
    interface of the CoreNLP tools.  Takes a string of the parser results
    and then returns a Python list of dictionaries, one for each parsed
    sentence.

    updated for newer version of stanford corenlp -- 2015
    """
    data_list = []
    data = None
    lastline = None
    following_line = None
    state = STATE_START
    #for line in re.split("\r\n(?![^\[]*\])",text):
    seqs = re.split("\r\n", text)
    i = 0

    #for line in re.split("\r\n", text):
    while i < len(seqs):
        line = seqs[i]
        line = line.strip()

        if line.startswith('NLP>'): # end
            if data: data_list.append(data) # add last one
            break
        if line.startswith("Sentence #"):
            if data: data_list.append(data)
            data = Data()
            if SENTENCE_NO_PATTERN.match(line):
                state = STATE_TEXT
            else:
                lastline = line
                state = STATE_SENT_ERROR
            i += 1
            
        elif state == STATE_SENT_ERROR:
            line = lastline + line
            assert SENTENCE_NO_PATTERN.match(line) is not None
            state = STATE_TEXT
            i += 1
            
        elif state == STATE_TEXT_ERROR:
            line = line + following_line
            data.addText(line)
            state = STATE_WORDS
            i += 2
        
        elif state == STATE_TEXT:
            Data.newSen()
            data.addText(line)
            state = STATE_WORDS
            i += 1
        
        elif state == STATE_WORDS:
            if len(line) == 0:
                continue
            if not line.startswith("[Text="):
                #raise Exception('Parse error. Could not find "[Text=" in: %s' % line)
                print >> sys.stderr, 'Parse error. Could not find "[Text=" in: %s' % line
                print >> sys.stderr, 'Attempt to fixing error.'
                following_line = line
                state = STATE_TEXT_ERROR
                i -= 1
                continue
                
            #for s in WORD_PATTERN.findall(line):
            wline = line
            while WORD_PATTERN.match(wline):
                t = parse_bracketed(wline[1:-1])
                if t[0] == '':
                    i += 1
                    wline = seqs[i]
                    continue
                data.addToken(t[0], t[1][u'CharacterOffsetBegin'], t[1][u'CharacterOffsetEnd'],
                              t[1][u'Lemma'],t[1][u'PartOfSpeech'],t[1][u'NamedEntityTag'])
                i += 1
                wline = seqs[i]

            if WORD_ERROR_PATTERN.match(wline): # handle format error
                wline = wline + seqs[i+1]
                wline = wline.strip()
                t = parse_bracketed(wline[1:-1])
                data.addToken(t[0], t[1][u'CharacterOffsetBegin'], t[1][u'CharacterOffsetEnd'],
                              t[1][u'Lemma'],t[1][u'PartOfSpeech'],t[1][u'NamedEntityTag'])
                i+=2
                state = STATE_WORDS
                continue
            state = STATE_TREE
            parsed = []
        
        elif state == STATE_TREE:
            if len(line) == 0:
                state = STATE_DEPENDENCY
                parsed = " ".join(parsed)
                i += 1
                #data.addTree(Tree.parse(parsed))
            else:
                parsed.append(line)
                i += 1
        
        elif state == STATE_DEPENDENCY:
            if len(line) == 0:
                state = STATE_COREFERENCE
            else:
                pass
                '''
                # don't need here
                split_entry = re.split("\(|, ", line[:-1])
                if len(split_entry) == 3:
                    rel, l_lemma, r_lemma = split_entry
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', l_lemma)
                    l_lemma, l_index = m.group('lemma'), m.group('index')
                    m = re.match(r'(?P<lemma>.+)-(?P<index>[^-]+)', r_lemma)
                    r_lemma, r_index = m.group('lemma'), m.group('index')

                    data.addDependency( rel, l_lemma, r_lemma, l_index, r_index)
                '''

            i += 1
        elif state == STATE_COREFERENCE:
            if "Coreference set" in line:
                #if 'coref' not in results:
                #    results['coref'] = []
                coref_set = []
                data.addCoref(coref_set)
            else:
                for src_i, src_pos, src_l, src_r, sink_i, sink_pos, sink_l, sink_r, src_word, sink_word in CR_PATTERN.findall(line):
                    src_i, src_pos, src_l, src_r = int(src_i), int(src_pos), int(src_l), int(src_r)
                    sink_i, sink_pos, sink_l, sink_r = int(sink_i), int(sink_pos), int(sink_l), int(sink_r)
                    coref_set.append(((src_word, src_i, src_pos, src_l, src_r), (sink_word, sink_i, sink_pos, sink_l, sink_r)))

            i += 1
        else:
            i += 1
        
    return data_list






def add_sep_dependency(instances,result):
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
                
                instances[i].addDependency( rel, l_lemma, r_lemma, l_index, r_index )
                
        else:
            i += 1

class StanfordCoreNLP(object):
    """
    Command-line interaction with Stanford's CoreNLP java utilities.
    Can be run as a JSON-RPC server or imported as a module. We use CoreNLP 
    to preprocess the sentence to get universal tokenization, lemma, name entity 
    and POS if possible. However, we may use different dependency parsers here.
    """


    def __init__(self):
        Data.current_sen = 1
        pass
        
    def setup(self):
        """
        Checks the location of the jar files.
        Spawns the server as a process.
        """
        #jars = ["stanford-corenlp-3.2.0.jar",
        #        "stanford-corenlp-3.2.0-models.jar",
        #        "joda-time.jar",
        #        "xom.jar",
        #        "jollyday.jar"]

        jars = ["stanford-corenlp-3.5.2.jar",
                "stanford-corenlp-3.5.2-models.jar",
                "joda-time.jar",
                "xom.jar",
                "jollyday.jar",
                "protobuf.jar",
                "javax.json.jar",
                "ejml-0.23.jar"]
        
       
        # if CoreNLP libraries are in a different directory,
        # change the corenlp_path variable to point to them
        corenlp_path = os.path.relpath(__file__).split('/')[0]+"/stanford-corenlp-full-2015-04-20/"
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
        start_corenlp = "%s -Xmx2500m -cp %s %s %s" % (java_path, ':'.join(jars), classname, props)
        if VERBOSE: print start_corenlp
        self.corenlp = pexpect.spawn(start_corenlp)
        
        # show progress bar while loading the models
        widgets = ['Loading Models: ', Fraction()]
        pbar = ProgressBar(widgets=widgets, maxval=4, force_update=True).start()
        self.corenlp.expect("done.", timeout=20) # Load pos tagger model (~5sec)
        pbar.update(1)
        self.corenlp.expect("done.", timeout=200) # Load NER-all classifier (~33sec)
        pbar.update(2)
        self.corenlp.expect("done.", timeout=600) # Load NER-muc classifier (~60sec)
        pbar.update(3)
        self.corenlp.expect("done.", timeout=600) # Load CoNLL classifier (~50sec)
        pbar.update(4)
#        self.corenlp.expect("done.", timeout=200) # Loading PCFG (~3sec)
#        pbar.update(5)
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
        max_expected_time = min(20, 3 + len(text) / 20.0)
        end_time = time.time() + max_expected_time
        
        incoming = ""
        while True:
            # Time left, read more data
            try:
                incoming += self.corenlp.read_nonblocking(40000, 1800)
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

    '''
    def sep_depparsing(self,sent_filename):
        """
        separate dependency parser
        """

        jars = ["stanford-parser-3.3.1-models.jar",
                "stanford-parser.jar"]
       
        # if CoreNLP libraries are in a different directory,
        # change the corenlp_path variable to point to them
        corenlp_path = os.path.relpath(__file__).split('/')[0]+"/stanford-parser/"
        
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
    '''

    def parse(self, sent_filename):
        """ 
        This function takes a text string, sends it to the Stanford CoreNLP,
        reads in the result, parses the results and returns a list
        of data instances for each parsed sentence. Dependency parsing may operate 
        seperately for easy changing dependency parser.
        """
        
        instances = []
        prp_filename = sent_filename+'.prp' # preprocessed file
        if os.path.exists(prp_filename):

            prp_result = open(prp_filename,'r').read()
            #i = 0
            for i, result in enumerate(prp_result.split('-'*40)[1:]):
                #result_list = [line for line in result.split('\r\n') if line != '']
                #if i == 862:
                #    import pdb
                #    pdb.set_trace()
                #i += 1
                
                # fix additional newline bug for old version of stanford nlp
                #if len(result_list) == 6:
                #    result_list[3] = result_list[3] + result_list[4]
                #    result_list[4] = result_list[5]
                #    result_list.pop()
                #    result = '\r\n'.join(result_list)
                if i > 0 and i % 100 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
        
                try:
                    data = parse_parser_results(result)
                except Exception, e:
                    if VERBOSE: print traceback.format_exc()
                    raise e
                if isinstance(data, list):
                    instances += data
                else:
                    instances.append(data)
                #if len(instances) == 4291:
                #    import pdb
                #    pdb.set_trace()
            sys.stdout.write('\n')

        else:
            output_prp = open(prp_filename,'w')          

            for i,line in enumerate(open(sent_filename,'r').readlines()):
                result = self._parse(line)
                output_prp.write("%s\n%s"%('-'*40,result))
                try:
                    data = parse_parser_results_new(result)
                except Exception, e:
                    if VERBOSE: print traceback.format_exc()
                    raise e
                if isinstance(data, list):
                    instances += data
                else:
                    instances.append(data)
            output_prp.close()

        '''
        if seq_depparsing:
            dep_filename = sent_filename.rsplit('_',1)[0]+'_dep.txt'
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
         '''   
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
