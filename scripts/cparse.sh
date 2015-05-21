STANFORD_PATH="/home/j/llc/cwang24/R_D/AMRParsing/stanfordnlp/stanford-parser"

java -Xmx1800m -cp $STANFORD_PATH/stanford-parser-3.3.1-models.jar:$STANFORD_PATH/stanford-parser.jar edu.stanford.nlp.parser.lexparser.LexicalizedParser -tokenized -sentences newline edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $1