
DATA_PATH=$1
DATADIR="$( cd "$( dirname "${1}" )" && pwd )"
DIR="$( cd "$( dirname "${BASH_SOURCE[ 0]}" )" && pwd )"
CORENLP_PATH="/home/j/llc/cwang24/Tools/stanford-corenlp-full-2015-04-20"

java -Xmx25000m -cp "${CORENLP_PATH}/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props "${CORENLP_PATH}/default.properties" -file $DATA_PATH -outputDirectory $DATADIR

mv $DATA_PATH.xml $DATA_PATH.prp.xml
