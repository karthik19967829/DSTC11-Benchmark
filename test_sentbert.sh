#


# from scipy.spatial.distance import cdist
# --needs to access the actual lib; not sure what goes wrong with container
export LD_LIBRARY_PATH=/home/air/anaconda3/lib

dataset="fed-turn"
processor=cuda  # cpu | cuda

scoring=wor  # wr | wor | dial
# wr: reference-based; turn-level
# wor: reference-free; turn-level
# dial: global dialog rating

eval_type=zh
if [ $# -eq 0 ]; then echo "$0: -d <dataset> -p <processor> -s <scoring> -e <eval_type>"; exit 2 ; fi

SHORT=d:,p:,s:,e:,h
LONG=dataset:,processor:,scoring:,eval_type:,help
OPTS=$(getopt --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"
while [ : ]; do
    case "$1" in
	-d | --dataset )
	    dataset=$2
	    shift 2
	    ;;
	-p | --processor )
	    processor=$2
	    shift 2
	    ;;
	-s | scoring )
	    scoring=$2
	    shift 2
	    ;;
	-e | eval_type)
	    eval_type=$2
	    shift 2
	    ;;       
	-- )
	    shift;
	    break
	    ;;
	* | ? )
	    echo "unexpected option: $1 "
	    exit 1
	    ;;
    esac
done

if [[ $? -ne 0 ]]; then  exit 1; fi

# coordinates
echo "dataset:    ${dataset}"
echo "processor:  ${processor}"
echo "scoring:    ${scoring}"
echo "eval_type": ${eval_type}
python --version
echo ""
# debug="-m pdb"

# score selected system
python ${debug} compute_sent_${scoring}.py \
       \
       --dataset=${dataset} \
       --device=${processor} \
       --eval_type=${eval_type}
       --am_model_path=finetuned_lm/embedding_models/full_am \
       --fm_model_path=finetuned_lm/language_models/full_fm # \
#   | grep 'annotations.Overall' > "${dataset}_${scoring}__results.score"
#cat  "${dataset}_${scoring}__results.score}"  # show the result on screen
echo "---------------------"


#

