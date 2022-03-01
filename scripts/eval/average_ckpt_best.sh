d=${1:-"checkpoints/"}
n=${2:-3}
ls $d/*pt -lht
ckpts=`ls $d/checkpoint*best_*pt -lht | tail -n $n | rev | cut -d " " -f1 | rev`
echo $ckpts
python $( dirname $0 )/average_checkpoints.py --inputs $ckpts --output $d/checkpoint_average_best-${n}.pt
