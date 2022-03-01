x=$1
python $( dirname $0 )/get_hypothesis.py $x $x.hyp
python $( dirname $0 )/gpt2_decode.py $x.hyp $x.text
