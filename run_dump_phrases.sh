for (( start=0; start < 5800; start+=200))
do
    make dump-large MODEL_NAME=dph-nqsqd-pb2 START=$((start)) END=$((start+200))
done
make dump-large MODEL_NAME=dph-nqsqd-pb2 START=5800 END=5903
