#PY=wide_deep_mse_train.py
#txt=out.txt
#times=new_times.txt
#echo "\n" > $txt
#echo "\n" > $times
#
#for i in 1 2 3 4 5
#do
#  start_time=$(date +%s)
#  echo "*********iteration $i ***********" >> $times
#  CUDA_VISIBLE_DEVICES=0 python3 $PY --epochs 100 --eps 0 --cuda True >> $txt
#  elapsed=$(($(date +%s) - start_time))
#  echo "elapsed: " >> $times
#  echo $elapsed >> $times
#done
#
#

ogtxt=og_out.txt
newtxt=new_out.txt
ogtimes=og_times.txt
newtimes=new_times.txt
extension=.py
og=_original
all=all_comparisons.txt
echo "" > $all
for experiment in wide_deep_mse_train #wide_deep_logit_train mlp_mse_train mlp_logit_train mf_mse_train mf_logit_train dfm_mse_train dfm_logit_train
do
  echo "\n" > $ogtxt
  echo "\n" > $newtxt
  echo "" > $ogtimes
  echo "" > $newtimes
  for i in 2 1
  do
    if [ $i -gt 1 ]
    then
      for j in 1 2 3 4 5
      do
        PY=$experiment$extension
        start_time=$(date +%s)
#        echo "*********iteration $j ***********" >> $newtimes
        CUDA_VISIBLE_DEVICES=0 python3 $PY --epochs 10 --batch_size 512 --eps 0 --num_workers 8 --cuda True 2>/dev/null
        elapsed=$(($(date +%s) - start_time))
#        echo "elapsed: " >> $newtimes
        echo $elapsed >> $newtimes
      done
    else
      for j in 1 2 3 4 5
      do
      PY=$experiment$og$extension
      start_time=$(date +%s)
#        echo "*********iteration $j ***********" >> $ogtimes
        CUDA_VISIBLE_DEVICES=0 python3 $PY --epochs 10  --batch_size 512 --eps 0 --cuda True 2>/dev/null
        elapsed=$(($(date +%s) - start_time))
#        echo "elapsed: " >> $ogtimes
        echo $elapsed >> $ogtimes
      done
    fi
  done
  python3 compare_times.py --original $ogtimes --new $newtimes --experiment $experiment >> $all
done