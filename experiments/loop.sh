PY=wide_deep_mse_train.py
txt=out.txt
times=new_times.txt
echo "\n" > $txt
echo "\n" > $times

for i in 1 2 3 4 5
do
  start_time=$(date +%s)
  echo "*********iteration $i ***********" >> $times
  CUDA_VISIBLE_DEVICES=0 python3 $PY --epochs 100 --eps 0 --cuda True >> $txt
  elapsed=$(($(date +%s) - start_time))
  echo "elapsed: " >> $times
  echo $elapsed >> $times
done


