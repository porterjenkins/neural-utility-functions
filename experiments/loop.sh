PY=dfm_logit_train.py
txt=out.txt
echo "\n" > $txt
for i in {1..5}
do
  echo "*********iteration $i ***********\n" >> $txt
  python3 $PY >> $txt
done

python3 csv_constructor.py --file $txt


