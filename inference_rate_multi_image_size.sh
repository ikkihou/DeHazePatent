#! /usr/bin/env bash

OUTPUT=out.txt

echo '' >$OUTPUT

for i in 128 256 512 1024; do
    echo $i >>$OUTPUT
    python infer.py -c ./config/framework_da.json -s $i | grep "avg_infer_time" | awk '{print $10}' >>$OUTPUT
done
