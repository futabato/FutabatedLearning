#!/bin/bash

# クライアントの数
NUM_CLIENTS=2

for ((i=0; i<$NUM_CLIENTS; i++))
do
   echo "Starting client $i"
   python client_producer.py &
done

# 全てのクライアントが終了するのを待つ
wait
echo "All clients finished."
