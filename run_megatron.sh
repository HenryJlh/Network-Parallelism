# set the environment variables.
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LOCAL_RANK=0

# torchrun the megatron
# change the node_rank to 0-3. (same as local rank)
# nproc_per_node: number of gpus per node.
# master_addr: generally node1's internal ip.
# master_port: randomly set.
# tensor-model-parallel-size, pipeline-model-parallel-size, explicitly set the parallelism.
# data-model-parallel-size, implicitly set by the code. (DP = num_gpus/TP/PP) 
torchrun --nproc_per_node 1 \
    --nnodes 4 \
    --node_rank 0\
    --master_addr 10.10.1.1\
    --master_port 23456\
   /data/workspace/megatron-lm/pretrain_gpt.py \
    --tensor-model-parallel-size 1\
    --pipeline-model-parallel-size 1\
    --num-layers 12\
    --hidden-size 512\
    --num-attention-heads 8\
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 100\
    --lr-decay-iters 320000 \
    --data-path ./data/meg-gpt2_text_document \
    --vocab-file ./data/gpt2-vocab.json \
    --merge-file ./data/gpt2-merges.txt \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --recompute-activations \
    --log-interval 10 \
    --save-interval 1 \
    --eval-interval 1 \
    --eval-iters 20\
    --fp16\
>> 100mbit_1tensor_1pipe_4data_ring_vlan.txt

#============== Optional ====================
# use "ifconfig", you can find the connection socket between nodes. (e.g. vlan1102 10.10.1.1,10.10.1.2)
ifconfig

# use "iperf3", you can test the bandwidth between nodes.(e.g. measure the bandwidth between node1 and node2)
# assume node1 and node2 are connected by 10.10.1.x (ifconfig, you can find node1 is 10.10.1.1, node2 is 10.10.1.2)
iperf3 -s
iperf3 -c 10.10.1.1

# use "tc", you can force the bandwidth. (e.g. set vlan1102 100mbit/s)
tc qdisc add dev vlan1102 root handle 1: htb default 11
tc class add dev vlan1102 parent 1: classid 1:1 htb rate 100mbit
tc class add dev vlan1102 parent 1:1 classid 1:11 htb rate 100mbit
tc qdisc show dev vlan1102
# delete one tc setting
tc qdisc del dev vlan1102 root
# after you set the bandwidth, you can enforce the socket used for communication by setting env-var.
export NCCL_SOCKET_IFNAME=vlan1102,vlan1126,vlan1125,vlan1122 

