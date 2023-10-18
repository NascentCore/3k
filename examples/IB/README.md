# IB

InfiniBand verifications

## ib_{read,write}_bw

```
ssh -f <remote_ip> ib_read_bw
ib_read_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

ssh -f <remote_ip> ib_write_bw
ib_write_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

# RDMA outlets
mst status -v
```

## ibping

Node A 
```
ibstat 
从ibstat的输出中任意选择一个CA以其Port，记录其Base Lid

CA和Port是上面选出的
ibping -S -C <CA> -P <Port>
```

Node B
```
ibping -L <前面记录的Lid>
```
