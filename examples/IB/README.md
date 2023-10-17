# IB

InfiniBand verifications

## ib_read_bw

```
ssh -f <remote_ip> ib_{read/write}_bw
ib_{read/write}_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

# RDMA outlets
mst status -v
```
