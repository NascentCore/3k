# IB

InfiniBand verifications

## ib_read_bw

```
ssh -f <remote_ip> ib_read_bw
ib_read_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

ssh -f <remote_ip> ib_write_bw
ib_write_bw <remote_ip> -d <local_rdma_outlet> --report_gbits --run_infinitely

# RDMA outlets
mst status -v
```
