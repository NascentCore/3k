# IB

```
numactl -N 0 ib_read_bw -F 1.1.1.177 -d mlx5_0 -b --report_gbits --run_infinitely
cat << EOF > /etc/modules-load.d/custom.conf
nvidia_peermem
ib_umad
rdma_ucm
rdma_cm
ib_core
EOF
```
硬件、交换机、物理拓扑设计不合理、驱动固件版本、抖动、内核模块
