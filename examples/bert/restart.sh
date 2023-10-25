k delete mpijob --all -n bert
k get pods -n bert
k apply -f k8s_config/mpi_bert_ds.yaml -n bert
