# README

To launch Bert training MPI Job, run below command:

```
kubectl apply -f mpi_bert_ds.yaml -n bert-training
# Check pods are created
kubectl get pods -n bert-training
```

This will create a MPIJob in the `bert-training` namespace.
