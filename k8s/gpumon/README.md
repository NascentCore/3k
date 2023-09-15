# GPUMon

Tools and systems for monitoring GPUs.

* https://github.com/NVIDIA/go-nvml
* https://github.com/NVIDIA/go-dcgm
* https://github.com/NVIDIA/dcgm-exporter

#Usage
##deloy kube-prometheus-stack if needed
helm install -n gpumon promethues-grafana prometheus-community/kube-prometheus-stack --set kubeStateMetrics.enabled=false --set nodeExporter.enabled=false --set grafana.enabled=true --set prometheusOperator.admissionWebhooks.patch.image.registry=registry.aliyuncs.com  --set prometheusOperator.admissionWebhooks.patch.image.repository=google_containers/kube-webhook-certgen 

##deploy DCGM-exporter
kubectl apply -f dcgm_exporter.yaml

##create a ServiceMonitor(a crd of kube-prometheus-stack) kind resource 
kubectl apply -f promethues-gpumonitor.yaml

