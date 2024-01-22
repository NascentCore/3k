# portal synch
1. synch cpodjobs with portal
2. upload cpod info to portal

## depends on
configmap named " cpod-info "
it should be created like
```yaml
apiVersion: v1
data:
  access_key: xxx
  cpod_id: xxx
kind: ConfigMap
metadata:
  name: cpod-info
  namespace: cpodjob-system
```