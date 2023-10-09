# build
## create serviceaccount and clusterrole
kbuectl apply -f cpodmanager.yaml -n cpod

## the pod cpodmanager runs in , should be create like this :
apiVersion: v1
kind: Pod
metadata:
  name: xxxxxxx
  namespace: cpod
spec:
  serviceAccountName: sa-cpodmanager
  ......