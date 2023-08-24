# CIFAR-10 image

This builds the CIFAR-10 image.
```
docker build . -t cifar
docker tag cifar <cloud-repository>:<tag>
docker push <cloud-repository>:<tag>
```
