/*
Copyright 2023.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"sync"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
)

type JupyterLabOption struct {
	Image            string
	StorageClassName string
	OssOption        OssOption
}

// JupyterLabReconciler reconciles a JupyterLab object
type JupyterLabReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	Option *JupyterLabOption
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=jupyterlabs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=jupyterlabs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=jupyterlabs/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the JupyterLab object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.17.0/pkg/reconcile
func (r *JupyterLabReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	jupyterlab := &cpodv1beta1.JupyterLab{}
	if err := r.Client.Get(ctx, req.NamespacedName, jupyterlab); client.IgnoreNotFound(err) != nil {
		return ctrl.Result{}, err
	}
	oldjuypyterlabStatus := jupyterlab.DeepCopy().Status

	defer func() {
		if !equality.Semantic.DeepEqual(oldjuypyterlabStatus, &jupyterlab.Status) {
			if err := r.Client.Status().Update(ctx, jupyterlab); err != nil {
				logger.Error(err, "unable to update jupyter status")
				reterr = err
			}
		}
	}()

	if !jupyterlab.Status.DataReady {
		if err := r.prepareData(ctx, jupyterlab); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to prepare data: %w", err)
		}
		jupyterlab.Status.DataReady = true
	}

	sts := &appsv1.StatefulSet{}
	if err := r.Client.Get(ctx, req.NamespacedName, sts); err != nil {
		if apierrors.IsNotFound(err) {

			if err := r.createService(ctx, jupyterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create service: %w", err)
			}

			if err := r.createIngress(ctx, jupyterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create ingress: %w", err)
			}
			if err := r.createSts(ctx, jupyterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create sts: %w", err)
			}

			jupyterlab.Status.Phase = cpodv1beta1.JupyterLabJobPhasePending
			// TODO: update status
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, err
	}

	currentPhase := cpodv1beta1.JupyterLabJobPhasePending
	if sts.Status.AvailableReplicas == 1 {
		currentPhase = cpodv1beta1.JupyterLabJobPhaseRunning
	}
	jupyterlab.Status.Phase = currentPhase

	return ctrl.Result{}, nil
}

func (r *JupyterLabReconciler) prepareData(ctx context.Context, jupyterlab *v1beta1.JupyterLab) error {
	wg := sync.WaitGroup{}
	errChan := make(chan error, len(jupyterlab.Spec.Models)+len(jupyterlab.Spec.Datasets))
	for _, model := range jupyterlab.Spec.Models {
		wg.Add(1)
		go func(model v1beta1.Model) {
			defer wg.Done()
			if err := r.prepareModel(ctx, *jupyterlab, model); err != nil {
				errChan <- err
				return
			}
		}(model)
	}

	for _, dataset := range jupyterlab.Spec.Datasets {
		wg.Add(1)
		go func(dataset v1beta1.Dataset) {
			defer wg.Done()
			if err := r.prepareDataset(ctx, jupyterlab, dataset); err != nil {
				errChan <- err
				return
			}
		}(dataset)

	}

	if len(errChan) > 0 {
		for err := range errChan {
			if err != nil {
				return err
			}
		}
	}
	return nil

}

func (r *JupyterLabReconciler) prepareModel(ctx context.Context, juypterlab v1beta1.JupyterLab, model v1beta1.Model) error {
	modelSize := int64(model.ModelSize)
	modelReadableName := model.Name
	modelTemplate := model.Template

	if model.ModelIspublic {
		modelName := model.ModelStorage + v1beta1.CPodPublicStorageSuffix
		ms := &cpodv1.ModelStorage{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: juypterlab.Namespace, Name: modelName}, ms); err != nil {
			if apierrors.IsNotFound(err) {
				publicMs := &cpodv1.ModelStorage{}
				if err := r.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: model.ModelStorage}, publicMs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdMs, err := createModelstorage(ctx, r.Client, model.ModelStorage, modelReadableName, modelSize, modelTemplate, v1beta1.CPodPublicNamespace, r.Option.StorageClassName); err != nil {
							return fmt.Errorf("failed to create model storage for public model %s: %v", model.ModelStorage, err)
						} else {
							publicMs = createdMs
						}
					} else {
						return fmt.Errorf("failed to get public model %s: %v", model.ModelStorage, err)
					}
				}
				if publicMs != nil && publicMs.Status.Phase != "done" {
					jobName := "model-" + model.ModelStorage
					job := &batchv1.Job{}
					if err := r.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, r.Client, r.Option.OssOption, "model", model.ModelStorage, modelReadableName, modelSize, juypterlab.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public model %s: %v", model.ModelStorage, err)
							}
						} else {
							return fmt.Errorf("failed to get public model %s: %v", model.ModelStorage, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public model downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public model %s is not done", model.ModelStorage)
				}
				if err := CopyPublicModelStorage(ctx, r.Client, model.ModelStorage, juypterlab.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", model.ModelStorage, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public model %v's copy  %s: %v", model.ModelStorage, modelName, err)
			}
		}
		if ms.Status.Phase != "done" {
			return fmt.Errorf("public model copy  %s is not done", model.ModelStorage)
		}
		return nil
	}
	ms := &cpodv1.ModelStorage{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: juypterlab.Namespace, Name: model.ModelStorage}, ms); err != nil {
		if apierrors.IsNotFound(err) {
			if createdMs, err := createModelstorage(ctx, r.Client, model.ModelStorage, modelReadableName, modelSize, modelTemplate, juypterlab.Namespace, r.Option.StorageClassName); err != nil {
				return fmt.Errorf("failed to create model storage for private model %s: %v", model.ModelStorage, err)
			} else {
				ms = createdMs
			}
		} else {
			return fmt.Errorf("failed to get private model %s: %v", model.ModelStorage, err)
		}
	}
	if ms != nil && ms.Status.Phase != "done" {
		jobName := "model-" + model.ModelStorage
		job := &batchv1.Job{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: juypterlab.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, r.Client, r.Option.OssOption, "model", model.ModelStorage, modelReadableName, modelSize, juypterlab.Namespace, juypterlab.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private model %s: %v", model.ModelStorage, err)
				}
			} else {
				return fmt.Errorf("failed to get private model %s: %v", model.ModelStorage, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("model downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private model %s is not done", model.ModelStorage)
	}
	return nil
}

func (r *JupyterLabReconciler) prepareDataset(ctx context.Context, jupyterlab *v1beta1.JupyterLab, dataset v1beta1.Dataset) error {
	logger := log.FromContext(ctx)
	datasetSize := int64(dataset.DatasetSize)
	datasetReadableName := dataset.Name
	if dataset.DatasetIspublic {
		dsName := dataset.DatasetStorage + v1beta1.CPodPublicStorageSuffix
		ds := &cpodv1.DataSetStorage{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: jupyterlab.Namespace, Name: dsName}, ds); err != nil {
			if apierrors.IsNotFound(err) {
				logger.Info("public dataset copy not found, create it", "dataset", dataset.DatasetStorage)
				publicDs := &cpodv1.DataSetStorage{}
				if err := r.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: dataset.DatasetStorage}, publicDs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdDs, err := createDatasetStorage(ctx, r.Client, dataset.DatasetStorage, datasetReadableName, datasetSize, v1beta1.CPodPublicNamespace, r.Option.StorageClassName); err != nil {
							return fmt.Errorf("failed to create dataset storage for public model %s: %v", dataset.DatasetStorage, err)
						} else {
							publicDs = createdDs
						}
					} else {
						return fmt.Errorf("failed to get public dataset %s: %v", dataset.DatasetStorage, err)
					}
				}
				if publicDs != nil && publicDs.Status.Phase != "done" {
					jobName := "dataset-" + dataset.DatasetStorage
					job := &batchv1.Job{}
					if err := r.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, r.Client, r.Option.OssOption, "dataset", dataset.DatasetStorage, datasetReadableName, datasetSize, jupyterlab.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public dataset %s: %v", dataset.DatasetStorage, err)
							}
						} else {
							return fmt.Errorf("failed to get public dataset %s: %v", dataset.DatasetStorage, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public dataset downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public dataset %s is not done", dataset.DatasetStorage)
				}
				if err := CopyPublicDatasetStorage(ctx, r.Client, dataset.DatasetStorage, jupyterlab.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", dataset.DatasetStorage, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public dataset %v's copy  %s: %v", dataset.DatasetStorage, dsName, err)
			}
		}
		if ds.Status.Phase != "done" {
			return fmt.Errorf("public dataset copy  %s is not done", dataset.DatasetStorage)
		}
		return nil
	}
	ds := &cpodv1.DataSetStorage{}
	if err := r.Client.Get(ctx, client.ObjectKey{Namespace: jupyterlab.Namespace, Name: dataset.DatasetStorage}, ds); err != nil {
		if apierrors.IsNotFound(err) {
			if createdDs, err := createDatasetStorage(ctx, r.Client, dataset.DatasetStorage, datasetReadableName, datasetSize, jupyterlab.Namespace, r.Option.StorageClassName); err != nil {
				return fmt.Errorf("failed to create dataset storage for private dataset %s: %v", dataset.DatasetStorage, err)
			} else {
				ds = createdDs
			}
		} else {
			return fmt.Errorf("failed to get private dataset %s: %v", dataset.DatasetStorage, err)
		}
	}
	if ds != nil && ds.Status.Phase != "done" {
		jobName := "dataset-" + dataset.DatasetStorage
		job := &batchv1.Job{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: jupyterlab.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, r.Client, r.Option.OssOption, "dataset", dataset.DatasetStorage, datasetReadableName, datasetSize, jupyterlab.Namespace, jupyterlab.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private dataset %s: %v", dataset.DatasetStorage, err)
				}
			} else {
				return fmt.Errorf("failed to get private dataset %s: %v", dataset.DatasetStorage, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("dataset downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private dataset %s is not done", dataset.DatasetStorage)
	}
	return nil
}

func (r *JupyterLabReconciler) createSts(ctx context.Context, jupyterlab *cpodv1beta1.JupyterLab) error {

	image := jupyterlab.Spec.Image
	if image == "" {
		image = r.Option.Image
	}
	workerSpacePVCName := "pvc-" + jupyterlab.Name + "-0"

	volmeMounts := []corev1.VolumeMount{
		{
			Name:      "workspace",
			MountPath: "/workspace",
		},
	}

	volumes := []corev1.Volume{
		{
			Name: "workspace",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: workerSpacePVCName,
				},
			},
		},
	}

	for _, model := range jupyterlab.Spec.Models {
		modelstorage := &cpodv1.ModelStorage{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: jupyterlab.Namespace, Name: model.ModelStorage}, modelstorage); err != nil {
			return fmt.Errorf("failed to get model storage %v: %w", model.ModelStorage, err)
		}

		volmeMounts = append(volmeMounts, corev1.VolumeMount{
			Name:      model.ModelStorage,
			MountPath: model.MountPath,
		})
		volumes = append(volumes, corev1.Volume{
			Name: model.ModelStorage,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: modelstorage.Spec.PVC,
				},
			},
		})
	}

	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jupyterlab.Name,
			Namespace: jupyterlab.Namespace,
			Labels:    map[string]string{"app": "jupyterlab"},
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, jupyterlab),
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": jupyterlab.Name},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": jupyterlab.Name},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "juypterlab",
							Image: image,
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(jupyterlab.Spec.Memory),
									corev1.ResourceCPU:    resource.MustParse(jupyterlab.Spec.CPUCount),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(jupyterlab.Spec.Memory),
									corev1.ResourceCPU:    resource.MustParse(jupyterlab.Spec.CPUCount),
									"nvidia.com/gpu":      *resource.NewQuantity(int64(jupyterlab.Spec.GPUCount), resource.DecimalSI),
								},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "JUPYTER_TOKEN",
									Value: jupyterlab.Name,
								},
							},
							Command: []string{
								"jupyter",
								"lab",
								fmt.Sprintf("--ServerApp.base_url=/jupyterlab/%s/", jupyterlab.Name),
								"--allow-root",
								"--ip=0.0.0.0",
							},
							VolumeMounts: volmeMounts,
						},
					},
					Volumes: volumes,
				},
			},
			VolumeClaimTemplates: []corev1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pvc",
					},
					Spec: v1.PersistentVolumeClaimSpec{
						AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
						StorageClassName: &r.Option.StorageClassName,
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceStorage: resource.MustParse(jupyterlab.Spec.DataVolumeSize),
							},
						},
					},
				},
			},
		},
	}

	if jupyterlab.Spec.GPUProduct != "" {
		sts.Spec.Template.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.product": jupyterlab.Spec.GPUProduct}
	}
	if err := r.Client.Create(ctx, sts); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (r *JupyterLabReconciler) createService(ctx context.Context, jupyterlab *cpodv1beta1.JupyterLab) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jupyterlab.Name + "-svc",
			Namespace: jupyterlab.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, jupyterlab),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": jupyterlab.Name},
			Ports: []corev1.ServicePort{
				{
					Port:       8888,
					TargetPort: intstr.FromInt(8888),
				},
			},
			Type: corev1.ServiceTypeClusterIP,
		},
	}
	if err := r.Client.Create(ctx, svc); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (r *JupyterLabReconciler) createIngress(ctx context.Context, jupyterlab *cpodv1beta1.JupyterLab) error {
	pathType := networkingv1.PathTypeImplementationSpecific
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jupyterlab.Name + "-ing",
			Namespace: jupyterlab.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, jupyterlab),
			},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/jupyterlab/" + jupyterlab.Name + "/$2",
			},
		},
		Spec: networkingv1.IngressSpec{
			IngressClassName: ptr.To("nginx"),
			Rules: []networkingv1.IngressRule{
				{
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									PathType: &pathType,
									Path:     "/jupyterlab/" + jupyterlab.Name + "(/|$)(.*)",
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: jupyterlab.Name + "-svc",
											Port: networkingv1.ServiceBackendPort{
												Number: 8888,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if err := r.Client.Create(ctx, ingress); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (r *JupyterLabReconciler) generateOwnerRefJuypterLab(ctx context.Context, jupyterlab *cpodv1beta1.JupyterLab) metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion:         jupyterlab.APIVersion,
		Kind:               jupyterlab.Kind,
		Name:               jupyterlab.Name,
		UID:                jupyterlab.UID,
		BlockOwnerDeletion: ptr.To(true),
		Controller:         ptr.To(true),
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *JupyterLabReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.JupyterLab{}).
		Complete(r)
}
