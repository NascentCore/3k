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
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
)

type JupyterLabOption struct {
	Image            string
	StorageClassName string
}

// JuypterLabReconciler reconciles a JuypterLab object
type JuypterLabReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	Option *JupyterLabOption
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=juypterlabs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=juypterlabs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=juypterlabs/finalizers,verbs=update
//+kubebuilder:rbac:groups="apps",resources=statfulsets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="apps",resources=statfulsets/status,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the JuypterLab object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.17.0/pkg/reconcile

func (r *JuypterLabReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	juypterlab := &cpodv1beta1.JuypterLab{}
	if err := r.Client.Get(ctx, req.NamespacedName, juypterlab); client.IgnoreNotFound(err) != nil {
		return ctrl.Result{}, err
	}
	oldjuypyterlabStatus := juypterlab.DeepCopy().Status

	defer func() {
		if !equality.Semantic.DeepEqual(oldjuypyterlabStatus, &juypterlab.Status) {
			if err := r.Client.Status().Update(ctx, juypterlab); err != nil {
				logger.Error(err, "unable to update jupyter status")
				reterr = err
			}
		}
	}()

	sts := &appsv1.StatefulSet{}
	if err := r.Client.Get(ctx, req.NamespacedName, sts); err != nil {
		if apierrors.IsNotFound(err) {

			if err := r.createService(ctx, juypterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create service: %w", err)
			}

			if err := r.createIngress(ctx, juypterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create ingress: %w", err)
			}
			if err := r.createSts(ctx, juypterlab); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to create sts: %w", err)
			}

			juypterlab.Status.Phase = cpodv1beta1.JupyterLabJobPhasePending
			// TODO: update status
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{}, err
	}

	currentPhase := cpodv1beta1.JupyterLabJobPhasePending
	if sts.Status.AvailableReplicas == 1 {
		currentPhase = cpodv1beta1.JupyterLabJobPhaseRunning
	}
	juypterlab.Status.Phase = currentPhase

	return ctrl.Result{}, nil
}

func (r *JuypterLabReconciler) createSts(ctx context.Context, juypterlab *cpodv1beta1.JuypterLab) error {

	image := juypterlab.Spec.Image
	if image == "" {
		image = r.Option.Image
	}
	workerSpacePVCName := "pvc-" + juypterlab.Name + "-0"

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

	for _, model := range juypterlab.Spec.Models {
		modelstorage := &cpodv1.ModelStorage{}
		if err := r.Client.Get(ctx, client.ObjectKey{Namespace: juypterlab.Namespace, Name: model.ModelStorage}, modelstorage); err != nil {
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
			Name:      juypterlab.Name,
			Namespace: juypterlab.Namespace,
			Labels:    map[string]string{"app": "jupyterlab"},
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, juypterlab),
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": juypterlab.Name},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": juypterlab.Name},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "juypterlab",
							Image: image,
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(juypterlab.Spec.Memory),
									corev1.ResourceCPU:    resource.MustParse(juypterlab.Spec.CPUCount),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceMemory: resource.MustParse(juypterlab.Spec.Memory),
									corev1.ResourceCPU:    resource.MustParse(juypterlab.Spec.CPUCount),
									"nvidia.com/gpu":      *resource.NewQuantity(int64(juypterlab.Spec.GPUCount), resource.DecimalSI),
								},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "JUPYTER_TOKEN",
									Value: juypterlab.Name,
								},
							},
							Command: []string{
								"jupyter",
								"lab",
								fmt.Sprintf("--ServerApp.base_url=/jupyterlab/%s/", juypterlab.Name),
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
								v1.ResourceStorage: resource.MustParse(juypterlab.Spec.DataVolumeSize),
							},
						},
					},
				},
			},
		},
	}

	if juypterlab.Spec.GPUProduct != "" {
		sts.Spec.Template.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.product": juypterlab.Spec.GPUProduct}
	}
	if err := r.Client.Create(ctx, sts); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func (r *JuypterLabReconciler) createService(ctx context.Context, juypterlab *cpodv1beta1.JuypterLab) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      juypterlab.Name + "-svc",
			Namespace: juypterlab.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, juypterlab),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": juypterlab.Name},
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

func (r *JuypterLabReconciler) createIngress(ctx context.Context, juypterlab *cpodv1beta1.JuypterLab) error {
	pathType := networkingv1.PathTypeImplementationSpecific
	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      juypterlab.Name + "-ing",
			Namespace: juypterlab.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				r.generateOwnerRefJuypterLab(ctx, juypterlab),
			},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/jupyterlab/" + juypterlab.Name + "/$2",
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
									Path:     "/jupyterlab/" + juypterlab.Name + "(/|$)(.*)",
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: juypterlab.Name + "-svc",
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

func (r *JuypterLabReconciler) generateOwnerRefJuypterLab(ctx context.Context, juypterlab *cpodv1beta1.JuypterLab) metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion:         juypterlab.APIVersion,
		Kind:               juypterlab.Kind,
		Name:               juypterlab.Name,
		UID:                juypterlab.UID,
		BlockOwnerDeletion: ptr.To(true),
		Controller:         ptr.To(true),
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *JuypterLabReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.JuypterLab{}).
		Complete(r)
}
