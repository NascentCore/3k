package controller

import (
	"context"
	"fmt"
	"strings"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	netv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/pointer"
	"knative.dev/pkg/apis"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type InferenceOptions struct {
	// IngressClass string
	Domain              string
	InferenceWebuiImage string
}

// CPodJobReconciler reconciles a CPodJob object
type InferenceReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	Recorder record.EventRecorder

	Options InferenceOptions
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/finalizers,verbs=update
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="apps",resources=deployments,verbs=get;list;watch;create;update;patch;delete

func (i *InferenceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	inference := &cpodv1beta1.Inference{}
	if err := i.Get(ctx, req.NamespacedName, inference); client.IgnoreNotFound(err) != nil {
		logger.Error(err, "unable to fetch Inference")
		return ctrl.Result{}, err
	}

	if inference.DeletionTimestamp != nil {
		return ctrl.Result{}, nil
	}

	inferenceDeepcopy := inference.DeepCopy()

	inferenceService := &kservev1beta1.InferenceService{}
	if err := i.Get(ctx, types.NamespacedName{Namespace: inference.Namespace, Name: i.getInferenceServiceName(inference)}, inferenceService); err != nil {
		if apierrors.IsNotFound(err) {
			createErr := i.CreateBaseInferenceServices(ctx, inference)
			if createErr != nil {
				i.Recorder.Eventf(inference, corev1.EventTypeWarning, "CreateInferenceServiceFailed", createErr.Error())
				return ctrl.Result{}, createErr
			}

			// 推理服务创建成功后，部署Web UI
			webUIErr := i.DeployWebUI(ctx, inference)
			if webUIErr != nil {
				logger.Error(webUIErr, "unable to deploy Web UI")
				return ctrl.Result{}, webUIErr
			}

			// Web UI部署成功后，创建Ingress
			ingressErr := i.DeployWebUIIngress(ctx, inference)
			if ingressErr != nil {
				logger.Error(ingressErr, "unable to deploy Web UI Ingress")
				return ctrl.Result{}, ingressErr
			}

			return ctrl.Result{}, err
		}
		return ctrl.Result{}, err
	}
	defer func() {
		if !equality.Semantic.DeepEqual(inference.Status, inferenceDeepcopy.Status) {
			if err := i.Client.Status().Update(ctx, inferenceDeepcopy); err != nil {
				logger.Error(err, "unable to update CPodJob status")
				reterr = err
			}
		}
	}()

	inferenceDeepcopy.Status.Ready = inferenceServiceReadiness(inferenceService.Status)
	if !inferenceDeepcopy.Status.Ready {
		inferenceDeepcopy.Status.Conditions = inferenceService.Status.Conditions
	} else {
		inferenceDeepcopy.Status.Conditions = nil
		url, err := i.GetIngress(ctx, inference, inferenceService)
		if err != nil {
			return ctrl.Result{}, err
		}
		inferenceDeepcopy.Status.URL = &url
	}

	return ctrl.Result{}, nil
}

func (i InferenceReconciler) getInferenceServiceName(inference *cpodv1beta1.Inference) string {
	return inference.Name + "-is"
}

func (i *InferenceReconciler) CreateBaseInferenceServices(ctx context.Context, inference *cpodv1beta1.Inference) error {
	if len(inference.Spec.Predictor.GetImplementations()) == 0 {
		return fmt.Errorf("the implementation of predictor is null")
	}
	predictor := inference.Spec.Predictor.GetImplementation()
	if sourceURI := predictor.GetStorageUri(); sourceURI != nil {
		if strings.HasPrefix(*sourceURI, cpodv1beta1.ModelStoragePrefix) {
			modelstorageName, err := parseModelStorageURI(*sourceURI)
			if err != nil {
				return err
			}
			modelstorage := cpodv1.ModelStorage{}
			if err := i.Client.Get(ctx, client.ObjectKey{
				Namespace: inference.Namespace,
				Name:      modelstorageName,
			}, &modelstorage); err != nil {
				if apierrors.IsNotFound(err) {
					// TODO: 更新condition
					i.Recorder.Eventf(inference, corev1.EventTypeWarning, "GetModelstorageFailed", "modelstorage not found")
					return err
				}
				return err
			}

			if modelstorage.Status.Phase != "done" {
				return fmt.Errorf("modelstorage %s is not ready", modelstorage.Name)
			}

			pre := cpodv1beta1.Predictor(inference.Spec.Predictor)
			pre.SetStorageURI("pvc://" + modelstorage.Spec.PVC)

		}
	}

	is := kservev1beta1.InferenceService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-is",
			Namespace: inference.Namespace,
			Labels:    inference.Labels,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: kservev1beta1.InferenceServiceSpec{
			Predictor: inference.Spec.Predictor,
		},
	}

	if err := i.Client.Create(ctx, &is); err != nil {
		return err
	}

	return nil
}

func (i *InferenceReconciler) generateOwnerRefInference(ctx context.Context, inference *cpodv1beta1.Inference) metav1.OwnerReference {
	yes := true
	return metav1.OwnerReference{
		APIVersion:         cpodv1beta1.GroupVersion.String(),
		Kind:               "Inference",
		Name:               inference.Name,
		UID:                inference.GetUID(),
		Controller:         &yes,
		BlockOwnerDeletion: &yes,
	}
}

func (i *InferenceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.Inference{}).
		Owns(&kservev1beta1.InferenceService{}).
		Complete(i)
}

func (i *InferenceReconciler) GetIngress(ctx context.Context, inference *cpodv1beta1.Inference, inferenceservice *kservev1beta1.InferenceService) (string, error) {
	ingressName := inference.Name + "-ingress"

	var ingress netv1.Ingress
	if err := i.Client.Get(ctx, types.NamespacedName{Namespace: inference.Namespace, Name: ingressName}, &ingress); err != nil {
		if apierrors.IsNotFound(err) {
			// 获取对应preidector service name
			svcName := PredictorServiceName(inferenceservice.Name)
			// create ingress
			path := "/"
			var rules []netv1.IngressRule
			rules = append(rules, i.generateRule(inference.Name, svcName, path, 80))
			ingress := &netv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ingressName,
					Namespace: inference.Namespace,
					Annotations: map[string]string{
						"kubernetes.io/ingress.class": "nginx",
					},
					OwnerReferences: []metav1.OwnerReference{
						i.generateOwnerRefInference(ctx, inference),
					},
				},
				Spec: netv1.IngressSpec{
					Rules: rules,
				},
			}
			if err := i.Client.Create(ctx, ingress); err != nil {
				ctrl.Log.Info("create ingress failed", "err", err)
				i.Recorder.Eventf(inference, corev1.EventTypeWarning, "CreateIngressFailed", "create ingress failed")
				return "", err
			}
			return "", nil
		}
		return "", err
	}

	return fmt.Sprintf("%v.%v", inference.Name, i.Options.Domain), nil
}

func PredictorServiceName(name string) string {
	return name + "-predictor"
}

// func DefaultPredictorServiceName(name string) string {
// 	return name + "-predictor-" + InferenceServiceDefault
// }

func (i *InferenceReconciler) generateRule(inferenceName, componentName string, path string, port int32) netv1.IngressRule {
	pathType := netv1.PathTypePrefix
	rule := netv1.IngressRule{
		Host: fmt.Sprintf("%v.%v", inferenceName, i.Options.Domain),
		IngressRuleValue: netv1.IngressRuleValue{
			HTTP: &netv1.HTTPIngressRuleValue{
				Paths: []netv1.HTTPIngressPath{
					{
						Path:     path,
						PathType: &pathType,
						Backend: netv1.IngressBackend{
							Service: &netv1.IngressServiceBackend{
								Name: componentName,
								Port: netv1.ServiceBackendPort{
									Number: port,
								},
							},
						},
					},
				},
			},
		},
	}
	return rule
}

func inferenceServiceReadiness(status kservev1beta1.InferenceServiceStatus) bool {
	return status.Conditions != nil &&
		status.GetCondition(apis.ConditionReady) != nil &&
		status.GetCondition(apis.ConditionReady).Status == corev1.ConditionTrue
}

func inferenceServiceURL(status kservev1beta1.InferenceServiceStatus) *string {
	if com, ok := status.Components[kservev1beta1.PredictorComponent]; ok {
		if com.URL != nil {
			URL := com.URL.String()
			return &URL
		}
	}
	return nil
}

func parseModelStorageURI(srcURI string) (modelStorageName string, err error) {
	parts := strings.Split(strings.TrimPrefix(srcURI, cpodv1beta1.ModelStoragePrefix), "/")
	if len(parts) != 1 {
		return "", fmt.Errorf("Invalid URI must be modelstorage://<modelstorage-name>: %s", srcURI)
	}
	return parts[0], nil
}

func (i *InferenceReconciler) DeployWebUI(ctx context.Context, inference *cpodv1beta1.Inference) error {
	webUISvc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-web-ui",
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app": inference.Name + "-web-ui",
			},
			Ports: []corev1.ServicePort{
				{
					Port:       8000,
					TargetPort: intstr.FromInt(8000),
				},
			},
		},
	}

	webUIDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-web-ui",
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: pointer.Int32Ptr(1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": inference.Name + "-web-ui",
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": inference.Name + "-web-ui",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  inference.Name + "-web-ui",
							Image: i.Options.InferenceWebuiImage,
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 8000,
								},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "API_URL",
									Value: fmt.Sprintf("http://%s.%s.svc.cluster.local/v1/chat/completions", PredictorServiceName(i.getInferenceServiceName(inference)), inference.Namespace),
								},
								{
									Name:  "ENV_BASE_URL",
									Value: fmt.Sprintf("/inference/%s/", inference.Name),
								},
							},
						},
					},
				},
			},
		},
	}

	if err := i.Client.Create(ctx, webUISvc); err != nil {
		return err
	}
	if err := i.Client.Create(ctx, webUIDeployment); err != nil {
		return err
	}

	return nil
}

func (i *InferenceReconciler) DeployWebUIIngress(ctx context.Context, inference *cpodv1beta1.Inference) error {
	pathType := netv1.PathTypeImplementationSpecific
	webUIIngressName := inference.Name + "-web-ui-ingress"
	webUISvcName := inference.Name + "-web-ui"
	webUIPort := int32(8000)

	ingress := &netv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      webUIIngressName,
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/$2",
			},
		},
		Spec: netv1.IngressSpec{
			IngressClassName: pointer.StringPtr("nginx"),
			Rules: []netv1.IngressRule{
				{
					IngressRuleValue: netv1.IngressRuleValue{
						HTTP: &netv1.HTTPIngressRuleValue{
							Paths: []netv1.HTTPIngressPath{
								{
									PathType: &pathType,
									Path:     "/inference/" + inference.Name + "(/|$)(.*)",
									Backend: netv1.IngressBackend{
										Service: &netv1.IngressServiceBackend{
											Name: webUISvcName,
											Port: netv1.ServiceBackendPort{
												Number: webUIPort,
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

	if err := i.Client.Create(ctx, ingress); err != nil {
		return err
	}

	return nil
}

// pointerTo returns a pointer to the provided PathType
func pointerTo(pt netv1.PathType) *netv1.PathType {
	return &pt
}
