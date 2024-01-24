package controller

import (
	"context"
	"fmt"
	"strings"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"knative.dev/pkg/apis"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// CPodJobReconciler reconciles a CPodJob object
type InferenceReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/finalizers,verbs=update
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices/status,verbs=get;update;patch

func (i *InferenceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	inference := &cpodv1beta1.Inference{}
	if err := i.Get(ctx, req.NamespacedName, inference); client.IgnoreNotFound(err) != nil {
		logger.Error(err, "unable to fetch Inference")
		return ctrl.Result{}, err
	}

	inferenceDeepcopy := inference.DeepCopy()

	inferenceService := &kservev1beta1.InferenceService{}
	if err := i.Get(ctx, types.NamespacedName{Namespace: inference.Namespace, Name: i.getInferenceServiceName(inference)}, inferenceService); err != nil {
		if apierrors.IsNotFound(err) {
			createErr := i.CreateBaseInferenceServices(ctx, inference)
			if createErr != nil {

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
	inferenceDeepcopy.Status.URL = inferenceServiceURL(inferenceService.Status)

	return ctrl.Result{}, nil
}

func (i InferenceReconciler) getInferenceServiceName(inference *cpodv1beta1.Inference) string {
	return inference.Name + "-is"
}

func (i *InferenceReconciler) CreateBaseInferenceServices(ctx context.Context, inference *cpodv1beta1.Inference) error {

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
					return nil
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
