package controller

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// YAMLResourceReconciler reconciles a YAMLResource object
type YAMLResourceReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=yamlresources/finalizers,verbs=update
//+kubebuilder:rbac:groups=*,resources=*,verbs=get;list;watch;create;update;patch;delete

func (r *YAMLResourceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Starting Reconcile", "request", req)

	yamlResource := &cpodv1beta1.YAMLResource{}
	if err := r.Get(ctx, req.NamespacedName, yamlResource); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("YAMLResource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get YAMLResource")
		return ctrl.Result{}, err
	}

	logger.Info("YAMLResource found", "yamlResource", yamlResource)

	// 更新状态为 "Pending" 或 "Updating"
	if _, err := r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhasePending, "Processing YAML resources"); err != nil {
		logger.Error(err, "Failed to update status to Creating")
		return ctrl.Result{}, err
	}

	// 使用 yaml.NewYAMLOrJSONDecoder 来处理多个资源
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(yamlResource.Spec.YAML)), 4096)
	for {
		var rawObj runtime.RawExtension
		if err := decoder.Decode(&rawObj); err != nil {
			if err == io.EOF {
				break // 已经处理完所有资源
			}
			logger.Error(err, "Failed to decode YAML")
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to decode YAML: "+err.Error())
		}

		obj, _, err := unstructured.UnstructuredJSONScheme.Decode(rawObj.Raw, nil, nil)
		if err != nil {
			logger.Error(err, "Failed to decode raw object")
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, "Failed to decode raw object: "+err.Error())
		}

		unstructuredObj := obj.(*unstructured.Unstructured)

		// 设置资源的命名空间
		if unstructuredObj.GetNamespace() == "" {
			unstructuredObj.SetNamespace(yamlResource.Namespace)
		}

		// 添加标签
		// labels := unstructuredObj.GetLabels()
		// if labels == nil {
		// 	labels = make(map[string]string)
		// }
		// labels["app.kubernetes.io/name"] = yamlResource.Spec.AppName
		// labels["app.kubernetes.io/instance"] = yamlResource.Spec.AppID
		// labels["cpod.cpod/user-id"] = yamlResource.Spec.UserID
		// unstructuredObj.SetLabels(labels)

		// 添加 OwnerReferences
		ownerReference := metav1.OwnerReference{
			APIVersion:         yamlResource.APIVersion,
			Kind:               yamlResource.Kind,
			Name:               yamlResource.Name,
			UID:                yamlResource.UID,
			Controller:         ptr.To(true),
			BlockOwnerDeletion: ptr.To(true),
		}
		unstructuredObj.SetOwnerReferences([]metav1.OwnerReference{ownerReference})

		// 创建或更新资源
		created, err := r.createOrUpdateResource(ctx, unstructuredObj)
		if err != nil {
			logger.Error(err, "Failed to create or update resource", "resource", unstructuredObj.GetKind())
			return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseFailed, fmt.Sprintf("Failed to %s resource %s: %v",
				map[bool]string{true: "create", false: "update"}[created],
				unstructuredObj.GetKind(), err))
		}
		logger.Info("Resource processed successfully", "resource", unstructuredObj.GetKind(), "operation", map[bool]string{true: "created", false: "updated"}[created])
	}

	// 更新 YAMLResource 状态
	return r.updateStatus(ctx, yamlResource, cpodv1beta1.YAMLResourcePhaseRunning, "All resources created/updated successfully")
}

func (r *YAMLResourceReconciler) createOrUpdateResource(ctx context.Context, obj *unstructured.Unstructured) (bool, error) {
	existingObj := &unstructured.Unstructured{}
	existingObj.SetGroupVersionKind(obj.GroupVersionKind())

	err := r.Get(ctx, client.ObjectKey{Namespace: obj.GetNamespace(), Name: obj.GetName()}, existingObj)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// 资源不存在,创建它
			return true, r.Create(ctx, obj)
		}
		return false, err
	}

	// 资源已存在,更新它
	obj.SetResourceVersion(existingObj.GetResourceVersion())
	return false, r.Update(ctx, obj)
}

func (r *YAMLResourceReconciler) updateStatus(ctx context.Context, yamlResource *cpodv1beta1.YAMLResource, phase cpodv1beta1.YAMLResourcePhase, message string) (ctrl.Result, error) {
	yamlResource.Status.Phase = phase
	yamlResource.Status.Message = message
	yamlResource.Status.LastSyncTime = &metav1.Time{Time: time.Now()}

	condition := metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             string(phase),
		Message:            message,
	}
	meta.SetStatusCondition(&yamlResource.Status.Conditions, condition)

	if err := r.Status().Update(ctx, yamlResource); err != nil {
		log.FromContext(ctx).Error(err, "Failed to update YAMLResource status")
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *YAMLResourceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.YAMLResource{}).
		Complete(r)
}
