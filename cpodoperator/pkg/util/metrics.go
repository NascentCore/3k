package util

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var (
	jobsCreatedCount = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cpodjob_operator_jobs_created_total",
			Help: "The total number of jobs created",
		},
		[]string{"job_namespace", "job_type"},
	)
)

func init() {
	metrics.Registry.MustRegister(jobsCreatedCount)
}

func CreatedJobsCounterInc(job_namespace, job_type string) {
	jobsCreatedCount.WithLabelValues(job_namespace, job_type).Inc()
}
