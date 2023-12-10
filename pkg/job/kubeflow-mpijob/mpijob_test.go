package kubeflowmpijob

import (
	"testing"
)

func Test_genJsonData(t *testing.T) {
	mpiJob := MPIJob{
		ExecutionDurationSeconds: "1000",
	}
	m := mpiJob.genJsonData()
	mIntf, ok := m["metadata"]
	if !ok {
		t.Fail()
	}
	metadataMap, ok := mIntf.(map[string]interface{})
	if !ok {
		t.Fail()
	}
	labelsIntf, ok := metadataMap["labels"]
	if !ok {
		t.Fail()
	}
	labelsMap, ok := labelsIntf.(map[string]interface{})
	if !ok {
		t.Fail()
	}
	exeDuIntf, ok := labelsMap["executionDurationSeconds"]
	if !ok {
		t.Fail()
	}
	exeDu, ok := exeDuIntf.(string)
	if !ok {
		t.Fail()
	}
	if exeDu != "1000" {
		t.Fail()
	}
}
