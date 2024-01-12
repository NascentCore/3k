package config

import (
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	got := parse("a: foo\nb:bar \n c:baz")
	expected := map[string]string {
		"a": "foo",
		"b": "bar",
		"c": "baz",
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got %v expect %v", got, expected)
	}
}
