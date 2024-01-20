package main

import (
	"reflect"
	"testing"
)

func TestPointStringConv(t *testing.T) {
	if pointToString(point{1, 1}) != "1,1" {
		t.Fail()
	}
	if !reflect.DeepEqual(stringToPoint("1,1"), point{1, 1}) {
		t.Fail()
	}
}

func TestGetNeighbors(t *testing.T) {
	got := getNeighbors(point{1, 1})
	expected := []point{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}, {2, 2}}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("countAlive failed, got %v expected %v", got, expected)
	}
}

func TestCountAlive(t *testing.T) {
	b := make(map[string]bool)
	b["0,0"] = true
	b["0,1"] = true
	b["1,0"] = true

	if c := countAlive(b, point{1, 1}); c != 3 {
		t.Errorf("countAlive failed, got %d expected %d", c, 3)
	}

	if c := countAlive(b, point{0, 1}); c != 2 {
		t.Errorf("countAlive failed, got %d expected %d", c, 2)
	}

	if c := countAlive(b, point{2, 0}); c != 1 {
		t.Errorf("countAlive failed, got %d expected %d", c, 1)
	}
}

func TestNext(t *testing.T) {
	b := make(map[string]bool)
	b["0,0"] = true
	b["0,1"] = true
	b["0,2"] = true

	got := next(b)

	expected := make(map[string]bool)
	expected["1,1"] = true
	expected["0,1"] = true
	expected["-1,1"] = true

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got %v expected %v", got, expected)
	}
}
