package median

import "testing"

func TestGetMedian(t *testing.T) {
	var m FastMedian
	m.AddNum(1)
	m.AddNum(2)
	if m.GetMedian() != 1.5 {
		t.Fail()
	}
	m.AddNum(3)
	if m.GetMedian() != 2.0 {
		t.Fail()
	}
}
