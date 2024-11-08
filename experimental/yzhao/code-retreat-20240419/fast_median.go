package median

import "sort"

type FastMedian []int

func (this *FastMedian) AddNum(v int) {
	*this = append(*this, v)
}

func (this FastMedian) GetMedian() float64 {
	if len(this) == 0 {
		panic("Empty!")
	}
	if len(this) == 1 {
		return float64(this[0])
	}
	sort.Ints(this)
	midIdx := len(this) / 2
	if len(this)%2 == 0 {
		return (float64(this[midIdx]) + float64(this[midIdx-1])) / 2
	} else {
		return float64(this[midIdx])
	}
}
