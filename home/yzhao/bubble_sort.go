package bubblesort

func BubbleSort(ints []int) {
	for i := len(ints); i >= 0; i -= 1 {
		for j := 0; j < i - 1; j += 1 {
			if ints[j] > ints[j+1] {
				ints[j], ints[j+1] = ints[j+1], ints[j]
			}
		}
	}
}
