package main

import (
	"fmt"
)

func bubbleSort(arr[] int) []int {
	length := len(arr)
	if length <=1 {
		return arr
	}
	for i :=0;i < length - 1;i++ {
		for j :=0;j < length - i -1;j++ {
			if arr[j+1] > arr[j] {
				arr[j],arr[j+1] = arr[j+1],arr[j]
			}
		} 
	}
	return arr
}

func main() {
	arr := []int{11,8,2,5,7,10,3,6}
	fmt.Println(bubbleSort(arr))

}
