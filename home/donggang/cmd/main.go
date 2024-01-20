package main

import (
	"sxwl/3k/home/donggang"
	"time"
)

func main() {
	var grid donggang.Grid

	// 设置一些初始的活细胞
	grid.Set(5, 5, true)
	grid.Set(5, 6, true)
	grid.Set(5, 7, true)
	grid.Set(6, 7, true)
	grid.Set(7, 6, true)

	for {
		grid.Print()
		time.Sleep(200 * time.Millisecond)
		grid = grid.NextGeneration()
	}
}
