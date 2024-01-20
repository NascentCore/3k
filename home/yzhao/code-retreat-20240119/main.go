package main

import (
	"fmt"
)

type point struct {
	x int
	y int
}

func pointToString(p point) string {
	return fmt.Sprintf("%d,%d", p.x, p.y)
}

func getNeighbors(p point) []point {
	res := make([]point, 0, 8)
	for _, i := range []int{-1, 0, 1} {
		for _, j := range []int{-1, 0, 1} {
			if i == 0 && j == 0 {
				continue
			}
			res = append(res, point{p.x+i, p.y+j})
		}
	}
	return res
}

func countAlive(board map[string]bool, p point) int {
	return 0
}

func next(board map[string]bool) map[string]bool {
	return make(map[string]bool)
}
