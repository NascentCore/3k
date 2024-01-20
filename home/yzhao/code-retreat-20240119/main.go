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
	return make([]point, 0, 8)
}

func countAlive(board map[string]bool, p point) int {
	return 0
}

func next(board map[string]bool) map[string]bool {
	return make(map[string]bool)
}
