package main

import (
	"fmt"
	"strings"
	"strconv"
)

type point struct {
	x int
	y int
}

func pointToString(p point) string {
	return fmt.Sprintf("%d,%d", p.x, p.y)
}

func stringToPoint(s string) point {
	strs := strings.Split(s, ",")
	x, _ := strconv.Atoi(strs[0])
	y, _ := strconv.Atoi(strs[1])
	return point{x, y}
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
	count := 0
	for _, p := range getNeighbors(p) {
		if _, ok := board[pointToString(p)]; ok {
			count += 1
		}
	}
	return count
}

func getPointsToCheck(board map[string]bool) []point {
	res := make([]point, 0, len(board))
	for pStr, _ := range board {
		res = append(res, getNeighbors(stringToPoint(pStr))...)
	}
	return res
}

func next(board map[string]bool) map[string]bool {
	res := make(map[string]bool)
	for _, p := range getPointsToCheck(board) {
		c := countAlive(board, p)
		_, isAlive := board[pointToString(p)]
		if isAlive && (c == 2 || c == 3) {
			res[pointToString(p)] = true
		}
		if !isAlive && c == 3 {
			res[pointToString(p)] = true
		}
	}
	return res
}
