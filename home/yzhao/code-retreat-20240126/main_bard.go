package main

import (
    "fmt"
    "math/rand"
)

type CellState int

const (
    CellAlive CellState = 1
    CellDead CellState = 0
)

type Cell struct {
    x, y int
    state CellState
}

type Grid struct {
    width, height int
    cells [][]Cell
}

func NewGrid(width, height int) *Grid {
    grid := &Grid{
        width: width,
        height: height,
        cells: make([][]Cell, width),
    }
    for i := 0; i < width; i++ {
        grid.cells[i] = make([]Cell, height)
    }
    return grid
}

func (g *Grid) Init() {
    for i := 0; i < g.width; i++ {
        for j := 0; j < g.height; j++ {
            g.cells[i][j].state = CellAlive
        }
    }
}

func (g *Grid) NextGeneration() {
    newGrid := NewGrid(g.width, g.height)
    for i := 0; i < g.width; i++ {
        for j := 0; j < g.height; j++ {
            neighbors := 0
            for k := -1; k <= 1; k++ {
                for l := -1; l <= 1; l++ {
                    if 0 <= i+k < g.width && 0 <= j+l < g.height && g.cells[i+k][j+l].state == CellAlive {
                        neighbors++
                    }
                }
            }
            newGrid.cells[i][j].state = CellState(neighbors)
        }
    }
    g.cells = newGrid.cells
}

func (g *Grid) Print() {
    for i := 0; i < g.width; i++ {
        for j := 0; j < g.height; j++ {
            switch g.cells[i][j].state {
            case CellAlive:
                fmt.Print("*")
            case CellDead:
                fmt.Print(" ")
            }
        }
        fmt.Println()
    }
}

func main() {
    grid := NewGrid(100, 100)
    grid.Init()
    for {
        grid.Print()
        grid.NextGeneration()
    }
}

