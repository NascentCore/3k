package donggang

import "fmt"

func (g *Grid) get(x, y int) bool {
	return g[x][y]
}

const (
	width  = 40
	height = 20
)

type Grid [width][height]bool

func (g *Grid) Set(x, y int, state bool) {
	g[x][y] = state
}

func (g *Grid) Get(x, y int) bool {
	return g[x][y]
}

func (g *Grid) CountNeighbors(x, y int) int {
	count := 0
	for i := x - 1; i <= x+1; i++ {
		for j := y - 1; j <= y+1; j++ {
			if i == x && j == y {
				continue
			}
			nx, ny := (i+width)%width, (j+height)%height
			if g.get(nx, ny) {
				count++
			}
		}
	}
	return count
}

func (g *Grid) NextGeneration() Grid {
	var next Grid
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			neighbors := g.CountNeighbors(x, y)
			switch {
			case g.get(x, y) && neighbors < 2:
				next.Set(x, y, false)
			case g.get(x, y) && (neighbors == 2 || neighbors == 3):
				next.Set(x, y, true)
			case g.get(x, y) && neighbors > 3:
				next.Set(x, y, false)
			case !g.get(x, y) && neighbors == 3:
				next.Set(x, y, true)
			default:
				next.Set(x, y, g.get(x, y))
			}
		}
	}
	return next
}

func (g *Grid) Print() {
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if g.get(x, y) {
				fmt.Print("■ ")
			} else {
				fmt.Print("□ ")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}
