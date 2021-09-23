## usage of the A star algorithm interface

### parameter
|name|info|type|options|
|:----|:----|:----|:---|
|maze|the information of the maze|list|/|
|rows|count of the rows|int|/|
|columns|count of the columns|int|/|
|model|model to calculate distance|string|"E":"Euclidean Distance" (default)<br/>"M":"Manhattan Distance"<br/>"C":"Chebyshev Distance"|

### using example

```

from A_star import A_star_search

if __name__ == '__main__':
    
    maze=[[0,0,0,0,0,0,0,0,0,1],
          [0,0,0,0,0,0,0,0,1,0],
          [0,0,1,0,0,0,0,1,0,0],
          [0,1,0,0,0,0,1,0,0,0],
          [1,0,0,0,0,1,1,0,0,0],
          [0,0,0,0,1,0,0,1,1,0],
          [0,0,0,1,0,0,1,1,1,0],
          [0,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0]]
    result=A_star_search(maze,10,10, model="C")
    print(result)

```

### result example

```
[[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [5, 2], [6, 2], [6, 1], [7, 1], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [9, 9]]
```


## usage of the repeat forward A star algorithm interface

### Description

The repeat forward A star algorithm will repeat the route because it always starts at the cell where it encounters an obstacle in the last iteration. So the length of the path will much larger than the length of the shortest path.

### parameter
|name|info|type|options|
|:----|:----|:----|:---|
|maze|the information of the maze|list|/|
|rows|count of the rows|int|/|
|columns|count of the columns|int|/|
|model|model to calculate distance|string|"E":"Euclidean Distance" (default)<br/>"M":"Manhattan Distance"<br/>"C":"Chebyshev Distance"|
|start_cell|start cell of the maze|list|/|
|end_cell|the goal of the maze|list|/|

### using example
```angular2html
from Repeat_A_star import repeat_A_star
if __name__ == '__main__':
    maze=[[0,0,0,0,0,0,0,0,0,1],
          [0,0,0,0,0,0,0,0,1,0],
          [0,0,1,0,0,0,0,1,0,0],
          [0,1,0,0,0,0,1,0,0,0],
          [1,0,0,0,0,1,1,0,0,0],
          [0,0,0,0,1,0,0,1,1,0],
          [0,0,0,1,0,0,1,1,1,0],
          [0,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0]]
    
    result=repeat_A_star(maze=maze,
                         rows=10,
                         columns=10,
                         start_cell=[0,0],
                         end_cell=[9,9],
                         model='E')
    print(result)

```

### result example

```
[[0, 0], [1, 0], [1, 1], [2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [3, 4], [3, 5], [2, 5], [2, 6], [1, 6], [1, 7], [0, 7], [0, 8], [0, 7], [1, 7], [1, 6], [2, 6], [2, 5], [3, 5], [3, 4], [4, 4], [4, 3], [5, 3], [5, 2], [6, 2], [6, 1], [7, 1], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [9, 9]]

```

## usage of the improved repeat forward A star algorithm interface



### parameter
|name|info|type|options|
|:----|:----|:----|:---|
|maze|the information of the maze|list|/|
|rows|count of the rows|int|/|
|columns|count of the columns|int|/|
|model|model to calculate distance|string|"E":"Euclidean Distance" (default)<br/>"M":"Manhattan Distance"<br/>"C":"Chebyshev Distance"|
|start_point|start cell of the maze|list|/|
|end_point|the goal of the maze|list|/|

### using example
```angular2html
from Improvement_repeat_A_star import improved_repeat_A_star

if __name__ == '__main__':
    maze=[[0,0,0,0,0,0,0,0,0,1],
          [0,0,0,0,0,0,0,0,1,0],
          [0,0,1,0,0,0,0,1,0,0],
          [0,1,0,0,0,0,1,0,0,0],
          [1,0,0,0,0,1,1,0,0,0],
          [0,0,0,0,1,0,0,1,1,0],
          [0,0,0,1,0,0,1,1,1,0],
          [0,0,1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0]]

    result=improved_repeat_A_star(start_point=[0,0],
                                  end_point=[9,9],
                                  maze=maze,
                                  rows=10,
                                  columns=10,
                                  model="E")
    print(result)
```

### result example

```
[[0, 0], [0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [4, 2], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 8], [9, 9]]


```