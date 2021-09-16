## usage of the A star algorithm interface

### parameter
|name|info|type|options|
|:----|:----|:----|:---|
|maze|the information of the maze|list|/|
|rows|count of the rows|int|/|
|columns|count of the columns|int|/|
|model|model to calculate distance|string|"E":"Euclidean Distance"<br/>"M":"Manhattan Distance"<br/>"C":"Chebyshev Distance"|

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
