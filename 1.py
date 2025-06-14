import numpy as np


a =[
    [2.155,3.2551,4.155,6.2222,0.888,3.0,100.0],
    [2.155,3.2551,4.155,6.2222,0.888,3.0,100.0],
    [2.155,3.2551,4.155,6.2222,0.888,3.0,100.0],
    [2.155,3.2551,4.155,6.2222,0.888,4.0,100.0]
]

def preprocess_boxes(boxes: np.ndarray, c: int) -> list[list[int]]:
    return [list(map(int, listtmp)) for listtmp in boxes.tolist() if int(listtmp[5]) == c]
npa = np.array(a)

print(preprocess_boxes(npa, 3))