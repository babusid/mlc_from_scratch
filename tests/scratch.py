import tmlc.ops.ops_basic as ops_basic
import tmlc.evaluator as e
import tmlc.ndarray as nd
import numpy as np

x = ops_basic.input(shape=(2, 2), label="x")
y = ops_basic.input(shape=(2, 2), label="y")
z = x + y * 3

print(z)

output = e.run(
        inputs={
            x: np.array([[1, 2], [3, 4]]),
            y: np.array([[5, 6], [7, 8]])
        },
        outputs=[z]
    )

print(output[0][0])
