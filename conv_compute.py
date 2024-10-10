"""
以检测灰度图像的边缘为例，展示卷积层的计算逻辑
"""

import numpy as np
def conv(input, kernel):
    input_height, input_width = input.shape
    kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            output[h][w] = np.sum(input[h:h+kernel_height, w:w+kernel_width] * kernel)
    return output

input = np.random.randint(0, 255, (5, 5))
kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

print("input: ",input)
output = conv(input, kernel)
print("output: ", output)
