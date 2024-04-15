def calculate_cnn_output_dim(input_size, kernel_sizes, strides, paddings):
    output_size = input_size

    for kernel_size, stride, padding in zip(kernel_sizes, strides, paddings):
        output_size = ((output_size - kernel_size + 2 * padding) // stride) + 1

    return output_size
