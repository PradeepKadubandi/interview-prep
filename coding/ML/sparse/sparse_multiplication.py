import numpy as np

# Useful links/blog posts:
# To understand at a high level the different sparse matrix represenations, their strengths and weaknesses:
#   https://rushter.com/blog/scipy-sparse-matrices/

# Below is another blog post, a little verbose (didn't read it fully), has Naive implementations (likely bad reference for the interviews) of only basic get/set operations but maybe be beginner friendly.
#   https://heydenberk.com/blog/posts/sparse-matrix-representations-in-scipy/ 

# History: I asked Chat GPT for sample code for sparse matrix multiplication but it gave me wrong code for a couple of attempts.
# I took the code and corrected it to make it work.
# However, this is assuming COO representation (an ineffective one by taking multiple lists insted of one object with multiple lists) and uses
#   intermediate nested dictionary for multiplication. Maybe there are better ways?

def sparse_matrix_multiply(A_data, A_row_indices, A_col_indices, B_data, B_row_indices, B_col_indices, shape):
    # Create a dictionary to store intermediate results
    intermediate_results = {}
    
    # Create the result matrix
    for i in range(len(A_data)):
        a_row = A_row_indices[i]
        a_col = A_col_indices[i]
        a_value = A_data[i]
        
        if a_col in intermediate_results:
            intermediate_results[a_col][a_row] = a_value
        else:
            intermediate_results[a_col] = {a_row: a_value}

    final_result = {}    
    # Multiply A and B matrices
    for i in range(len(B_data)):
        b_row = B_row_indices[i]
        b_col = B_col_indices[i]
        b_value = B_data[i]
        
        a_col_values = intermediate_results.get(b_row, {})
        for a_row, a_value in a_col_values.items():
            if a_row not in final_result:
                final_result[a_row] = {}
            final_result[a_row][b_col] = a_value * b_value + (final_result.get(a_row, {}).get(b_col, 0))

    result_data = []
    result_row_indices = []
    result_col_indices = []
    for r_row, row_items in final_result.items():
        for r_col, r_value in row_items.items():
            result_data.append(r_value)
            result_row_indices.append(r_row)
            result_col_indices.append(r_col)

    result_matrix = np.zeros(shape)
    result_matrix[result_row_indices, result_col_indices] = result_data
    
    return result_matrix

# Example sparse matrices A and B
A_data = np.array([1, 2, 3, 4, 5, 6])
A_row_indices = np.array([0, 0, 1, 2, 2, 3])
A_col_indices = np.array([0, 2, 2, 0, 1, 3])
A_shape = (4, 4)

B_data = np.array([2, 3, 4, 5])
B_row_indices = np.array([0, 1, 2, 3])
B_col_indices = np.array([0, 1, 2, 3])
B_shape = (4, 4)

result = sparse_matrix_multiply(A_data, A_row_indices, A_col_indices, B_data, B_row_indices, B_col_indices, A_shape)
print(result)