import numpy as np
import os

## INPUT/OUTPUT UTILITY FUNCTIONS

def generate_random_input(*constraints):
    input_dict = {}
    for dict in constraints:
        for param in dict:
            constraint = dict[param]
            if type(constraint) == list:
                # Constraint contains upper and lower bound
                # Uniformly sample value within bound
                lower_bound, upper_bound = constraint[0], constraint[1]
                sample = np.random.uniform(lower_bound, upper_bound)
                input_dict[param] = sample
            elif type(constraint) == int or type(constraint) == float:
                # Constraint is constant
                input_dict[param] = constraint
    return input_dict

def get_perturbable_indices(*constraints):
    idxs = {}
    for dict in constraints:
        for param in dict:
            constraint = dict[param]
            if type(constraint) == list:
                # Constraint contains upper and lower bound
                idxs[param] = True
            elif type(constraint) == int or type(constraint) == float:
                # Constraint is constant, therefore not perturbable
                idxs[param] = False
    return to_vector(idxs)

def to_vector(input_dict):
    vec = np.zeros(shape=(len(input_dict)))
    i = 0
    for key in input_dict:
        vec[i] = input_dict[key]
        i+=1
    return vec

def to_dict(input_vec, like_dict):
    assert input_vec.shape[0] == len(like_dict)
    i = 0
    out_dict = {}
    for key in like_dict:
        out_dict[key] = input_vec[i]
        i += 1
    return out_dict

def perturb_input(input, perturbable_indices, delta):
    # For each input, compute input + delta, input + delta
    N = input.shape[0]
    M = perturbable_indices[perturbable_indices == True].shape[0]
    out = np.zeros(shape=(M, N, 2))
    j = 0
    for i in range(N):
        if not perturbable_indices[i]:
            continue
        # delta_vec = [0, 0, ..., delta, ..., 0, 0] s.t. delta is at index i 
        delta_vec = np.zeros(shape=(input.shape[0]))
        delta_vec[i] = delta
        # Perturb input by delta
        out[j,:,0] = input - delta_vec
        out[j,:,1] = input + delta_vec
        j+=1
    return out

def save_perturbed_inputs(inputs, directory, like_dict):
    #inputs should have shape [num perturbable params, num input params, 2]
    try:
        os.mkdir(directory)
        print(f"Directory '{directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory}' already exists.")
    
    M = inputs.shape[0]
    for i in range(M):
        for j in range(2):
            input = inputs[i,:,j]
            input_dict = to_dict(input, like_dict)
            input_dir = os.path.join(directory, f'input-{format_input_num((i*2)+j)}')
            input_file = os.path.join(input_dir, 'input.tglf')
            try:
                os.mkdir(input_dir)
                print(f"Directory '{input_dir}' created successfully.")
            except FileExistsError:
                print(f"Directory '{input_dir}' already exists.")
            save_input(input_dict, input_file)

def save_input(input, path):
    with open(path, "w") as file:
        for key in input:
            file.write(f"{key} = {input[key]}\n")
        file.close()
    print(f'Successfully saved {path}')

def compute_partial_central_differencing(out_right, out_left, delta):
    return (out_right - out_left) / (2 * delta)

def compute_jacobian_central_differencing(out_right, out_left, delta):
    #out shapes should be [num_outputs, num_inputs]
    assert out_left.shape == out_right.shape

    M = out_left.shape[0]
    N = out_left.shape[1]

    J = np.zeros(shape=(M,N))
    for i in range(M):
        for j in range(N):
            partial_ij = compute_partial_central_differencing(out_right[i,j], out_left[i,j], delta)
            J[i,j] = partial_ij
   
    return J

def format_input_num(num):
    if num < 10:
        return f'00{num}'
    elif num < 100:
        return f'0{num}'
    else:
        return f'{num}'