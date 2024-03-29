# import math

# import numpy as np
# import pandas as pd
# import torch
# from scipy import stats
# from sklearn.metrics.pairwise import cosine_similarity


# def attestedfl_1(step: int, worker: int, warm_up: int) -> bool:
#     """
#     The attestedFL_1 algorithm checks
#         the Euclidean distances of a worker's weights vs the chief's weights.
#     """
#     # Calculate the step number for comparison, which is the previous step
#     previous_step: int = step - 1

#     # Load the current and global model weights as PyTorch tensors
#     n_matrix: torch.Tensor = torch.from_numpy(
#         np.load(
#             "/workspace/data/attestedfl/"
#             + worker
#             + "/local_model_"
#             + str(step)
#             + ".npy",
#             allow_pickle=True,
#         )
#     )
#     global_m: torch.Tensor = torch.from_numpy(
#         np.load(
#             "/workspace/data/attestedfl/global_model_"
#             + str(previous_step)
#             + ".npy",
#             allow_pickle=True,
#         )
#     )

#     # Initialize euclidean_distances or load if exists
#     try:
#         euclidean_distances: list[torch.Tensor] = np.load(
#             "/workspace/data/attestedfl/"
#             + worker
#             + "/euclidean_distances_"
#             + str(step)
#             + ".npy",
#             allow_pickle=True,
#         ).tolist()
#     except FileNotFoundError:
#         print(f"step: {step}, no previous euclidean distance file found.")

#     # If it's the first step, calculate and save the Euclidean distance
#     if step == 1:
#         euclidean_distance: torch.Tensor = torch.norm(
#             n_matrix - global_m, p=2
#         ).item()
#         e_d_array: list[list[torch.Tensor]] = [[euclidean_distance]]
#         np.save(
#             f"/workspace/data/attestedfl/{worker}/euclidean_distances_{step}.npy",  # NOQA
#             e_d_array,
#         )
#     else:
#         # Otherwise, append the new Euclidean distance to the list and save
#         euclidean_distance: torch.Tensor = torch.norm(
#             n_matrix - global_m, p=2
#         ).item()
#         e_d_array: list[list[torch.Tensor]] = [[euclidean_distance]]
#         euclidean_distances.append(e_d_array)
#         np.save(
#             f"/workspace/data/attestedfl/{worker}/euclidean_distances_{step}.npy",  # NOQA
#             euclidean_distances,
#         )

#     # After the warm-up period, perform reliability check
#     # based on the calculated Euclidean distances
#     if step > warm_up:
#         euclidean_distances = np.load(
#             f"/workspace/data/attestedfl/{worker}/euclidean_distances_{step}.npy",  # NOQA
#             allow_pickle=True,
#         )
#         c: int = step - warm_up
#         euclidean_distance_to_test: list[torch.Tensor] = euclidean_distances[
#             warm_up:5
#         ]
#         delta_array: list = []

#         # Calculate the delta values for reliability testing
#         for idx, e_d in enumerate(euclidean_distance_to_test):
#             delta = e_d[0]
#             delta_1 = euclidean_distances[warm_up + idx + 1][0]
#             t = warm_up + idx
#             delta_sum = 1 - math.exp(-t / (c * (delta_1 + delta)))
#             delta_array.append(delta_sum)

#         # Calculate average, mean, and standard deviation of deltas
#         delta_avg = np.sum(delta_array) / c
#         delta_mean = np.mean(delta_array)
#         delta_std = np.std(delta_array)

#         # If condition is met, consider the worker reliable
#         if delta_avg <= delta_mean - 4 * delta_std:
#             return True
#         else:
#             return False
#     return True


# def attestedfl_2(step: int, worker: int, warm_up: int) -> bool:
#     """
#     The attestedFL_2 algorithm checks
#         the cosine similarity on the last layer of the CNN model.
#     If the similarity varies significantly
#         from one step to the previous one,
#         the worker is deemed unreliable.
#     """

#     # Only begin checking after the specified warm-up period
#     if step > warm_up:
#         # Calculate the index for the previous step's data
#         previous_step: int = step - 1

#         # Initially assume the worker is not reliable

#         # Load the local model
#         # for the previous step and current step for the given worker
#         n_1_matrix = np.load(
#             f"/workspace/data/attestedfl/{worker}/local_model_{previous_step}.npy",  # NOQA
#             allow_pickle=True,
#         )
#         n_matrix = np.load(
#             f"/workspace/data/attestedfl/{worker}/local_model_{step}.npy",
#             allow_pickle=True,
#         )

#         # Load global model for the previous step
#         global_m = np.load(
#             f"/workspace/data/attestedfl/global_model_{previous_step}.npy",
#             allow_pickle=True,
#         )

#         # Initialize lists to store absolute values of cosine similarities
#         first: list = []
#         second: list = []

#         # Flatten the last layer of each matrix to a 1D array
#         # for cosine similarity calculation
#         n_1: np.array[list[list]] = n_1_matrix[6].reshape(
#             1, -1
#         )  # Previous local model's last layer
#         n: np.array[list[list]] = n_matrix[6].reshape(
#             1, -1
#         )  # Current local model's last layer
#         g: np.array[list[list]] = global_m[6].reshape(
#             1, -1
#         )  # Global model's last layer

#         # Compute the cosine similarities
#         # between the global model's last layer and local models' last layers
#         similarities: np.array = cosine_similarity(n_1, g)
#         similarities_two: np.array = cosine_similarity(n, g)

#         # Append the absolute values of the computed cosine similarities
#         # to the lists
#         first.append(abs(similarities))
#         second.append(abs(similarities_two))

#         # Combine the lists into an array for statistical analysis
#         total: np.array = np.array([first, second])

#         # Uncomment the following line to print the total array for debugging
#         # print(total)

#         # Perform chi-squared test to determine
#         # if there's a significant difference between the similarities
#         _, p_val, _, _ = stats.chi2_contingency(total)

#         # Log the calculated similarities in a CSV file
#         with open(
#             f"data_paper/logs/cosine_attacker_{worker}.csv", "a"
#         ) as logger:
#             logger.write(
#                 f"{step},\
#                     {worker},\
#                         {float(abs(similarities))},\
#                             {float(abs(similarities_two))}\n"
#             )

#         # Based on the p-value from chi-squared test,
#         # determine if the worker is reliable or not
#         if p_val < 0.1:
#             print(
#                 f"{worker} is not reliable"
#             )  # Print message if the worker is not reliable
#             return False
#         else:
#             return True
#     else:
#         # If the step is within the warm-up period,
#         # default to considering the worker reliable
#         return True


# def attestedfl_3(step: int, worker: int, warm_up: int) -> bool:
#     """
#     The attestedfl_3 function evaluates
#         a worker's training progress
#         by fitting a logarithmic curve to the reported errors.
#     It considers the slope of this curve
#         if the slope is negative or very small (indicating diminishing errors),
#         it deems the worker as actually training.
#         A steep positive slope would suggest the contrary.
#     """

#     # Initially assume the worker is reliable

#     # Only perform checks after the specified warm-up period
#     if step > warm_up:
#         # If we are past the warm-up,
#         # we start with the assumption that the worker is not reliable

#         # Read the CSV file which contains the errors
#         # for each iteration for a specific worker
#         errors_table = pd.read_csv(
#             f"data_paper/logs/attestedFL-3/errors_{worker}.csv", header=None
#         )

#         # Column 0 contains the iteration number
#         iteration = errors_table[0]

#         # Column 2 is assumed to contain the error values
#         # for the corresponding iterations
#         errors = errors_table[2]

#         # Fit a logarithmic curve to the errors data
#         # (log(iteration) vs. errors)
#         fittedParameters = np.polyfit(np.log(iteration), errors, 1)

#         # Calculate predictions using the fitted curve
#         # at the first and current step
#         first_prediction = np.polyval(fittedParameters, 1)
#         last_prediction = np.polyval(fittedParameters, step)

#         # Calculate the slope of the line
#         # between the first and the last prediction
#         slope = (last_prediction - first_prediction) / (step - 1)

#         # Check the slope against the thresholds
#         # to determine if the worker is training or not
#         if slope <= 0:
#             # Negative slope: errors are decreasing,
#             # indicating proper training
#             return True
#         else:
#             if slope <= 0.4:
#                 # Small positive slope: may still be indicative of training
#                 return True
#             else:
#                 # Large positive slope:
#                 # errors aren't reducing as expected, thus unreliable
#                 return False

#     # If we're within the warm-up period,
#     # remain with the initial assumption that the worker is reliable
#     return True


# def attestedfl(step: int, worker: int) -> bool:
#     # Set the warmup period
#     # for the workers to start converging towards the global model.
#     warm_up: int = 30

#     # Sequentially test reliability
#     # through attestedfl_1, attestedfl_2, and attestedfl_3
#     if attestedfl_1(step, worker, warm_up):
#         if attestedfl_2(step, worker, warm_up):
#             if attestedfl_3(step, worker, warm_up):
#                 return True

#     # Return false if any of the tests fail
#     return False
