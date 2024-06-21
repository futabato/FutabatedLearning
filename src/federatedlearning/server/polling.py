import os
import time


def poll_for_model_updates(
    directory, num_clients, num_rounds, polling_interval=5
):
    """
    Polling function to check if sufficient model updates are present in the directory.

    :param directory: str - Path to the shared directory
    :param num_clients: int - Number of client updates to wait for
    :param num_rounds: int - Number of rounds
    :param polling_interval: int - Time interval (in seconds) between each poll
    """
    round = 0
    while round < num_rounds:
        # List all files in the directory
        try:
            files = os.listdir(os.path.join(directory, f"round_{round}"))
        except FileNotFoundError:
            time.sleep(polling_interval)
            continue
        print(os.path.join(directory, f"round_{round}"))
        print(files)

        # Filter out only model update files (assuming they have a specific naming convention)
        # For this example, let's assume the files are named 'client_{i}.pth' where {i} is the client ID.
        update_files = [f for f in files if f.startswith("client_")]

        # Check if we have received updates from all clients
        if len(update_files) >= num_clients:
            print(
                f"Received updates from {num_clients} clients. Proceeding with aggregation."
            )
            round += 1

        print(
            f"Waiting for more updates. Currently have {len(update_files)} out of {num_clients}."
        )

        # Wait for the specified polling interval before checking again
        time.sleep(polling_interval)


if __name__ == "__main__":
    shared_directory = "/shared"
    num_clients = 10
    num_rounds = 10
    polling_interval_seconds = 5

    poll_for_model_updates(
        shared_directory, num_clients, polling_interval_seconds
    )
