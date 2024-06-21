import grpc
import pickle
import torch
from federatedlearning import LocalUpdate
from federatedlearning import federated_pb2, federated_pb2_grpc

def run_client(client_id, server_address='localhost:50051'):
    dataset, client_groups = get_dataset(cfg)
    client_data = client_groups[client_id]

    local_update = LocalUpdate(cfg, dataset, client_id, client_data)
    local_weights, local_loss = local_update.update_weights(global_model)

    with grpc.insecure_channel(server_address) as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)
        weights_bytes = pickle.dumps(local_weights)
        request = federated_pb2.LocalUpdate(weights=weights_bytes, loss=local_loss)
        response = stub.SendLocalUpdate(request)
        print(response.message)

if __name__ == '__main__':
    run_client(client_id=0)  # クライアントIDを指定
