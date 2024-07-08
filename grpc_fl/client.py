import pickle

import federated_pb2
import federated_pb2_grpc
import grpc
from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNMnist
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


# @hydra.main(
#     version_base="1.1", config_path="/workspace/config", config_name="default"
# )
def run_client(cfg: DictConfig, client_id: int, server_address: str ='localhost:50051'):
    train_dataset, test_dataset, client_groups = get_dataset(cfg)

    with grpc.insecure_channel(server_address) as channel:
        stub = federated_pb2_grpc.FederatedLearningStub(channel)

        # グローバルモデルの取得
        response = stub.GetGlobalModel(federated_pb2.Empty())
        global_weights = pickle.loads(response.weights)

        # グローバルモデルのロード
        global_model = CNNMnist(cfg)
        global_model.load_state_dict(global_weights)

        local_update = LocalUpdate(cfg=cfg, dataset=train_dataset, client_id=client_id, idxs=client_groups[client_id])
        local_weights, _ = local_update.update_weights(global_model, global_round=0) # ラウンド数の指定が必要、今は0にしている

        weights_bytes = pickle.dumps(local_weights)
        request = federated_pb2.LocalUpdate(weights=weights_bytes)
        response = stub.SendLocalUpdate(request)
        print(response.message)

if __name__ == '__main__':
    initialize(version_base="1.1", config_path="../config", job_name="jupyterlab")
    cfg: OmegaConf = compose(config_name="default")
    print(OmegaConf.to_yaml(cfg))
    run_client(cfg=cfg, client_id=0)  # クライアントIDを指定
