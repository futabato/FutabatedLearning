import pickle
import time
from concurrent import futures

import federated_pb2
import federated_pb2_grpc
import grpc
import torch.nn as nn
from federatedlearning.models.cnn import CNNCifar, CNNMnist
from federatedlearning.server.aggregations.aggregators import average_weights
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


class FederatedLearningServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rounds = cfg.federatedlearning.rounds
        self.current_round = 0
        self.global_model = initialize_global_model(self.cfg)  # 初期のグローバルモデルを設定
        self.local_updates = []

    def SendLocalUpdate(self, request, context):
        local_weights = pickle.loads(request.weights)
        # local_loss = request.loss

        self.local_updates.append(local_weights)

        if len(self.local_updates) == self.cfg.federatedlearning.num_clients:
            self.aggregate()
        
        message = f"Round {self.current_round} completed"
        self.current_round += 1
        response = federated_pb2.ServerResponse(message=message)
        return response

    def GetGlobalModel(self, request, context):
        global_weight_bytes = pickle.dumps(self.global_model.state_dict())
        return federated_pb2.GlobalModel(weights=global_weight_bytes)

    def aggregate(self):
        # 重みを集約する処理を実装
        self.global_model.load_state_dict(
            average_weights(self.local_updates)
        )
        self.local_updates = []
    
    def conduct_round(self):
        self.current_round += 1

def serve(cfg: DictConfig):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServer(cfg), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)  # 1日待機
    except KeyboardInterrupt:
        server.stop(0)

def initialize_global_model(cfg: DictConfig):
    # グローバルモデルの初期設定
    global_model: nn.Module
    if cfg.train.dataset == "mnist":
        global_model = CNNMnist(cfg=cfg)
    elif cfg.train.dataset == "cifar":
        global_model = CNNCifar(cfg=cfg)
    return global_model

if __name__ == '__main__':
    initialize(version_base="1.1", config_path="../config", job_name="jupyterlab")
    cfg: OmegaConf = compose(config_name="default")
    print(OmegaConf.to_yaml(cfg))
    serve(cfg=cfg)
