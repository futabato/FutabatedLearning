import grpc
from concurrent import futures
import time
import torch
import pickle
from federatedlearning import federated_pb2, federated_pb2_grpc

class FederatedLearningServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.global_model = initialize_global_model()  # 初期のグローバルモデルを設定

    def SendLocalUpdate(self, request, context):
        local_weights = pickle.loads(request.weights)
        local_loss = request.loss

        # ローカルモデルの重みを集約する処理
        self.aggregate(local_weights)

        response = federated_pb2.ServerResponse(message="Received and processed")
        return response

    def aggregate(self, local_weights):
        # 重みを集約する処理を実装
        pass

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)  # 1日待機
    except KeyboardInterrupt:
        server.stop(0)

def initialize_global_model():
    # グローバルモデルの初期設定
    model = ...  # モデルインスタンスを作成
    return model

if __name__ == '__main__':
    serve()
