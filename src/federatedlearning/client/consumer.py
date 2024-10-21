import copy
import pickle

import hydra
import pika
import torch
import torch.nn as nn
from federatedlearning.client.training import LocalUpdate
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNMnist
from omegaconf import DictConfig


class FLClient:
    def __init__(
        self,
        cfg: DictConfig,
        host: str = "rabbitmq",
        local_queue: str = "local_model_queue",
        exchage_name: str = "global_model_exchange",
        username: str = "guest",
        password: str = "guest",
    ):
        self.host = host
        self.credentials = pika.PlainCredentials(
            username=username, password=password
        )
        self.local_queue = local_queue
        self.exchange_name = exchage_name
        self.connection = None
        self.channel = None

        # 実験設定
        self.cfg = cfg
        self.client_id = cfg.client.client_id
        # Load the dataset and partition it according to the client groups
        (
            self.train_dataset,
            self.test_dataset,
            self.client_groups,
        ) = get_dataset(self.cfg)
        # Determine the computing device (GPU or CPU)
        self.device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )
        self.num_epochs = 3

        self._connect()

    def _connect(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.host, credentials=self.credentials)
        )
        self.channel = self.connection.channel()
        self.result = self.channel.queue_declare(queue="", exclusive=True)
        self.global_queue = self.result.method.queue
        self.channel.queue_bind(
            exchange=self.exchange_name, queue=self.global_queue
        )

    def receive_global_model(self):
        def callback(ch, method, properties, body):
            round = properties.headers.get("round")
            state_dict = pickle.loads(body)
            global_model = CNNMnist(self.cfg)
            global_model.load_state_dict(state_dict)
            global_model.to(self.device)
            print(" [x] Received initial global model")

            # ローカル学習
            local_model = self.local_train(global_model, round)

            # ローカルモデルをサーバに送信
            self.send_local_model(model=local_model, client_id=self.client_id)

        self.channel.basic_consume(
            queue=self.global_queue,
            on_message_callback=callback,
            auto_ack=True,
        )
        print(" [*] Waiting for initial global model. To exit press CTRL+C")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop_consuming()

    def local_train(self, global_model: nn.Module, round: int):
        print("[x] Training model locally...")
        local_model = LocalUpdate(
            cfg=self.cfg,
            dataset=self.train_dataset,
            client_id=self.client_id,
            idxs=self.client_groups[self.client_id],
        )
        weight, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=round
        )

        return weight

    def send_local_model(self, model: nn.Module, client_id: str):
        serialized_model = pickle.dumps(model)
        headers = {"client_id": client_id}
        self.channel.basic_publish(
            exchange="",
            routing_key=self.local_queue,
            body=serialized_model,
            properties=pika.BasicProperties(headers=headers),
        )
        print(f" [x] Sent updated local model from client {client_id}")

    def stop_consuming(self):
        if self.channel:
            self.channel.stop_consuming()

    def close(self):
        if self.connection:
            self.connection.close()


@hydra.main(
    version_base="1.1", config_path="/workspace/config", config_name="default"
)
def main(cfg: DictConfig):
    client = FLClient(cfg)
    try:
        # サーバからのグローバルモデルを待ち、それを受信
        client.receive_global_model()
    finally:
        # 終了時に接続を閉じる
        client.close()


if __name__ == "__main__":
    main()
