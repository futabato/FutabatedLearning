import pickle
import sys

import hydra
import pika
import torch
from federatedlearning.datasets.common import get_dataset
from federatedlearning.models.cnn import CNNMnist
from federatedlearning.server.aggregations.aggregators import average_weights
from federatedlearning.server.inferencing import inference
from omegaconf import DictConfig
from pika.exceptions import AMQPError


class FLServerPublisher:
    def __init__(
        self,
        host="rabbitmq",
        exchage_name="global_model_exchange",
        username="guest",
        password="guest",
    ):
        self.host = host
        self.credentials = pika.PlainCredentials(
            username=username, password=password
        )
        self.exchange_name = exchage_name
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.host, credentials=self.credentials)
        )
        self.channel = self.connection.channel()
        self.channel.exchange_declare(
            exchange=self.exchange_name, exchange_type="fanout"
        )

    def publish(self, round: int, serialized_data: bytes):
        try:
            headers = {"round": round}
            self.channel.basic_publish(
                exchange=self.exchange_name,
                routing_key="",
                body=serialized_data,
                properties=pika.BasicProperties(headers=headers),
            )
            print(f" [x] Published serialized model to {self.exchange_name}")
        except AMQPError as e:
            # エラーハンドリング: 必要に応じてログを記録したり再試行したりする
            print("An error occurred: ", e)
            self._connect()  # 必要なら再接続を試みる

    def close(self):
        if self.connection:
            self.connection.close()


class FLServerSubscriber:
    def __init__(
        self,
        cfg: DictConfig,
        host: str = "rabbitmq",
        queue_name: str = "local_model_queue",
        username: str = "guest",
        password: str = "guest",
    ):
        self.host = host
        self.credentials = pika.PlainCredentials(
            username=username, password=password
        )
        self.queue_name = queue_name
        self.connection = None
        self.channel = None

        self.cfg = cfg
        # Load the dataset and partition it according to the client groups
        (
            self.train_dataset,
            self.test_dataset,
            self.client_groups,
        ) = get_dataset(self.cfg)
        self.device: torch.device = (
            torch.device(f"cuda:{cfg.train.gpu}")
            if cfg.train.gpu is not None and cfg.train.gpu >= 0
            else torch.device("cpu")
        )

        # モデル集約と重みのリスト
        self.local_models = []
        self.local_model_weights = []
        self.client_set = set()

        # 実験設定
        self.round = 0

        self.publisher = FLServerPublisher()

        self._connect()

    def _connect(self):
        self.connection = pika.BlockingConnection(
            parameters=pika.ConnectionParameters(
                self.host, credentials=self.credentials
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name)

    def start_consuming(self):
        # コールバック関数を設定してメッセージを消費し始める
        def callback(ch, method, properties, body):
            client_id = properties.headers.get("client_id")
            print(f" [x] Received from client_id: {client_id}")

            global_model = CNNMnist(self.cfg)

            # ローカルのモデルおよびオプティマイザのインスタンスを作成
            local_model = CNNMnist(self.cfg)
            model_binary = pickle.loads(body)
            local_model.load_state_dict(model_binary)

            # ローカルモデルの重みを追加
            self.local_model_weights.append(local_model.state_dict())
            self.local_models.append(local_model)

            print(f" [x] Received model from client {client_id}")

            # 集約のタイミングをここで決める（例えば、2つのローカルモデルが集まったら）
            if len(self.local_models) >= 2:
                global_model_weight = average_weights(self.local_model_weights)
                global_model.load_state_dict(global_model_weight)
                global_model.to(self.device)
                self.local_models.clear()
                self.local_model_weights.clear()
                self.client_set.clear()

                serialized_model = pickle.dumps(global_model_weight)

                print(f" [x] Round {self.round} DONE")

                test_acc, _ = inference(
                    cfg=self.cfg,
                    model=global_model,
                    test_dataset=self.test_dataset,
                )
                print(f"[x] Accuracy: {test_acc}")
                if self.round < self.cfg.federatedlearning.rounds:
                    self.round += 1

                    # メッセージ送信
                    self.publisher.publish(self.round, serialized_model)
                else:
                    # 終わり
                    sys.exit()

        # Consumeの開始
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback,
            auto_ack=True,
        )
        print(" [*] Waiting for messages. To exit press CTRL+C")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.stop_consuming()

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
    global_model = CNNMnist(cfg=cfg)
    state_dict = global_model.state_dict()
    serialized_model = pickle.dumps(state_dict)

    publisher = FLServerPublisher()
    # メッセージ送信
    publisher.publish(round=0, serialized_data=serialized_model)
    subscriber = FLServerSubscriber(cfg)
    try:
        # メッセージの受信を開始
        subscriber.start_consuming()
    finally:
        # 終了時に接続を閉じる
        subscriber.close()


if __name__ == "__main__":
    main()
