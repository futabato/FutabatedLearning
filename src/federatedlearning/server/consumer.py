import pickle
import sys

import pika
import torch
import torch.nn as nn
from federatedlearning.server.aggregations.aggregators import average_weights


# PyTorchの簡単なモデル定義
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


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

    def publish(self, serialized_data: bytes):
        try:
            self.channel.basic_publish(
                exchange=self.exchange_name,
                routing_key="",
                body=serialized_data,
            )
            print(f" [x] Published serialized model to {self.exchange_name}")
        except pika.exceptions.AMQPError as e:
            # エラーハンドリング: 必要に応じてログを記録したり再試行したりする
            print("An error occurred: ", e)
            self._connect()  # 必要なら再接続を試みる

    def close(self):
        if self.connection:
            self.connection.close()


class FLServerSubscriber:
    def __init__(
        self,
        host="rabbitmq",
        queue_name="local_model_queue",
        username="guest",
        password="guest",
    ):
        self.host = host
        self.credentials = pika.PlainCredentials(
            username=username, password=password
        )
        self.queue_name = queue_name
        self.connection = None
        self.channel = None

        # モデル集約と重みのリスト
        self.local_models = []
        self.local_model_weights = []
        self.client_set = set()

        # 実験設定
        self.num_rounds = 10
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

            # ローカルのモデルおよびオプティマイザのインスタンスを作成
            local_model = SimpleModel()
            model_binary = pickle.loads(body)
            local_model.load_state_dict(model_binary)

            # ローカルモデルの重みを追加
            self.local_model_weights.append(local_model.state_dict())
            self.local_models.append(local_model)

            print(len(self.local_models))

            print(f" [x] Received model from client {client_id}")

            # 集約のタイミングをここで決める（例えば、5つのローカルモデルが集まったら）
            if len(self.local_models) >= 2:
                global_model = average_weights(self.local_model_weights)
                self.local_models.clear()
                self.local_model_weights.clear()
                self.client_set.clear()

                serialized_model = pickle.dumps(global_model)

                print(f" [x] Round {self.round} DONE")
                if self.round < self.num_rounds:
                    self.round += 1

                    # メッセージ送信
                    self.publisher.publish(serialized_model)
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


def load_global_model(global_model_path: str) -> nn.Module:
    # グローバルモデルの作成と初期化
    global_model = SimpleModel()
    global_model.load_state_dict(torch.load(global_model_path))
    print(f" [x] Loaded global model from {global_model_path}")
    return global_model


if __name__ == "__main__":
    global_model = SimpleModel()
    state_dict = global_model.state_dict()
    serialized_model = pickle.dumps(state_dict)

    publisher = FLServerPublisher()
    # メッセージ送信
    publisher.publish(serialized_model)
    subscriber = FLServerSubscriber()
    try:
        # メッセージの受信を開始
        subscriber.start_consuming()
    finally:
        # 終了時に接続を閉じる
        subscriber.close()
