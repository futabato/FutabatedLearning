import copy
import pickle

import pika
import torch
import torch.nn as nn


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
            print(f" [x] Received {body} from client_id: {client_id}")

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
                global_model = aggregate_models(self.local_model_weights)
                self.local_models.clear()
                self.local_model_weights.clear()
                self.client_set.clear()

                serialized_model = pickle.dumps(global_model)

                # メッセージ送信
                self.publisher.publish(serialized_model)

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


def aggregate_models(
    local_weights: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Averages the weights from multiple state dictionaries (each representing model parameters).

    Args:
        local_weights (list of dict): A list where each element is a state dictionary of model weights.

    Returns:
        A dict of the same structure as the input but with averaged weights.
    """
    # Initialize the averaged weights with deep copied weights from the first model
    weight_avg: dict[str, torch.Tensor] = copy.deepcopy(local_weights[0])

    # Iterate over each key in the weight dictionary
    for weight_key in weight_avg.keys():
        # Sum the corresponding weights from all models starting from the second one
        for weight_i in range(1, len(local_weights)):
            weight_avg[weight_key] += local_weights[weight_i][weight_key]
        # Divide the summed weights by the number of models to get the average
        weight_avg[weight_key] = torch.div(
            weight_avg[weight_key], len(local_weights)
        )
    print(" [x] Aggregated models and updated global model")
    # Return the averaged weights
    return weight_avg


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
