import pickle

import pika
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class FLClient:
    def __init__(
        self,
        host="rabbitmq",
        local_queue="local_model_queue",
        exchage_name="global_model_exchange",
        username="guest",
        password="guest",
    ):
        self.host = host
        self.credentials = pika.PlainCredentials(
            username=username, password=password
        )
        self.local_queue = local_queue
        self.exchange_name = exchage_name
        self.connection = None
        self.channel = None
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
            state_dict = pickle.loads(body)
            model = SimpleModel()
            model.load_state_dict(state_dict)
            print(" [x] Received initial global model")

            # ローカル学習を行い、モデルをサーバに送信
            self.send_local_model(model, client_id="1")

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

    def send_local_model(self, model: nn.Module, client_id: str):
        # Simulate local training (could be replaced with actual training logic)
        # local_train(model)

        serialized_model = pickle.dumps(model.state_dict())
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


if __name__ == "__main__":
    client = FLClient()
    try:
        # サーバからのグローバルモデルを待ち、それを受信
        client.receive_global_model()
    finally:
        # 終了時に接続を閉じる
        client.close()
