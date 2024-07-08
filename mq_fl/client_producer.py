import pika
import pickle

# 学習済みモデルのダミーデータ（例）
local_model = {'weights': [1, 2, 3]}

# RabbitMQサーバーに接続
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

# キューを宣言
channel.queue_declare(queue='local_models')

# モデルをシリアライズして送信
channel.basic_publish(exchange='', routing_key='local_models', body=pickle.dumps(local_model))
print(" [x] Sent local model")

connection.close()
