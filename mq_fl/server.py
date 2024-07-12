import pika
import pickle

# RabbitMQサーバーに接続
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

# キューを宣言
channel.queue_declare(queue='local_models')

# 集約するローカルモデルの数
REQUIRED_LOCAL_MODELS = 2
received_models = []

def callback(ch, method, properties, body):
    local_model = pickle.loads(body)
    print(" [x] Received local model: %s" % local_model)
    received_models.append(local_model)
    # ここでローカルモデルを集約し、グローバルモデルを作成するロジックを追加
    # ダミーのグローバルモデルを送信
    if len(received_models) >= REQUIRED_LOCAL_MODELS:
        global_model = aggregate_models(received_models)
        received_models.clear()
        send_global_model(global_model)

def aggregate_models(models):
    # ここにモデルの集約ロジックを実装
    # 以下はダミーの集約ロジックです
    aggregated_weights = [sum(x) / len(models) for x in zip(*[model['weights'] for model in models])]
    return {'weights': aggregated_weights}

def send_global_model(global_model):
    global_connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
    global_channel = global_connection.channel()
    global_channel.queue_declare(queue='global_model')
    global_channel.basic_publish(exchange='', routing_key='global_model', body=pickle.dumps(global_model))
    print(" [x] Sent global model")
    global_connection.close()

channel.basic_consume(queue='local_models', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for local models. To exit press CTRL+C')
channel.start_consuming()
