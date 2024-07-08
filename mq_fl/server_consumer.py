import pika
import pickle

# RabbitMQサーバーに接続
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

# キューを宣言
channel.queue_declare(queue='local_models')

def callback(ch, method, properties, body):
    local_model = pickle.loads(body)
    print(" [x] Received local model: %s" % local_model)
    # ここでローカルモデルを集約し、グローバルモデルを作成するロジックを追加
    # ダミーのグローバルモデルを送信
    # global_model = {'weights': [4, 5, 6]}
    global_model = aggregate(local_model)
    send_global_model(global_model)

def aggregate(local_model):
    global_model = local_model.copy()
    for i, weight in enumerate(local_model['weights']):
        global_model['weights'][i] = weight * 2
    return global_model

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
