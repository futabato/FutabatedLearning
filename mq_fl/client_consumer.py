import pika
import pickle

# RabbitMQサーバーに接続
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

# キューを宣言
channel.queue_declare(queue='global_model')

def callback(ch, method, properties, body):
    global_model = pickle.loads(body)
    print(" [x] Received global model: %s" % global_model)

channel.basic_consume(queue='global_model', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for global model. To exit press CTRL+C')
channel.start_consuming()
