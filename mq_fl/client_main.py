import pika

credentials = pika.PlainCredentials("guest", "guest")
pika_param = pika.ConnectionParameters(host="rabbitmq", credentials=credentials)
connection = pika.BlockingConnection(pika_param)
channel = connection.channel()

channel.queue_declare(queue='hello')

channel.basic_publish(exchange='', routing_key='hello', body='Hello, World')

connection.close()