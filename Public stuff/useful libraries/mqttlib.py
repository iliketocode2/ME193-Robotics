'''
mqttlib.py -- small MQTT publish/subscribe wrapper around paho-mqtt.

See MQTTLIB.md for the full reference (MQTTClient's constructor,
connect/disconnect, publish/subscribe, and example usage).

Usage:
    from mqttlib import MQTTClient

    with MQTTClient() as client:
        client.publish("ME193", "hello world")
'''

import threading

import paho.mqtt.client as mqtt


class MQTTClient:
    '''A small wrapper around paho-mqtt for publish/subscribe over MQTT.

    Defaults to the public test.mosquitto.org broker, so two programs
    (or two laptops) can talk to each other with no broker of your own
    -- see MQTTLIB.md's security notes before publishing anything
    sensitive there.
    '''

    def __init__(self, broker='test.mosquitto.org', port=1883, client_id=''):
        self.broker = broker
        self.port = port
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._client.on_connect = self._on_connect
        self._connected_event = threading.Event()

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties):
        self._connected_event.set()

    def connect(self, timeout=5):
        '''Connect to the broker and start a background thread that
        handles all network traffic. Blocks until the broker confirms
        the connection (or `timeout` seconds pass), so any subscribe()
        call right after connect() is guaranteed to actually reach the
        broker.'''
        self._connected_event.clear()
        self._client.connect(self.broker, self.port)
        self._client.loop_start()
        if not self._connected_event.wait(timeout):
            self._client.loop_stop()
            raise TimeoutError(
                f'Could not connect to MQTT broker {self.broker}:{self.port} '
                f'within {timeout}s'
            )

    def disconnect(self):
        '''Stop the background thread and close the connection.'''
        self._client.disconnect()
        self._client.loop_stop()

    def publish(self, topic, message, qos=0, retain=False):
        '''Publish `message` (a string) to `topic`.'''
        self._client.publish(topic, message, qos=qos, retain=retain)

    def subscribe(self, topic, callback, qos=0):
        '''Call `callback(topic, payload)` -- both strings -- every time
        a message arrives on `topic`. Each call to subscribe() replaces
        any previous callback registered for that exact topic string.'''
        def _on_topic_message(client, userdata, message):
            callback(message.topic, message.payload.decode())
        self._client.message_callback_add(topic, _on_topic_message)
        self._client.subscribe(topic, qos=qos)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
