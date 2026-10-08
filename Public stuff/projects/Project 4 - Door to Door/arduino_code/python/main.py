import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

BROKER_HOST = "test.mosquitto.org"
BROKER_PORT = 1883

TOPIC_CHANNELS = {
    "ME193/Will-Courtland/green": "green",
    "ME193/Will-Courtland/blue": "blue",
}


def on_connect(client, userdata, flags, reason_code, properties=None):
    for topic in TOPIC_CHANNELS:
        client.subscribe(topic, qos=0)


def on_message(client, userdata, msg):
    channel = TOPIC_CHANNELS.get(msg.topic)
    if channel is None:
        return
    payload = msg.payload.decode("utf-8", errors="replace").strip()
    if not payload:
        return
    # Forward the raw JSON straight to the sketch, which parses it with ArduinoJson.
    Bridge.notify(channel, payload)


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
mqtt_client.connect_async(BROKER_HOST, BROKER_PORT)
mqtt_client.loop_start()

App.run()
