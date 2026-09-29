# mqttlib — MQTT publish/subscribe reference

`mqttlib.py` is a small wrapper around [paho-mqtt](https://pypi.org/project/paho-mqtt/) that makes it easy to publish and subscribe to an MQTT broker — by default the public [test.mosquitto.org](https://test.mosquitto.org/) broker, so two programs (or two laptops) can talk to each other without setting up a broker of your own.

```python
from mqttlib import MQTTClient

with MQTTClient() as client:
    client.publish("ME193", "hello world")
```

---

## MQTTClient(broker="test.mosquitto.org", port=1883, client_id="")

Create a client. Nothing connects to the network yet — call `connect()` (or use it as a context manager) first.

**Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `broker` | `"test.mosquitto.org"` | Broker hostname. Use `"localhost"` for a local Mosquitto install. |
| `port` | `1883` | Broker port (unencrypted MQTT). |
| `client_id` | `""` | MQTT client id. Leave blank to let the broker assign a random one. |

---

## connect(timeout=5)

Connect to the broker and start a background thread that handles all network traffic. Blocks until the broker confirms the connection (or `timeout` seconds pass), so any `subscribe()` call right after `connect()` is guaranteed to actually reach the broker.

## disconnect()

Stop the background thread and close the connection.

## publish(topic, message, qos=0, retain=False)

Publish `message` (a string) to `topic`.

## subscribe(topic, callback, qos=0)

Call `callback(topic, payload)` — both strings — every time a message arrives on `topic`. Each call to `subscribe()` replaces any previous callback registered for that exact topic string.

## Context manager

`with MQTTClient() as client:` calls `connect()` on entry and `disconnect()` on exit.

---

## Basic usage — publish

```python
from mqttlib import MQTTClient

with MQTTClient() as client:
    client.publish("ME193", "hello world")
```

## Basic usage — subscribe

```python
import time
from mqttlib import MQTTClient

def on_message(topic, payload):
    print(f"[{topic}] {payload}")

client = MQTTClient()
client.connect()
client.subscribe("ME193", on_message)

while True:
    time.sleep(0.1)
```

## Reading back your own publish

Subscribing and publishing in the same script confirms the round trip actually works, since the broker relays every publish back to any subscriber on that topic — including yourself:

```python
import time
from mqttlib import MQTTClient

def on_message(topic, payload):
    print(f"Got it back: [{topic}] {payload}")

with MQTTClient() as client:
    client.subscribe("ME193", on_message)
    time.sleep(1)  # give the subscription time to reach the broker
    client.publish("ME193", "hello world")
    time.sleep(1)  # give the message time to come back before disconnecting
```

---

## Notes

- **`test.mosquitto.org` is public and unauthenticated** — anyone can publish or subscribe to any topic. Don't send sensitive data, and use a specific topic name (not something generic like `test`) to avoid crosstalk with other users' traffic.
- **Point at a local broker instead** with `MQTTClient(broker="localhost")` — install one with `brew install mosquitto && brew services start mosquitto`.
- **Ports**, if you run your own broker or use a hosted one (e.g. HiveMQ):

  | Port | Meaning |
  |------|---------|
  | 1883 | Plain MQTT (unencrypted, no password required) |
  | 1884 | Plain MQTT (unencrypted, requires authentication) |
  | 8883 | Encrypted MQTT over TLS/SSL (no password required) |
  | 8884 | Encrypted MQTT over TLS/SSL (requires a custom client certificate) |
  | 8885 | Encrypted MQTT over TLS/SSL (requires authentication) |
  | 9001 | MQTT over WebSockets (unencrypted) |

- **QoS** (Quality of Service, `qos=` on `publish()`/`subscribe()`) controls delivery guarantees:

  | QoS | Guarantee | Cost |
  |-----|-----------|------|
  | 0 | At most once — fire and forget, message may be lost | Cheapest, lowest latency |
  | 1 | At least once — guaranteed delivery, but duplicates possible | One extra round-trip (PUBACK) |
  | 2 | Exactly once — guaranteed, no duplicates | Slowest, four-part handshake |

  Typical robotics control loops use `qos=0`: a dropped sensor reading or motor command is usually superseded by the next one a few milliseconds later anyway, so the extra latency/overhead of QoS 1/2 isn't worth it.

- **Retain** (`retain=` on `publish()`) controls whether the broker stores the last message on a topic for future subscribers:
  - **`retain=False` (the default)**: a sensor publishes `temperature = 22°C`. Ten minutes later, a dashboard connects and subscribes to that topic — it stays blank until the sensor's *next* publish.
  - **`retain=True`**: the broker stores a copy of that message. A dashboard connecting ten minutes later gets it delivered instantly on subscribe, without waiting for a new reading.

  Typical robotics control loops use no retain (`retain=False`): a stale retained command (e.g. "drive forward") delivered to a robot that just joined the topic could be actively dangerous.
