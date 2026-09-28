# Debugging

`index.html` is a standalone MQTT console: subscribe to and publish on any
topic, on any broker, straight from the browser — no install, no server.
Open it directly (double-click, or `file://.../index.html`) or serve it via
GitHub Pages.

It connects over MQTT-over-WebSockets using
[MQTT.js](https://github.com/mqttjs/MQTT.js) (loaded from a CDN), since
browsers can't speak raw MQTT/TCP. Defaults are prefilled for
`test.mosquitto.org` (WebSocket ports 8080 `ws` / 8081 `wss`) and topic
`ME193/minifig`, matching what
[`ClassTest/GreenMinifig_yolo/stream_minifig_mqtt.py`](../../ClassTest/GreenMinifig_yolo/stream_minifig_mqtt.py)
publishes — connect, subscribe, and you'll see that live camera-tracked
position stream in.

`jumbo_mascot.png` is the Jumbo-with-robot illustration used in the header
— original artwork made for this class, safe to publish here.
