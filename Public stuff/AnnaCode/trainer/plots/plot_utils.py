import js
import json

def to_js(obj):
    return js.JSON.parse(json.dumps(obj))
