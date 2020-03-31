from flask import Flask, jsonify
import time

app = Flask(__name__)
startime = time.time()

@app.route("/")
def index():
	return jsonify({
		'status': 'running',
		'Uptime in seconds': time.time() - startime,
	})

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=80)