from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__)
socketio = SocketIO(app)

notifications = []

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/notifications')
def get_notifications():
    return jsonify(notifications)

@socketio.on('connect')
def handle_connect():
    emit('notification', {'data': 'Connected to real-time dashboard'})

# Example function to broadcast a notification
@app.route('/notify/<message>')
def notify(message):
    notifications.append({'message': message})
    socketio.emit('notification', {'data': message})
    return 'Notification sent'

if __name__ == '__main__':
    socketio.run(app, debug=True)
