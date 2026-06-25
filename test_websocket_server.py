#!/usr/bin/env python3
"""
Eenvoudige WebSocket test server voor Windows
Installeer eerst: pip install websocket-server
"""

from websocket_server import WebsocketServer

def new_client(client, server):
    """Nieuwe client verbonden"""
    print(f"🟢 Nieuwe client verbonden: {client['address']}")
    server.send_message_to_all("Hallo! Dit is Jarvis test server.")

def client_left(client, server):
    """Client verbroken"""
    print(f"🔴 Client verbroken: {client['address']}")

def message_received(client, server, message):
    """Bericht ontvangen van client"""
    print(f"💬 Bericht ontvangen: {message}")
    server.send_message_to_all(f"Server ontvang: {message}")

if __name__ == "__main__":
    # Server instellingen
    HOST = "0.0.0.0"  # Luistert op alle interfaces
    PORT = 8765

    print(f"🚀 WebSocket server start op {HOST}:{PORT}")
    print(f"📱 Android app kan verbinden met: ws://JOUW_WINDOWS_IP:{PORT}")

    # Server starten
    server = WebsocketServer(host=HOST, port=PORT)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)

    try:
        server.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server gestopt")
