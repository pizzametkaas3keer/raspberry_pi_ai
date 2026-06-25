#!/bin/bash
# Tailscale troubleshooting script voor Raspberry Pi
# Run met: bash check_tailscale.sh

PORT=8765

echo "============================================================"
echo "  JARVIS - TAILSCALE TROUBLESHOOTING"
echo "============================================================"
echo ""

# 1. Is tailscale geïnstalleerd?
echo ">>> [1/8] Tailscale installatie check..."
if command -v tailscale &> /dev/null; then
    echo "    ✅ tailscale binary gevonden: $(which tailscale)"
    TS_VERSION=$(tailscale version 2>/dev/null | head -1)
    echo "    Versie: $TS_VERSION"
else
    echo "    ❌ tailscale NIET geïnstalleerd"
    echo ""
    echo "    Installeer met:"
    echo "      curl -fsSL https://tailscale.com/install.sh | sh"
    echo "============================================================"
    exit 1
fi
echo ""

# 2. Draait tailscaled?
echo ">>> [2/8] tailscaled service check..."
if systemctl is-active --quiet tailscaled 2>/dev/null; then
    echo "    ✅ tailscaled service is ACTIEF"
else
    echo "    ❌ tailscaled service is NIET actief"
    echo ""
    echo "    Start met:"
    echo "      sudo systemctl enable --now tailscaled"
    echo "============================================================"
    exit 1
fi
echo ""

# 3. Tailscale IP?
echo ">>> [3/8] Tailscale IP check..."
TS_IP=$(tailscale ip -4 2>/dev/null)
if [ -n "$TS_IP" ]; then
    echo "    ✅ Tailscale IPv4: $TS_IP"
else
    echo "    ❌ Geen Tailscale IP gevonden"
    echo ""
    echo "    Tailscale is geïnstalleerd maar niet verbonden met een netwerk."
    echo "    Verbind met:"
    echo "      sudo tailscale up --accept-routes"
    echo "    (Log in met je Tailscale account in je browser)"
    echo "============================================================"
    exit 1
fi
echo ""

# 4. Tailscale status (online?)
echo ">>> [4/8] Tailscale verbinding check..."
TS_STATE=$(tailscale status --json 2>/dev/null | grep -o '"BackendState": *"[^"]*"' | head -1 | cut -d'"' -f4)
echo "    Backend state: $TS_STATE"
if [ "$TS_STATE" = "Running" ]; then
    echo "    ✅ Tailscale is verbonden en draait"
else
    echo "    ⚠️  Tailscale backend is niet 'Running' (huidige state: $TS_STATE)"
    echo "    Probeer: sudo tailscale up"
fi
echo ""

# 5. Welke peers zijn online?
echo ">>> [5/8] Tailscale peers (apparaten in je tailnet)..."
tailscale status 2>/dev/null | head -20
echo ""

# 6. Luistert de Jarvis server op poort $PORT?
echo ">>> [6/8] Jarvis WebSocket server check (poort $PORT)..."
if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
    echo "    ✅ Iets luistert op poort $PORT:"
    ss -tlnp 2>/dev/null | grep ":$PORT"
else
    echo "    ❌ Niets luistert op poort $PORT"
    echo "    Start pi_app.py op de Pi:"
    echo "      cd ~/raspberry-ai && python3 pi_app.py"
    echo "============================================================"
    exit 1
fi
echo ""

# 7. Firewall check
echo ">>> [7/8] Firewall check (iptables/nftables/ufw)..."
if command -v ufw &> /dev/null && ufw status 2>/dev/null | grep -q "active"; then
    echo "    UFW is actief:"
    ufw status 2>/dev/null
    if ! ufw status 2>/dev/null | grep -q "$PORT"; then
        echo "    ⚠️  Poort $PORT niet in UFW rules!"
        echo "    Open met: sudo ufw allow $PORT/tcp"
    fi
elif iptables -L -n 2>/dev/null | grep -q "."; then
    IPTABLES_DROP=$(iptables -L INPUT -n 2>/dev/null | grep -c "DROP")
    if [ "$IPTABLES_DROP" -gt 0 ]; then
        echo "    ⚠️  iptables heeft DROP rules in INPUT chain:"
        iptables -L INPUT -n 2>/dev/null
        echo "    Overweeg: sudo iptables -A INPUT -p tcp --dport $PORT -j ACCEPT"
    else
        echo "    ✅ Geen DROP rules in iptables INPUT"
    fi
else
    echo "    ℹ️  Geen UFW of iptables DROP rules gevonden (waarschijnlijk OK)"
fi
echo ""

# 8. Lokale connectie test
echo ">>> [8/8] Lokale connectie test..."
echo "    Test 1: connectie via lokaal IP..."
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LOCAL_IP" ]; then
    echo "    Lokaal IP: $LOCAL_IP"
    if timeout 3 bash -c "echo > /dev/tcp/$LOCAL_IP/$PORT" 2>/dev/null; then
        echo "    ✅ Lokale connectie naar $LOCAL_IP:$PORT SUCCESVOL"
    else
        echo "    ❌ Lokale connectie naar $LOCAL_IP:$PORT FAALDE"
    fi
fi

echo ""
echo "    Test 2: connectie via Tailscale IP..."
if timeout 3 bash -c "echo > /dev/tcp/$TS_IP/$PORT" 2>/dev/null; then
    echo "    ✅ Tailscale connectie naar $TS_IP:$PORT SUCCESVOL"
else
    echo "    ❌ Tailscale connectie naar $TS_IP:$PORT FAALDE"
    echo "    Dit is waarschijnlijk de oorzaak — de server luistert niet op"
    echo "    het Tailscale interface."
    echo "    Oplossing: start pi_app.py met --host 0.0.0.0"
fi
echo ""

echo "============================================================"
echo "  SAMENVATTING"
echo "============================================================"
echo "  Tailscale IP van deze Pi: $TS_IP"
echo "  Jarvis server poort:      $PORT"
echo "  Verbind Android/Laptop met: ws://$TS_IP:$PORT"
echo ""
echo "  In de Android app → Instellingen → Verbinding:"
echo "    Tailscale IP: $TS_IP"
echo "============================================================"
