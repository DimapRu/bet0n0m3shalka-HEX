# -*- coding: utf-8 -*-
"""E2E с динамическим SNR: старт 30 дБ (16QAM), после 2.5 с спад до 8 дБ.
Ожидание: RX увидит деградацию на пробнике и переключит TX на FSK (T_SWITCH)."""
import sys, threading, time, os
sys.path.insert(0, 'app')
import modem as md

test_path = 'app/_e2e_test.bin'
payload = bytes(range(256)) * 3 + b'END'
open(test_path, 'wb').write(payload)

chan_rx = md.Channel(loopback=True, loop_snr=30.0, role='rx')
chan_tx = md.Channel(loopback=True, loop_snr=30.0, role='tx')

def degrade():
    time.sleep(2.5)
    chan_rx.loop_snr = 8.0
    chan_tx.loop_snr = 8.0
    print('\n=== КАНАЛ ДЕГРАДИРОВАЛ: SNR 30 -> 8 дБ ===')

t = threading.Thread(target=md.cmd_recv, args=(chan_rx, 1), daemon=True)
t.start()
threading.Thread(target=degrade, daemon=True).start()
time.sleep(0.5)
md.cmd_send(chan_tx, test_path, pair=1)
t.join(timeout=180)

print('=== DONE ===')
d = 'app/received'
if os.path.isdir(d):
    for f in os.listdir(d):
        got = open(os.path.join(d, f), 'rb').read()
        print('RX file:', f, len(got), 'match:', got == payload)
