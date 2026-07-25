# -*- coding: utf-8 -*-
import sys, threading, time, os
sys.path.insert(0, 'app')
import modem as md

test_path = 'app/_e2e_test.bin'
payload = bytes(range(256)) * 3 + b'END'
open(test_path, 'wb').write(payload)

snr = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
chan_rx = md.Channel(loopback=True, loop_snr=snr, role='rx')
chan_tx = md.Channel(loopback=True, loop_snr=snr, role='tx')

t = threading.Thread(target=md.cmd_recv, args=(chan_rx, 1), daemon=True)
t.start()
time.sleep(0.5)
md.cmd_send(chan_tx, test_path, pair=1)
t.join(timeout=120)

print('=== DONE ===')
d = 'app/received'
if os.path.isdir(d):
    for f in os.listdir(d):
        got = open(os.path.join(d, f), 'rb').read()
        print('RX file:', f, len(got), 'match:', got == payload)
