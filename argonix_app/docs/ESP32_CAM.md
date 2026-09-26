# ESP32-CAM: second camera + PIR motion sensor

```
ESP32-CAM ── Wi-Fi 2.4 GHz ──► laptop running Argus
  OV2640  →  http://<esp-ip>:81/stream   (MJPEG)   → camera "esp32cam"
  PIR     →  http://<esp-ip>/pir         (JSON)    → sensor "esp32_pir", polled every 0.5 s
Arducam (USB) ──────────────────────────────────► camera "arducam"
```

Argus **pulls** both the stream and the PIR state. The dashboard can therefore stay local to the laptop (the default `--host 127.0.0.1`), and nothing on the network needs to reach it.

## 1. Wire the PIR (optional)

| PIR (e.g. HC-SR501) | ESP32-CAM |
|---|---|
| VCC | 5V |
| GND | GND |
| OUT | **GPIO13** |

Do not insert an SD card, because GPIO13 is shared with it. Set the HC-SR501's jumper to "repeat" (H) and turn its delay pot to the minimum.

## 2. Flash the firmware (Arduino IDE)

1. Copy `firmware/esp32cam_argonix/secrets.example.h` to `secrets.h` in the same folder and fill in your Wi-Fi name and password. **The network must be 2.4 GHz.** `secrets.h` is git-ignored.
2. Open `firmware/esp32cam_argonix/esp32cam_argonix.ino` in Arduino IDE.
3. Choose **Tools → Board → esp32 → AI Thinker ESP32-CAM**. **Tools → PSRAM → Enabled** is the default for this board.
4. Plug in the board:
   - **With the ESP32-CAM-MB USB base:** plug in USB, pick its COM port, and click **Upload**. If the upload hangs on "Connecting…", hold **IO0/BOOT** on the base and press **RST**.
   - **With a USB-serial (FTDI) adapter:** wire 5V→5V, GND→GND, TX→U0R and RX→U0T, and connect **IO0 to GND**. Upload, then remove the IO0–GND wire and press RST.
5. Open **Serial Monitor** at 115200 baud and press RST. It prints:
   `Connected. Camera stream: http://192.168.x.y:81/stream  PIR: http://192.168.x.y/pir`

The firmware compiles with the ESP32 Arduino core 3.3.9 (fqbn `esp32:esp32:esp32cam`): 32% of flash, 17% of RAM.

## 3. Point Argus at it

In `configs/arducam_esp32.json`, replace **both** `192.168.1.50` with the IP from the Serial Monitor. Check the Arducam's camera index too: it is `2` on this laptop while the phone is connected through Phone Link.

```bash
uv run --extra cpu python run.py --config configs/arducam_esp32.json --data data/run_esp32
```

Open http://127.0.0.1:8000. In `/api/stats` under `poll_sensors`, `esp32_pir` should show `online: true`. The poller is `argonix/poll_sensors.py` (config key `poll_sensors`); the signed door-controller webhooks in `argonix/sensors.py` are a separate feature.

## How the sensor is used

- A rising edge on the PIR (motion off → on, or the board's trigger counter going up between polls) becomes one `sensor_motion` event. It saves the ESP32-CAM's current frame as its snapshot.
- **On its own it is P4** (severity 1). It says "something warm moved here", never who.
- It shares `site_zone: "entrance"` with the ESP32-CAM's entrance zone. When a person is also detected there within 15 s, both events merge into one alert. The PIR adds a second, independent signal, so the alert's score rises.
- If the board goes offline, the sensor shows `online: false` with the error, and polling carries on until it returns.

## Test without the board

```bash
uv run python tools/sim_esp32cam.py --port 8090      # same /stream, /pir, /status as the firmware
# in the config use http://127.0.0.1:8090/stream and http://127.0.0.1:8090/pir
# open http://127.0.0.1:8090/pir/on to simulate motion, /pir/off to clear it
```

`tests/test_poll_sensors.py` runs this end to end: real pipeline, simulated node, stream frames, PIR trigger, then an alert with a snapshot.

## Limits

- ESP32-CAM video is 640×480 at roughly 10–20 fps over Wi-Fi. The quality is fine for room-scale person detection but weaker than the Arducam.
- The ESP's HTTP endpoints have no password. Keep the board on a private network such as Tailscale or a dedicated access point, never port-forwarded.
- A PIR sees heat and motion through a wide cone. It can be triggered by pets, heaters or sunlight, which is why it only adds weight to video and never raises a high-priority alert by itself.
