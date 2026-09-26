// Argus ESP32-CAM node (AI-Thinker ESP32-CAM): MJPEG camera stream + PIR motion sensor over Wi-Fi.
//
//   http://<ip>:81/stream   MJPEG stream  -> Argus camera source
//   http://<ip>/capture     one JPEG
//   http://<ip>/pir         {"motion":true|false,"count":N,"last_change_ms":..,"uptime_ms":..} -> Argus sensor poll
//   http://<ip>/status      {"ip":..,"rssi":..,"frame":"VGA","psram":true,...}
//
// Argus PULLS both (camera stream + /pir), so the laptop's dashboard never has to be reachable from the network.
// Wiring: PIR (e.g. HC-SR501) VCC -> 5V, GND -> GND, OUT -> GPIO13. Do not use an SD card (GPIO13 is shared with it).
// Wi-Fi: copy secrets.example.h to secrets.h and fill in the SSID/password (2.4 GHz only). secrets.h is git-ignored.
// Board: "AI Thinker ESP32-CAM" (fqbn esp32:esp32:esp32cam), PSRAM enabled.
#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include "secrets.h"

// ---- AI-Thinker ESP32-CAM pin map --------------------------------------------------------------------------------
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define PIR_PIN 13
#define FLASH_LED_PIN 4

static volatile bool g_motion = false;
static volatile uint32_t g_count = 0;
static volatile uint32_t g_last_change_ms = 0;

static httpd_handle_t web_httpd = NULL;
static httpd_handle_t stream_httpd = NULL;

// ---- PIR ------------------------------------------------------------------------------------------------------------
static void IRAM_ATTR on_pir_change() {
  bool m = digitalRead(PIR_PIN) == HIGH;
  if (m && !g_motion) g_count++;
  g_motion = m;
  g_last_change_ms = millis();
}

// ---- HTTP handlers --------------------------------------------------------------------------------------------------
static esp_err_t pir_handler(httpd_req_t *req) {
  char body[160];
  snprintf(body, sizeof(body), "{\"motion\":%s,\"count\":%lu,\"last_change_ms\":%lu,\"uptime_ms\":%lu}",
           g_motion ? "true" : "false", (unsigned long)g_count, (unsigned long)g_last_change_ms,
           (unsigned long)millis());
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, body, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *req) {
  char body[256];
  sensor_t *s = esp_camera_sensor_get();
  snprintf(body, sizeof(body),
           "{\"ip\":\"%s\",\"rssi\":%d,\"framesize\":%d,\"quality\":%d,\"psram\":%s,\"uptime_ms\":%lu,\"pir_pin\":%d}",
           WiFi.localIP().toString().c_str(), WiFi.RSSI(), s ? s->status.framesize : -1,
           s ? s->status.quality : -1, psramFound() ? "true" : "false", (unsigned long)millis(), PIR_PIN);
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, body, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

#define PART_BOUNDARY "argonixframe"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  char part[64];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
      break;
    }
    size_t hlen = snprintf(part, sizeof(part), STREAM_PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;  // client (Argus) disconnected; it will reconnect by itself
  }
  return res;
}

static void start_servers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  httpd_uri_t pir_uri = {.uri = "/pir", .method = HTTP_GET, .handler = pir_handler, .user_ctx = NULL};
  httpd_uri_t status_uri = {.uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL};
  httpd_uri_t capture_uri = {.uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL};
  if (httpd_start(&web_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(web_httpd, &pir_uri);
    httpd_register_uri_handler(web_httpd, &status_uri);
    httpd_register_uri_handler(web_httpd, &capture_uri);
  }
  // the stream blocks its server task, so it gets its own server on port 81
  config.server_port = 81;
  config.ctrl_port = 32769;
  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ---- setup / loop ---------------------------------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.println();
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always hand out the newest frame (no lag build-up)
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;  // 640x480: enough for person detection at room scale, ~10-20 fps
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed: check the ribbon cable and that the board is an AI-Thinker ESP32-CAM");
    delay(3000);
    ESP.restart();
  }

  pinMode(PIR_PIN, INPUT_PULLDOWN);
  g_motion = digitalRead(PIR_PIN) == HIGH;
  attachInterrupt(digitalPinToInterrupt(PIR_PIN), on_pir_change, CHANGE);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // Wi-Fi power save causes stream stalls
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - t0 > 30000) {
      Serial.println("\nWi-Fi failed (2.4 GHz only; check SSID/password in secrets.h). Restarting.");
      ESP.restart();
    }
  }
  Serial.printf("\nConnected. Camera stream: http://%s:81/stream  PIR: http://%s/pir\n",
                WiFi.localIP().toString().c_str(), WiFi.localIP().toString().c_str());
  start_servers();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {  // reconnect if the access point drops
    WiFi.reconnect();
    delay(5000);
  }
  delay(1000);
}
