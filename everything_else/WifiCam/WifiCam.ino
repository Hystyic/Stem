#include <esp_now.h>
#include <WiFi.h>

// Structure to hold the received data
struct Data {
  int id;
  int value;
};

// Callback function that will be called when data is received
void OnDataReceived(const uint8_t* mac, const uint8_t* data, int dataLen) {
  // Get the data from the received packet
  Data receivedData;
  memcpy(&receivedData, data, sizeof(receivedData));

  // Print the data
  Serial.print("Received data from: ");
  Serial.print(mac[0], HEX);
  Serial.print(mac[1], HEX);
  Serial.print(mac[2], HEX);
  Serial.print(mac[3], HEX);
  Serial.print(mac[4], HEX);
  Serial.print(mac[5], HEX);
  Serial.print(", id: ");
  Serial.print(receivedData.id);
  Serial.print(", value: ");
  Serial.println(receivedData.value);
}

void setup() {
  // Initialize Wi-Fi
  WiFi.mode(WIFI_STA);

  // Initialize the ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register the callback function that will be called when data is received
  esp_now_register_recv_cb(OnDataReceived);

  // Start Serial communication
  Serial.begin(115200);
}

void loop() {
  // Your loop code here (if needed)
}
