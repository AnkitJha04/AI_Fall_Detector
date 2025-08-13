#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }
  Serial.println("ax,ay,az,gx,gy,gz"); // CSV header
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // Convert raw data to g and deg/sec
  float ax_g = ax / 16384.0;
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;
  float gx_dps = gx / 131.0;
  float gy_dps = gy / 131.0;
  float gz_dps = gz / 131.0;

  Serial.print(ax_g); Serial.print(",");
  Serial.print(ay_g); Serial.print(",");
  Serial.print(az_g); Serial.print(",");
  Serial.print(gx_dps); Serial.print(",");
  Serial.print(gy_dps); Serial.print(",");
  Serial.println(gz_dps);

  delay(20); // ~50 Hz sampling rate
}
