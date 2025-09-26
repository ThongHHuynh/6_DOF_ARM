#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();  // default I2C address 0x40

// Hiwonder 270° servo calibration (tune if needed)
#define SERVO_MIN1  300   // ~500 µs
#define SERVO_MAX1  2600  // ~2500 µs

#define SERVO_MIN2  500   // microseconds
#define SERVO_MAX2  2600  // microseconds
#define MAX_STEPS 100
float traj[MAX_STEPS][6];
int numSteps = 0;
int currentStep = 0;
unsigned long lastUpdate = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 ready: controlling servo 0 (Hiwonder 270°)");

  pwm.begin();
  pwm.setPWMFreq(50);  // 50 Hz = standard servo
  delay(10);

  // === Home position ===
  moveServo1(0, 135);    // MG996R (0–180)

  moveServo1(2, 135);  
  
  moveServo1(4, 135);  

  moveServo2(6, 90);  

  moveServo2(8, 90);  

  moveServo2(10, 90);  

  Serial.println("Servo 0 homed");
}

// void loop() {
//   // Optional: listen for new angle commands via serial
//   if (Serial.available() >=26) {
//     byte header = Serial.read();
//     if (header == 0xAA) {
//       float angles[6];
//       Serial.readBytes((char*)angles, 6 * sizeof(float));
//       byte footer = Serial.read();

//       if (footer ==0x55){
//         Serial.printf("Valid packet");
//         moveServo1(0, angles[0]); //move angle for HPS2027
//         moveServo1(2, angles[1]);  //move angle for HPS2027
//         moveServo1(4,angles[2]); //move angle for HPS2027
//         moveServo2(6, angles[3]); //move angle for mg996r
//         moveServo2(8, angles[4]);  //move angle for mg996r
//         moveServo2(10,angles[5]);  //move angle for mg996r
//       }
//     }
//   }
// }

void loop() {
  // Receive trajectory once
  if (Serial.available() >= 3) {
    byte header = Serial.read();
    if (header == 0xAA) {
      uint16_t steps;
      Serial.readBytes((char*)&steps, 2);
      if (steps > MAX_STEPS) steps = MAX_STEPS;

      for (int i=0; i<steps; i++) {
        Serial.readBytes((char*)traj[i], 6 * sizeof(float));
      }
      byte footer = Serial.read();
      if (footer == 0x55) {
        numSteps = steps;
        currentStep = 0;
        Serial.printf("Got trajectory with %d steps\n", numSteps);
      }
    }
  }

  // Play trajectory at fixed rate (e.g. 20 Hz = every 50ms)
  if (numSteps > 0 && millis() - lastUpdate >= 50) {
    lastUpdate = millis();
    float* a = traj[currentStep];

    moveServo1(0, a[0]);
    moveServo1(2, a[1]);
    moveServo1(4, a[2]);
    moveServo2(6, a[3]);
    moveServo2(8, a[4]);
    moveServo2(10, a[5]);

    currentStep++;
    if (currentStep >= numSteps) {
      numSteps = 0; // finished
    }
  }
}

// void loop() {
//   // Optional: listen for new angle commands via serial
//   if (Serial.available() >=14) {
//     String data = Serial.readStringUntil('\n');
//     data.trim();

//     float angle = data.toFloat();
//     if (angle >= 0 && angle <= 270) {
//       moveServo2(6, angle);
//       Serial.printf("Servo 0 moved to %.2f°\n", angle);
//     }
//   }
// }

// === Function for Hiwonder 270° servo ===
void moveServo1(uint8_t channel, float angle) {
  if (angle < 0) angle = 0;
  if (angle > 270) angle = 270;

  int pulse = map(angle, 0, 270, SERVO_MIN1, SERVO_MAX1);
  int pwmVal = pulse * 4096 / 20000;  // convert µs to PCA9685 steps
  pwm.setPWM(channel, 0, pwmVal);
}

void moveServo2(uint8_t channel, float angle) {

  // Map 0–270 deg to 500–2500us pulse width
  int pulse = map(angle, 0, 180, SERVO_MIN2, SERVO_MAX2);

  // Convert pulse (in microseconds) to PCA9685 12-bit value (out of 4096 steps at 20ms)
  int pwmVal = pulse * 4096 / 20000;

  pwm.setPWM(channel, 0, pwmVal);
}