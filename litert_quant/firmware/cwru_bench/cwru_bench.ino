// Float versus int8 LiteRT Micro benchmark (CWRU bearing autoencoder) for the Arduino Nano 33 BLE.
// AI-assisted: drafted with Claude, checked by the author.
// Build with -DUSE_INT8 or -DUSE_FLOAT. Runs the test subset, times each inference
// with the DWT cycle counter and prints one line per window. The anomaly score is the
// mean squared error between the input and the reconstruction, computed after Invoke
// from the header input (the input tensor memory may be reused during Invoke).
// A window is flagged when its score is above the PC threshold in test_meta.h.
// Send 'r' over serial to run the whole subset again.
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_meta.h"

#if defined(USE_INT8)
#include "model_int8.h"
#include "test_input_int8.h"
#define MODEL_DATA g_model_int8
#define MODEL_LEN g_model_int8_len
#define MODEL_NAME "int8"
#define SCORE_THRESHOLD SCORE_THRESHOLD_INT8
#elif defined(USE_FLOAT)
#include "model_float.h"
#include "test_input_float.h"
#define MODEL_DATA g_model_float
#define MODEL_LEN g_model_float_len
#define MODEL_NAME "float"
#define SCORE_THRESHOLD SCORE_THRESHOLD_FLOAT
#else
#error "Define USE_INT8 or USE_FLOAT"
#endif

static volatile uint32_t* const kDemcr = (volatile uint32_t*)0xE000EDFC;
static volatile uint32_t* const kDwtCtrl = (volatile uint32_t*)0xE0001000;
static volatile uint32_t* const kDwtCyccnt = (volatile uint32_t*)0xE0001004;

constexpr int kArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static void cycle_counter_start() {
  *kDemcr |= (1UL << 24);
  *kDwtCyccnt = 0;
  *kDwtCtrl |= 1UL;
}

static float score_window(int base) {
  float sum = 0.0f;
#if defined(USE_INT8)
  const float in_scale = input->params.scale;
  const int in_zero = input->params.zero_point;
  const float out_scale = output->params.scale;
  const int out_zero = output->params.zero_point;
  for (int j = 0; j < TEST_SIZE; j++) {
    const float a = ((int)test_input_int8[base + j] - in_zero) * in_scale;
    const float b = ((int)output->data.int8[j] - out_zero) * out_scale;
    sum += (b - a) * (b - a);
  }
#else
  for (int j = 0; j < TEST_SIZE; j++) {
    const float d = output->data.f[j] - test_input_float[base + j];
    sum += d * d;
  }
#endif
  return sum / TEST_SIZE;
}

static void run_all() {
  Serial.print("START,");
  Serial.print(MODEL_NAME);
  Serial.print(",");
  Serial.println(TEST_COUNT);
  unsigned int correct = 0;
  for (int i = 0; i < TEST_COUNT; i++) {
    const int base = i * TEST_SIZE;
#if defined(USE_INT8)
    for (int j = 0; j < TEST_SIZE; j++) input->data.int8[j] = test_input_int8[base + j];
#else
    for (int j = 0; j < TEST_SIZE; j++) input->data.f[j] = test_input_float[base + j];
#endif
    noInterrupts();
    uint32_t t0 = *kDwtCyccnt;
    TfLiteStatus st = interpreter->Invoke();
    uint32_t t1 = *kDwtCyccnt;
    interrupts();
    if (st != kTfLiteOk) {
      Serial.println("ERROR,invoke");
      return;
    }
    const float score = score_window(base);
    const int flagged = score > SCORE_THRESHOLD ? 1 : 0;
    if (flagged == test_labels[i]) correct++;
    Serial.print("S,");
    Serial.print(i);
    Serial.print(",");
    Serial.print(test_labels[i]);
    Serial.print(",");
    Serial.print(score, 6);
    Serial.print(",");
    Serial.println((unsigned long)(t1 - t0));
  }
  Serial.print("DONE,");
  Serial.print(MODEL_NAME);
  Serial.print(",");
  Serial.print((unsigned long)interpreter->arena_used_bytes());
  Serial.print(",");
  Serial.print(TEST_COUNT);
  Serial.print(",");
  Serial.print(correct);
  Serial.print(",");
  Serial.println((unsigned long)MODEL_LEN);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }
  cycle_counter_start();
  const tflite::Model* model = tflite::GetModel(MODEL_DATA);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR,schema");
    while (true) {
    }
  }
  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR,allocate");
    while (true) {
    }
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  run_all();
}

void loop() {
  if (Serial.available() && Serial.read() == 'r') run_all();
}
