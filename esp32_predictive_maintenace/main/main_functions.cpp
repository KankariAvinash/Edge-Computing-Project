#include "main_functions.h"

#include "esp_mac.h"
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
//#include "freertos/semphr.h"

#include "micro_model_settings.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "esp_task_wdt.h"
#include "driver/temp_sensor.h"
#include "bsp/esp32_s3_eye.h"

// Ensure you have the display initialized somewhere in your setup
extern "C" void setup_display();

//extern SemaphoreHandle_t xSemaphore;


#define FEATURES_MEM_OVERFLOW -5

static int get_signal_data(size_t offset, size_t length, float *out_ptr);

static const int32_t features_size = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
static int16_t features[features_size] = {0};

static bool is_first_run_ = true;


void setup_temperature(){
  ei_printf("Install temperature sensor");
  temp_sensor_config_t temp_sensor_config = TSENS_CONFIG_DEFAULT();
  temp_sensor_get_config(&temp_sensor_config);
  temp_sensor_config.dac_offset = TSENS_DAC_DEFAULT;
  temp_sensor_set_config(temp_sensor_config);
  temp_sensor_start();
}



void set_input(int audio_samples_size, int16_t** audio_samples){
  //ei_printf("make space for storing new audio samples\r\n");
  for(int32_t i=0; i<features_size-audio_samples_size; i++){
    features[i] = features[i+audio_samples_size];
  }
  esp_task_wdt_reset();

  //ei_printf("set audio samples at the end of stream\r\n");
  for(int32_t i=0; i<audio_samples_size; i++){
    // assign from audio samples to features
    features[features_size-audio_samples_size+i] = (*audio_samples)[i];
  }
}


void read_audio(int32_t last_time_in_ms, int32_t time_in_ms, int* how_many_new_slices){
  const int last_step = (last_time_in_ms / kFeatureStrideMs);
  const int current_step = (time_in_ms / kFeatureStrideMs);
  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    is_first_run_ = false;
    slices_needed = kFeatureCount;
  }
  if (slices_needed > kFeatureCount) {
    slices_needed = kFeatureCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureCount - slices_needed;
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureCount;
                                ++new_slice) {
        // start_duration and duration_ms parameters not used
        int audio_samples_size = 0; // 512 when printed
        int16_t* audio_samples = nullptr;
        GetAudioSamples(0, 0, &audio_samples_size, &audio_samples);
        set_input(audio_samples_size, &audio_samples);
        esp_task_wdt_reset();
    }
  }
  esp_task_wdt_reset();
}


void pred_mant_audio(){
    setup_styles();
    setup_temperature();

    signal_t signal;           
    ei_impulse_result_t result;
    EI_IMPULSE_ERROR res;
    //features = new int16_t(features_size);

    // initialize features
    /*ei_printf("initialize features to 0\r\n");
    for(uint16_t i=0; i<features_size; i++){
      features[i]=0;
    }*/

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;

    // setup lcd

    int prev_command = -1;
    int found_command = -1;
    int32_t previous_time=0;

    float tsens_value;

    while(true){
        const int32_t current_time = LatestAudioTimestamp();
        ei_printf("time: %d\r\n",current_time);

        // read temperature
        ei_printf("Read temperature");
        temp_sensor_read_celsius(&tsens_value);

        // read sound
        int how_many_new_slices = 0;
        read_audio(previous_time, current_time, &how_many_new_slices);
        ei_printf("audio successfully read\r\n");
        previous_time = current_time;
        if (how_many_new_slices == 0){ return; }

        //xSemaphoreTake(xSemaphore, portMAX_DELAY);
        res = run_classifier(&signal, &result, false);
        esp_task_wdt_reset();
        //xSemaphoreGive(xSemaphore);

        ei_printf("run_classifier returned: %d\r\n", res);
        ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
                result.timing.dsp,
                result.timing.classification,
                result.timing.anomaly);

        ei_printf("Predictions:\r\n");
        float max_score = 0;
        found_command = -1;
        for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
            ei_printf("%.5f\r\n", result.classification[i].value);
            if(result.classification[i].value > max_score){
              max_score = result.classification[i].value;
              found_command = i;
            }
        }
        //ei_printf("command found");
        esp_task_wdt_reset();

        //float max_score = 0;
        //found_command = !found_command;
        bool is_new_command = prev_command != found_command;
        if(is_new_command){
            RespondToCommand(found_command, is_new_command,
                              max_score, tsens_value);
            prev_command = found_command;
        }
        //ei_printf("after respond to command\r\n");
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}


static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        if(offset + i >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE){
            return FEATURES_MEM_OVERFLOW;
        }
        out_ptr[i] = (features + offset)[i];
    }

    return EIDSP_OK;
}