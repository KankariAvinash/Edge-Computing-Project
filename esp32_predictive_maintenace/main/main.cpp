#include <string.h>
#include <sys/time.h>

#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
//#include "freertos/semphr.h"

#include "main_functions.h"
#include "esp_task_wdt.h"

#include "command_responder.h"

//SemaphoreHandle_t xSemaphore = NULL;

void audio_test_main(void) {
  while(true){
    pred_mant_audio();
  }
}


extern "C" void app_main() {
  //xSemaphore = xSemaphoreCreateBinary();
  //esp_task_wdt_init(10, true);
  xTaskCreate((TaskFunction_t)&audio_test_main, "audio_test",
                                        8 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}
