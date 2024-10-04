#ifndef AUDIO_TEST_MAIN_FUNCTIONS_H_
#define AUDIO_TEST_MAIN_FUNCTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdint>

void read_audio(int32_t last_time_in_ms, int32_t time_in_ms, int* how_many_new_slices);

void pred_mant_audio();

#ifdef __cplusplus
}
#endif

#endif  // AUDIO_TEST_MAIN_FUNCTIONS_H_
