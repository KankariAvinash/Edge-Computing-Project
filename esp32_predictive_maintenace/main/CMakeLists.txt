cmake_minimum_required(VERSION 3.13.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODEL_FOLDER ..)
set(EI_SDK_FOLDER ../edge-impulse-sdk)

if(NOT CMAKE_BUILD_EARLY_EXPANSION)
add_definitions(-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=1) # enables ESP-NN optimizations by Espressif
endif()

set(include_dirs
    ${MODEL_FOLDER}
    ${MODEL_FOLDER}/tflite-model
    ${MODEL_FOLDER}/model-parameters
    ${EI_SDK_FOLDER}
    ${EI_SDK_FOLDER}/classifier
    ${EI_SDK_FOLDER}/dsp
    ${EI_SDK_FOLDER}/porting
    ${EI_SDK_FOLDER}/porting/espressif/ESP-NN
    ${EI_SDK_FOLDER}/porting/espressif/ESP-NN/include
    ${EI_SDK_FOLDER}/tensorflow
    ${EI_SDK_FOLDER}/third_party
    ${EI_SDK_FOLDER}/third_party/ruy
    ${EI_SDK_FOLDER}/third_party/gemmlowp
    ${EI_SDK_FOLDER}/third_party/flatbuffers/include
)

include(${EI_SDK_FOLDER}/cmake/utils.cmake)

RECURSIVE_FIND_FILE_EXCLUDE_DIR(SOURCE_FILES "../edge-impulse-sdk" "CMSIS" "*.cpp")
RECURSIVE_FIND_FILE_EXCLUDE_DIR(CC_FILES "../edge-impulse-sdk" "CMSIS" "*.cc")
RECURSIVE_FIND_FILE_EXCLUDE_DIR(S_FILES "../edge-impulse-sdk" "CMSIS" "*.S")
RECURSIVE_FIND_FILE_EXCLUDE_DIR(C_FILES "../edge-impulse-sdk" "CMSIS" "*.c")

# Manually handle needed C only files b/c of build errors
list(APPEND C_FILES "../edge-impulse-sdk/CMSIS/DSP/Source/StatisticsFunctions/arm_rms_f32.c")

RECURSIVE_FIND_FILE(PORTING_FILES "../edge-impulse-sdk/porting/espressif" "*.cpp")

RECURSIVE_FIND_FILE(MODEL_FILES "../tflite-model" "*.cpp")


list(APPEND SOURCE_FILES ${S_FILES})
list(APPEND SOURCE_FILES ${C_FILES})
list(APPEND SOURCE_FILES ${CC_FILES})

list(APPEND SOURCE_FILES ${PORTING_FILES})

list(APPEND SOURCE_FILES ${MODEL_FILES})

idf_component_register(SRCS "main.cpp" "main_functions.cpp" 
                            "${SOURCE_FILES}"
                       INCLUDE_DIRS "include" "${include_dirs}")

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-maybe-uninitialized
    -Wno-missing-field-initializers
    -Wno-error=sign-compare
    -Wno-error=double-promotion
    -Wno-type-limits)