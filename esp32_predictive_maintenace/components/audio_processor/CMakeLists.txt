idf_component_register(
    SRCS "command_responder.cpp"
         "audio_provider.cpp" "ringbuf.c"
    INCLUDE_DIRS include
    REQUIRES spi_flash driver esp_timer
)
