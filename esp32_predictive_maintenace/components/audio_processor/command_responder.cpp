/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "command_responder.h"
//#include "tensorflow/lite/micro/micro_log.h"
#include "lvgl.h"
#include <cstdio>
#include "bsp/esp32_s3_eye.h"

#define MAX_TEMP 30

// Ensure you have the display initialized somewhere in your setup
extern "C" void setup_display();

// TODO 1: Create styles for different background colors ---------------------
static lv_style_t style_bg_red;
static lv_style_t style_bg_green;
static lv_style_t style_bg_blue;
static lv_style_t style_bg_orange;

        
// END TODO 1 ----------------------------------------------------------------

void setup_styles()
{   
    bsp_display_start();
    bsp_display_backlight_on();
    // TODO 2 : Initialize styles --------------------------------------------
    lv_style_init(&style_bg_red);
    lv_style_init(&style_bg_green);
    lv_style_init(&style_bg_blue);
    lv_style_init(&style_bg_orange);

    lv_style_set_bg_color(&style_bg_red, lv_color_hex(0xFF0000));
    lv_style_set_bg_color(&style_bg_green, lv_color_hex(0x00FF00));
    lv_style_set_bg_color(&style_bg_blue, lv_color_hex(0x0000FF));
    lv_style_set_bg_color(&style_bg_orange, lv_color_hex(0xFFA500));


    // END TODO 2 ------------------------------------------------------------
}

void RespondToCommand(int found_command, bool is_new_command, 
                        float score, float tsens_value)
{
    bsp_display_lock(0);
    if (is_new_command)
    {
        //MicroPrintf("Heard %d (%.4f) ms", found_command, score);

        // Display the recognized command on the LCD
        static lv_obj_t *my_label = nullptr;
        static lv_obj_t *screen = nullptr;
        if (my_label == nullptr)
        {
            screen = lv_scr_act();
            my_label = lv_label_create(screen);

            // Set label size to avoid overflow
            lv_obj_set_width(my_label, LV_HOR_RES - 20);

            // Enable word wrap and auto resize
            lv_label_set_long_mode(my_label, LV_LABEL_LONG_WRAP);

            // Set text style to increase size
            static lv_style_t style;
            lv_style_init(&style);

            // TODO 3: Set font size --------------------------------------------
            lv_style_set_text_font(&style, &lv_font_montserrat_14);
            // END TODO 3 -------------------------------------------------------

            lv_style_set_text_align(&style, LV_TEXT_ALIGN_CENTER);
            lv_obj_add_style(my_label, &style, 0);
        }
        else
        {
            lv_label_set_text(my_label, ""); // Clear previous text
        }

        // Convert the score to a string
        char score_str[32];
        snprintf(score_str, sizeof(score_str), "%.2f", score);

        // Convert temperature to string
        char temp_msg[64];
        if(tsens_value <= MAX_TEMP){
            snprintf(temp_msg, sizeof(temp_msg), 
            "Temperature: %.2f", tsens_value);
        }
        else{
            snprintf(temp_msg, sizeof(temp_msg), 
            "Temperature: %.2f\nTemperature above MAX value!", 
            tsens_value);
        }

        // TODO 4 : Create the final string to display -------------------------
        char display_str[128];
        if(found_command == 1){
            snprintf(display_str, sizeof(display_str), 
                    "Correct behavior\nScore: %s\n\n%s",
                    score_str, temp_msg);
        }
        else{
            snprintf(display_str, sizeof(display_str), 
                    "Anomaly detected\nScore: %s\n\n%s",
                    score_str, temp_msg);
        }
        // END TODO 4 ----------------------------------------------------------

        // Set the text of the label
        lv_label_set_text(my_label, display_str);
        lv_obj_align(my_label, LV_ALIGN_CENTER, 0, 0); // Center the my_label

        // TODO 5: Remove previous background style -----------------------------
        lv_obj_remove_style(screen, &style_bg_red, LV_PART_MAIN);
        lv_obj_remove_style(screen, &style_bg_green, LV_PART_MAIN);
        lv_obj_remove_style(screen, &style_bg_blue, LV_PART_MAIN);
        lv_obj_remove_style(screen, &style_bg_orange, LV_PART_MAIN);

        // END TODO 5 -----------------------------------------------------------

        // TODO 6: Change background color based on the prediction result -------
        //MicroPrintf("React to command");
        if(found_command == 1 && tsens_value <= MAX_TEMP){
            lv_obj_add_style(screen, &style_bg_green, LV_PART_MAIN);
        }
        else if(found_command == 1 && tsens_value > MAX_TEMP){
            lv_obj_add_style(screen, &style_bg_blue, LV_PART_MAIN);
        }
        else if(found_command != 1 && tsens_value <= MAX_TEMP){
            lv_obj_add_style(screen, &style_bg_orange, LV_PART_MAIN);
        }
        else{
            lv_obj_add_style(screen, &style_bg_red, LV_PART_MAIN);
        }

        // END TODO 6 -----------------------------------------------------------

        // Force display update
        lv_refr_now(NULL);
    }
    bsp_display_unlock();
}
