/*
Copyright (c) 2019 SparkFun Electronics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
Gesture demo for the SparkFun Edge (LIS2DH12): reads the accelerometer, runs the whole gesture chain on the chip
(gesture_pipeline.c: resample to 50 Hz, 128-sample windows, features, trained model) and prints one line per window.

Derived from this repository's imu_logger/main.c (itself derived from SparkFun's lis2dh12_accelerometer_uart example, MIT
licence, notice above). The sensor is set up exactly as the logger sets it up, so the samples look like the recordings the
model was trained on. Differences:
  * the timer is always the internal 3 MHz oscillator (the recordings were all timed with it, and it has a fine tick for
    timing the chain; the logger's 32 kHz crystal path is not used)
  * no per-sample output: one line per completed window, which is every 64 samples (1.28 s: 64 samples of the 50 Hz grid)

Output at 115200 baud. Lines starting with '#' are comments:
  # sparkfun_edge gesture demo v1, whoami 0x33
  # odr_hz=50 tick_hz=3000000 timer=hfrc3mhz window=128 step=64 features=40 classes=rest,shake_lr,shake_ud,circle,tilt
  # columns: G,window,t_ticks,gesture,compute_us
  G,1,123456789,rest,3120
  ...
  # stats windows=32 overruns=0 i2c_err=0 restarts=0 ignored=0 max_compute_us=3180
compute_us is the time gp_push() took for the sample that completed the window (resampling, features and the model).
overruns count samples the sensor replaced before they were read; with the chain running at 50 Hz there should be none.
*/

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

#include "lis2dh12_platform_apollo3.h"

#include "gesture_pipeline.h"

#define ODR_HZ 50
#define TICK_HZ 3000000u                       /* AM_HAL_STIMER_HFRC_3MHZ */
#define TICKS_PER_US (TICK_HZ / 1000000u)
#define STATS_EVERY_WINDOWS 20u

static axis3bit16_t data_raw_acceleration;
static gp_state g_gp;                          /* the chain's whole state, about 1.8 KB */

static lis2dh12_platform_apollo3_if_t dev_if = {
    .iomHandle  = NULL,                             // initialised in initAccel()
    .addCS      = AM_BSP_ACCELEROMETER_I2C_ADDRESS, // the accelerometer's I2C address on this board
    .useSPI     = false,                            // I2C
};

static lis2dh12_ctx_t dev_ctx = {
    .write_reg  = lis2dh12_write_platform_apollo3,
    .read_reg   = lis2dh12_read_platform_apollo3,
    .handle     = (void*)&dev_if,
};

// The free-running timer that stamps the samples: the internal 3 MHz oscillator, the same one every recording used.
static void timer_init(void)
{
    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_stimer_config(AM_HAL_STIMER_HFRC_3MHZ);
}

static uint32_t initAccel(uint8_t *whoami_out)
{
    uint32_t retVal32 = 0;
    uint8_t whoamI = 0;

    am_hal_iom_config_t i2cConfig = {
        .eInterfaceMode = AM_HAL_IOM_I2C_MODE,
        .ui32ClockFreq = AM_HAL_IOM_100KHZ,
    };

    retVal32 = am_hal_iom_initialize(AM_BSP_ACCELEROMETER_I2C_IOM, &(dev_if.iomHandle));
    if (retVal32 != AM_HAL_STATUS_SUCCESS) { return retVal32; }

    retVal32 = am_hal_iom_power_ctrl((dev_if.iomHandle), AM_HAL_SYSCTRL_WAKE, false);
    if (retVal32 != AM_HAL_STATUS_SUCCESS) { return retVal32; }

    retVal32 = am_hal_iom_configure((dev_if.iomHandle), &i2cConfig);
    if (retVal32 != AM_HAL_STATUS_SUCCESS) { return retVal32; }

    am_hal_gpio_pinconfig(AM_BSP_ACCELEROMETER_I2C_SDA_PIN, g_AM_BSP_ACCELEROMETER_I2C_SDA);
    am_hal_gpio_pinconfig(AM_BSP_ACCELEROMETER_I2C_SCL_PIN, g_AM_BSP_ACCELEROMETER_I2C_SCL);

    retVal32 = am_hal_iom_enable((dev_if.iomHandle));
    if (retVal32 != AM_HAL_STATUS_SUCCESS) { return retVal32; }

    lis2dh12_device_id_get(&dev_ctx, &whoamI);
    *whoami_out = whoamI;
    if (whoamI != LIS2DH12_ID) { return AM_HAL_STATUS_FAIL; }

    lis2dh12_block_data_update_set(&dev_ctx, PROPERTY_ENABLE);
    lis2dh12_data_rate_set(&dev_ctx, LIS2DH12_ODR_50Hz);
    lis2dh12_full_scale_set(&dev_ctx, LIS2DH12_2g);
    lis2dh12_operating_mode_set(&dev_ctx, LIS2DH12_HR_12bit);

    return 0;
}

int main(void)
{
    uint32_t stat;
    uint8_t whoami = 0;
    uint32_t overruns = 0, i2c_errors = 0, max_us = 0;
    int i;

    // Board-agnostic setup (as in SparkFun's example)
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    // Board-specific setup
    am_bsp_low_power_init();
    am_bsp_uart_printf_enable();

    timer_init();

    stat = initAccel(&whoami);
    if (stat != 0) {
        while (1) {
            am_util_stdio_printf("# error: accelerometer init failed, code %u, whoami 0x%02x (expected 0x33)\r\n",
                                 (unsigned)stat, (unsigned)whoami);
            am_util_delay_ms(1000);
        }
    }
    if (gp_init(&g_gp, TICK_HZ, 0u) != 0) {
        while (1) {
            am_util_stdio_printf("# error: gp_init refused the tick rate\r\n");
            am_util_delay_ms(1000);
        }
    }

    am_util_stdio_printf("# sparkfun_edge gesture demo v1, whoami 0x%02x\r\n", (unsigned)whoami);
    am_util_stdio_printf("# odr_hz=%d tick_hz=%u timer=hfrc3mhz window=%d step=%d features=%d classes=",
                         ODR_HZ, (unsigned)TICK_HZ, GP_WINDOW, GP_STEP, GESTURE_N_FEATURES);
    for (i = 0; i < GESTURE_N_CLASSES; ++i) {
        am_util_stdio_printf("%s%s", i ? "," : "", GESTURE_CLASS_NAMES[i]);
    }
    am_util_stdio_printf("\r\n# columns: G,window,t_ticks,gesture,compute_us\r\n");

    while (1) {
        lis2dh12_status_reg_t status;
        uint32_t t, t_start, t_end, us;
        int x, y, z, done;

        if (lis2dh12_status_get(&dev_ctx, &status) != 0) { i2c_errors++; continue; }
        if (!status.zyxda) { continue; }

        t = am_hal_stimer_counter_get();       // arrival time of this sample

        if (lis2dh12_acceleration_raw_get(&dev_ctx, data_raw_acceleration.u8bit) != 0) { i2c_errors++; continue; }
        if (status.zyxor) { overruns++; }      // a newer sample overwrote one we had not read

        // 12-bit high-resolution mode, +-2 g: the value sits in the top 12 bits and one count is 1 mg
        x = data_raw_acceleration.i16bit[0] >> 4;
        y = data_raw_acceleration.i16bit[1] >> 4;
        z = data_raw_acceleration.i16bit[2] >> 4;

        t_start = am_hal_stimer_counter_get();
        done = gp_push(&g_gp, t, x, y, z);
        t_end = am_hal_stimer_counter_get();

        if (!done) { continue; }

        us = (t_end - t_start) / TICKS_PER_US;     // wrap-safe for the unsigned 32-bit counter
        if (us > max_us) { max_us = us; }
        am_util_stdio_printf("G,%u,%u,%s,%u\r\n", (unsigned)gp_windows(&g_gp), (unsigned)t,
                             GESTURE_CLASS_NAMES[gp_class(&g_gp)], (unsigned)us);

        if (gp_windows(&g_gp) % STATS_EVERY_WINDOWS == 0u) {
            am_util_stdio_printf("# stats windows=%u overruns=%u i2c_err=%u restarts=%u ignored=%u max_compute_us=%u\r\n",
                                 (unsigned)gp_windows(&g_gp), (unsigned)overruns, (unsigned)i2c_errors,
                                 (unsigned)gp_restarts(&g_gp), (unsigned)gp_ignored(&g_gp), (unsigned)max_us);
        }
    }
}
