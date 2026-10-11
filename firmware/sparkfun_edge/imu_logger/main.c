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
Fixed-rate CSV accelerometer logger for the SparkFun Edge (LIS2DH12).

Derived from SparkFun's lis2dh12_accelerometer_uart example (MIT licence, notice above). Changes:
  * the sensor runs at ODR_HZ (default 50 Hz, to stay comparable with the UCI HAR data) instead of 25 Hz
  * one CSV line per new sample, in integer milli-g, stamped with a hardware timer tick
  * the sensor's overrun flag and I2C errors are counted, and a '#' statistics line with the MEASURED sample rate is
    printed every 10 seconds
  * no terminal-clear escape codes, so the stream parses cleanly

Output at 115200 baud. Lines starting with '#' are comments:
  # odr_hz=50 fs_g=2 mode=hr12 units=mg tick_hz=32768 timer=xtal32k
  t_ticks,x_mg,y_mg,z_mg
  12345678,-541,507,544
  ...
  # stats samples=500 overruns=0 i2c_err=0 rate_hz=49.987 tick_hz=32768 timer=xtal32k
t_ticks is a free-running 32-bit counter (it wraps; unwrap it on the PC). Divide by tick_hz to get seconds.
*/

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

#include "lis2dh12_platform_apollo3.h"

#ifndef ODR_HZ
#define ODR_HZ 50
#endif

#if   ODR_HZ == 25
#define ODR_CODE LIS2DH12_ODR_25Hz
#elif ODR_HZ == 50
#define ODR_CODE LIS2DH12_ODR_50Hz
#elif ODR_HZ == 100
#define ODR_CODE LIS2DH12_ODR_100Hz
#else
#error "ODR_HZ must be 25, 50 or 100"
#endif

#define STATS_PERIOD_SAMPLES (ODR_HZ * 10)

static axis3bit16_t data_raw_acceleration;
static uint32_t g_tick_hz = 0;
static const char *g_timer_name = "none";

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

// Start a free-running timer. The board support code stops the 32 kHz crystal, so start it again and check that it
// really counts (about 1638 ticks in 50 ms). If it does not, fall back to the internal 3 MHz oscillator, which is
// less accurate; the choice is printed so the PC side knows which tick rate to use.
static void timer_init(void)
{
    uint32_t t0, dt;

    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_XTAL_START, 0);
    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_stimer_config(AM_HAL_STIMER_XTAL_32KHZ);

    am_util_delay_ms(20);                      // let the crystal settle before checking it
    t0 = am_hal_stimer_counter_get();
    am_util_delay_ms(50);
    dt = am_hal_stimer_counter_get() - t0;

    if (dt >= 1300 && dt <= 2000) {
        g_tick_hz = 32768;
        g_timer_name = "xtal32k";
        return;
    }

    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_stimer_config(AM_HAL_STIMER_HFRC_3MHZ);
    g_tick_hz = 3000000;
    g_timer_name = "hfrc3mhz";
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
    lis2dh12_data_rate_set(&dev_ctx, ODR_CODE);
    lis2dh12_full_scale_set(&dev_ctx, LIS2DH12_2g);
    lis2dh12_operating_mode_set(&dev_ctx, LIS2DH12_HR_12bit);

    return 0;
}

int main(void)
{
    uint32_t stat;
    uint8_t whoami = 0;
    uint32_t n_win = 0, t_first = 0, t_last = 0;
    uint32_t overruns = 0, i2c_errors = 0;

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

    am_util_stdio_printf("# sparkfun_edge lis2dh12 logger v1, whoami 0x%02x\r\n", (unsigned)whoami);
    am_util_stdio_printf("# odr_hz=%d fs_g=2 mode=hr12 units=mg tick_hz=%u timer=%s\r\n",
                         ODR_HZ, (unsigned)g_tick_hz, g_timer_name);
    am_util_stdio_printf("t_ticks,x_mg,y_mg,z_mg\r\n");

    while (1) {
        lis2dh12_status_reg_t status;
        uint32_t t;
        int x, y, z;

        if (lis2dh12_status_get(&dev_ctx, &status) != 0) { i2c_errors++; continue; }
        if (!status.zyxda) { continue; }

        t = am_hal_stimer_counter_get();       // arrival time of this sample

        if (lis2dh12_acceleration_raw_get(&dev_ctx, data_raw_acceleration.u8bit) != 0) { i2c_errors++; continue; }
        if (status.zyxor) { overruns++; }      // a newer sample overwrote one we had not read

        // 12-bit high-resolution mode, +-2 g: the value sits in the top 12 bits and one count is 1 mg
        x = data_raw_acceleration.i16bit[0] >> 4;
        y = data_raw_acceleration.i16bit[1] >> 4;
        z = data_raw_acceleration.i16bit[2] >> 4;

        am_util_stdio_printf("%u,%d,%d,%d\r\n", (unsigned)t, x, y, z);

        if (n_win == 0) { t_first = t; }
        t_last = t;
        n_win++;

        if (n_win >= STATS_PERIOD_SAMPLES) {
            uint32_t elapsed = t_last - t_first;   // wrap-safe for the unsigned 32-bit counter
            float rate = (elapsed > 0) ? ((float)(n_win - 1) * (float)g_tick_hz / (float)elapsed) : 0.0f;
            am_util_stdio_printf("# stats samples=%u overruns=%u i2c_err=%u rate_hz=%.3f tick_hz=%u timer=%s\r\n",
                                 (unsigned)n_win, (unsigned)overruns, (unsigned)i2c_errors, rate,
                                 (unsigned)g_tick_hz, g_timer_name);
            n_win = 0;
        }
    }
}
