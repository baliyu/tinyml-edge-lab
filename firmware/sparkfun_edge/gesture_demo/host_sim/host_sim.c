/* The other half of the fake board: the HAL and driver functions that ../main.c calls, on the PC.
 *
 * Input: the file named by the environment variable SIM_SAMPLES, one sample per line:  ticks,x_mg,y_mg,z_mg[,flags]
 *   flags bit 0: the sensor reports that it overwrote an unread sample (zyxor)
 *   flags bit 1: the status read for this sample fails once (an I2C error)
 * The "timer" returns the current sample's tick and then moves on by 3000 ticks (1000 us) at every read, so the time main.c measures
 * around gp_push() is a known 1000 us. SIM_WHOAMI (hex) overrides the sensor's id, to test the start-up failure path.
 * If SIM_LOG names a file, the sensor and timer configuration calls and every sample main.c hands to the chain are written to it.
 * The program ends when the samples run out (or after 3 delays in the failure loop). It is a test aid, not part of the firmware. */
#include <stdio.h>
#include <stdlib.h>

#include "fake_board.h"
#include "gesture_pipeline.h"

const am_hal_cachectrl_config_t am_hal_cachectrl_defaults = {0};
const am_hal_gpio_pincfg_t g_AM_BSP_ACCELEROMETER_I2C_SDA = 1, g_AM_BSP_ACCELEROMETER_I2C_SCL = 2;

static FILE *g_in;
static unsigned long g_ticks, g_flags;
static long g_vx, g_vy, g_vz;
static uint32_t g_now;
static int g_delays;
static FILE *g_log;

static void note(const char *fmt, ...)
{
    va_list ap;
    if (!g_log) { const char *p = getenv("SIM_LOG"); if (!p) return; g_log = fopen(p, "a"); if (!g_log) return; }
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fflush(g_log);
}

/* main.c is built with -Dgp_push=sim_push so that what it passes to the chain can be recorded; the chain itself is the real one */
int sim_push(gp_state *g, uint32_t ticks, int32_t x, int32_t y, int32_t z)
{
    note("push,%lu,%ld,%ld,%ld\n", (unsigned long)ticks, (long)x, (long)y, (long)z);
    return gp_push(g, ticks, x, y, z);
}

uint32_t am_hal_iom_initialize(int module, void **handle) { (void)module; *handle = (void *)&g_now; return AM_HAL_STATUS_SUCCESS; }
uint32_t am_hal_iom_power_ctrl(void *handle, int state, bool retain) { (void)handle; (void)state; (void)retain; return AM_HAL_STATUS_SUCCESS; }
uint32_t am_hal_iom_configure(void *handle, am_hal_iom_config_t *cfg) { (void)handle; (void)cfg; return AM_HAL_STATUS_SUCCESS; }
uint32_t am_hal_iom_enable(void *handle) { (void)handle; return AM_HAL_STATUS_SUCCESS; }
void am_hal_gpio_pinconfig(int pin, am_hal_gpio_pincfg_t cfg) { (void)pin; (void)cfg; }
void am_hal_clkgen_control(int control, void *arg) { (void)control; (void)arg; }
void am_hal_cachectrl_config(const am_hal_cachectrl_config_t *cfg) { (void)cfg; }
void am_hal_cachectrl_enable(void) {}
void am_bsp_low_power_init(void) {}
void am_hal_stimer_config(uint32_t cfg) { note("stimer,%lu\n", (unsigned long)cfg); }
uint32_t am_hal_stimer_counter_get(void) { uint32_t t = g_now; g_now += 3000u; return t; }

void am_bsp_uart_printf_enable(void)
{
    const char *path = getenv("SIM_SAMPLES");
    g_in = path ? fopen(path, "r") : NULL;
    if (!g_in) { fprintf(stderr, "host_sim: set SIM_SAMPLES to a file of ticks,x,y,z[,flags] lines\n"); exit(2); }
}

void am_util_stdio_printf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

void am_util_delay_ms(uint32_t ms)
{
    (void)ms;
    if (++g_delays >= 3) { fflush(stdout); exit(0); }      /* the start-up failure loop would never end */
}

int32_t lis2dh12_write_platform_apollo3(void *handle, uint8_t reg, uint8_t *buf, uint16_t len) { (void)handle; (void)reg; (void)buf; (void)len; return 0; }
int32_t lis2dh12_read_platform_apollo3(void *handle, uint8_t reg, uint8_t *buf, uint16_t len) { (void)handle; (void)reg; (void)buf; (void)len; return 0; }

int32_t lis2dh12_device_id_get(lis2dh12_ctx_t *ctx, uint8_t *id)
{
    const char *w = getenv("SIM_WHOAMI");
    (void)ctx;
    *id = (uint8_t)(w ? strtoul(w, NULL, 16) : LIS2DH12_ID);
    return 0;
}
int32_t lis2dh12_block_data_update_set(lis2dh12_ctx_t *ctx, uint8_t val) { (void)ctx; note("block_data_update,%u\n", (unsigned)val); return 0; }
int32_t lis2dh12_data_rate_set(lis2dh12_ctx_t *ctx, lis2dh12_odr_t val) { (void)ctx; note("data_rate,%s\n", val == LIS2DH12_ODR_50Hz ? "50Hz" : "other"); return 0; }
int32_t lis2dh12_full_scale_set(lis2dh12_ctx_t *ctx, lis2dh12_fs_t val) { (void)ctx; note("full_scale,%s\n", val == LIS2DH12_2g ? "2g" : "other"); return 0; }
int32_t lis2dh12_operating_mode_set(lis2dh12_ctx_t *ctx, lis2dh12_op_t val) { (void)ctx; note("operating_mode,%s\n", val == LIS2DH12_HR_12bit ? "hr12" : "other"); return 0; }

int32_t lis2dh12_status_get(lis2dh12_ctx_t *ctx, lis2dh12_status_reg_t *status)
{
    char line[128];
    (void)ctx;
    status->xda = status->yda = status->zda = status->zyxda = status->xor = status->yor = status->zor = status->zyxor = 0;
    if (!fgets(line, sizeof line, g_in)) { fflush(stdout); exit(0); }          /* the recording is over */
    g_flags = 0;
    if (sscanf(line, "%lu,%ld,%ld,%ld,%lu", &g_ticks, &g_vx, &g_vy, &g_vz, &g_flags) < 4) { fprintf(stderr, "host_sim: bad line: %s", line); exit(2); }
    g_now = (uint32_t)g_ticks;
    if (g_flags & 2u) { return -1; }                                            /* this status read fails; the sample is lost */
    status->zyxda = 1;
    status->zyxor = (g_flags & 1u) ? 1 : 0;
    return 0;
}

int32_t lis2dh12_acceleration_raw_get(lis2dh12_ctx_t *ctx, uint8_t *buff)
{
    int16_t v[3];
    (void)ctx;
    v[0] = (int16_t)(g_vx * 16); v[1] = (int16_t)(g_vy * 16); v[2] = (int16_t)(g_vz * 16);   /* 12-bit value in the top of 16 bits */
    for (int i = 0; i < 6; ++i) { buff[i] = ((uint8_t *)v)[i]; }
    return 0;
}
