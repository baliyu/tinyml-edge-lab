/* A fake SparkFun Edge, just enough of the Ambiq HAL, the BSP and the LIS2DH12 driver for ../main.c to build and run on the PC.
 * The accelerometer replays samples from a file, the timer is the sample's own tick, the UART is stdout. See host_sim.c. */
#ifndef FAKE_BOARD_H
#define FAKE_BOARD_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#define AM_HAL_STATUS_SUCCESS 0u
#define AM_HAL_STATUS_FAIL 1u
#define AM_HAL_IOM_I2C_MODE 0
#define AM_HAL_IOM_100KHZ 100000
#define AM_HAL_SYSCTRL_WAKE 1
#define AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX 0
#define AM_HAL_STIMER_CFG_CLEAR 1u
#define AM_HAL_STIMER_CFG_FREEZE 2u
#define AM_HAL_STIMER_HFRC_3MHZ 4u
#define AM_BSP_ACCELEROMETER_I2C_IOM 0
#define AM_BSP_ACCELEROMETER_I2C_ADDRESS 0x19
#define AM_BSP_ACCELEROMETER_I2C_SDA_PIN 40
#define AM_BSP_ACCELEROMETER_I2C_SCL_PIN 39

#define LIS2DH12_ID 0x33
#define PROPERTY_ENABLE 1
typedef enum { LIS2DH12_ODR_50Hz = 4 } lis2dh12_odr_t;
typedef enum { LIS2DH12_2g = 0 } lis2dh12_fs_t;
typedef enum { LIS2DH12_HR_12bit = 1 } lis2dh12_op_t;

typedef struct { int eInterfaceMode; int ui32ClockFreq; } am_hal_iom_config_t;
typedef struct { int dummy; } am_hal_cachectrl_config_t;
typedef int am_hal_gpio_pincfg_t;
extern const am_hal_cachectrl_config_t am_hal_cachectrl_defaults;
extern const am_hal_gpio_pincfg_t g_AM_BSP_ACCELEROMETER_I2C_SDA, g_AM_BSP_ACCELEROMETER_I2C_SCL;

typedef union { int16_t i16bit[3]; uint8_t u8bit[6]; } axis3bit16_t;
typedef struct { unsigned xda : 1; unsigned yda : 1; unsigned zda : 1; unsigned zyxda : 1; unsigned xor : 1; unsigned yor : 1; unsigned zor : 1; unsigned zyxor : 1; } lis2dh12_status_reg_t;
typedef int32_t (*lis2dh12_write_ptr)(void *, uint8_t, uint8_t *, uint16_t);
typedef int32_t (*lis2dh12_read_ptr)(void *, uint8_t, uint8_t *, uint16_t);
typedef struct { lis2dh12_write_ptr write_reg; lis2dh12_read_ptr read_reg; void *handle; } lis2dh12_ctx_t;
typedef struct { void *iomHandle; uint8_t addCS; bool useSPI; } lis2dh12_platform_apollo3_if_t;

uint32_t am_hal_iom_initialize(int module, void **handle);
uint32_t am_hal_iom_power_ctrl(void *handle, int state, bool retain);
uint32_t am_hal_iom_configure(void *handle, am_hal_iom_config_t *cfg);
uint32_t am_hal_iom_enable(void *handle);
void am_hal_gpio_pinconfig(int pin, am_hal_gpio_pincfg_t cfg);
void am_hal_clkgen_control(int control, void *arg);
void am_hal_cachectrl_config(const am_hal_cachectrl_config_t *cfg);
void am_hal_cachectrl_enable(void);
void am_bsp_low_power_init(void);
void am_bsp_uart_printf_enable(void);
void am_hal_stimer_config(uint32_t cfg);
uint32_t am_hal_stimer_counter_get(void);
void am_util_stdio_printf(const char *fmt, ...);
void am_util_delay_ms(uint32_t ms);

int32_t lis2dh12_write_platform_apollo3(void *handle, uint8_t reg, uint8_t *buf, uint16_t len);
int32_t lis2dh12_read_platform_apollo3(void *handle, uint8_t reg, uint8_t *buf, uint16_t len);
int32_t lis2dh12_device_id_get(lis2dh12_ctx_t *ctx, uint8_t *id);
int32_t lis2dh12_block_data_update_set(lis2dh12_ctx_t *ctx, uint8_t val);
int32_t lis2dh12_data_rate_set(lis2dh12_ctx_t *ctx, lis2dh12_odr_t val);
int32_t lis2dh12_full_scale_set(lis2dh12_ctx_t *ctx, lis2dh12_fs_t val);
int32_t lis2dh12_operating_mode_set(lis2dh12_ctx_t *ctx, lis2dh12_op_t val);
int32_t lis2dh12_status_get(lis2dh12_ctx_t *ctx, lis2dh12_status_reg_t *status);
int32_t lis2dh12_acceleration_raw_get(lis2dh12_ctx_t *ctx, uint8_t *buff);

#endif
