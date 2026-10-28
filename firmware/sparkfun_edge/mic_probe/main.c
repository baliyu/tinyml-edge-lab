/*
mic_probe: first look at the SparkFun Edge's analog microphone (MIC1, GPIO 29, ADC single-ended channel 1).

What it does: samples the microphone at about 8000 Hz with the ADC (a timer triggers each conversion, DMA moves the
results into memory), and once per second prints one CSV line over the serial port (115200 baud):

  # mic_probe v1 ...
  # sec,n,min,max,mean,rms,rate_hz
  1,8000,8001,8412,8200,31,7999

  n        samples received in that second (8000 expected; fewer means lost samples)
  min,max  smallest and largest ADC value (14-bit counts)
  mean     average value (the microphone's resting level)
  rms      standard deviation around the mean: a simple loudness number. Silence is small, a clap is large.
  rate_hz  measured sample rate, from the 32768 Hz crystal timer (n * 32768 / ticks)

Adapted from the Ambiq SDK example adc_lpmode0_dma (timer-triggered ADC with DMA) and from this repository's
imu_logger (serial output and timer setup). The statistics are in mic_stats.c, which has a host test and is
included below so the SDK's Makefile needs no change.

Tested on one SparkFun Edge: sampling rate and quiet level checked; sharp claps clip at 16383 with the 2.0 V reference.
*/

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

#include "mic_stats.c"   /* pure statistics, host-tested; included so the SDK Makefile needs no change */

#define FS_HZ         8000u
#define CTIMER_HZ     11170500u               /* measured on this board: 8000 Hz at 11,170,500; a 12 MHz value gave 7447 Hz */
#define BLOCK         100u                    /* samples per DMA block; 8000 / 100 = 80 blocks per second */
#define BLOCKS_PER_S  (FS_HZ / BLOCK)
#define TICK_HZ       32768u                  /* crystal timer used for the measured rate */

static void *g_adc = NULL;
static uint32_t g_dma_words[BLOCK];
static am_hal_adc_sample_t g_samples[BLOCK];
static volatile bool g_dma_done = false;
static volatile bool g_dma_err = false;

/* GPIO 29 is the microphone near the LEDs; AM_HAL_PIN_29_ADCSE1 routes it to ADC channel 1 */
static const am_hal_gpio_pincfg_t g_pin_mic1 = { .uFuncSel = AM_HAL_PIN_29_ADCSE1 };

void am_adc_isr(void)
{
    uint32_t mask;
    if (am_hal_adc_interrupt_status(g_adc, &mask, false) != AM_HAL_STATUS_SUCCESS) return;
    am_hal_adc_interrupt_clear(g_adc, mask);
    if (mask & AM_HAL_ADC_INT_DCMP) g_dma_done = true;
    if (mask & AM_HAL_ADC_INT_DERR) g_dma_err = true;
}

static int arm_dma(void)
{
    am_hal_adc_dma_config_t c;
    c.bDynamicPriority = true;
    c.ePriority = AM_HAL_ADC_PRIOR_SERVICE_IMMED;
    c.bDMAEnable = true;
    c.ui32SampleCount = BLOCK;
    c.ui32TargetAddress = (uint32_t)g_dma_words;
    return (am_hal_adc_configure_dma(g_adc, &c) == AM_HAL_STATUS_SUCCESS) ? 0 : -1;
}

static int adc_init(void)
{
    am_hal_adc_config_t cfg;
    am_hal_adc_slot_config_t slot;

    if (am_hal_adc_initialize(0, &g_adc) != AM_HAL_STATUS_SUCCESS) return -1;
    if (am_hal_adc_power_control(g_adc, AM_HAL_SYSCTRL_WAKE, false) != AM_HAL_STATUS_SUCCESS) return -2;

    cfg.eClock     = AM_HAL_ADC_CLKSEL_HFRC;
    cfg.ePolarity  = AM_HAL_ADC_TRIGPOL_RISING;
    cfg.eTrigger   = AM_HAL_ADC_TRIGSEL_SOFTWARE;     /* the timer trigger is enabled below, as in the SDK example */
    cfg.eReference = AM_HAL_ADC_REFSEL_INT_2P0;       /* 2.0 V reference: more headroom than 1.5 V if the mic idles near mid-rail */
    cfg.eClockMode = AM_HAL_ADC_CLKMODE_LOW_LATENCY;
    cfg.ePowerMode = AM_HAL_ADC_LPMODE0;
    cfg.eRepeat    = AM_HAL_ADC_REPEATING_SCAN;
    if (am_hal_adc_configure(g_adc, &cfg) != AM_HAL_STATUS_SUCCESS) return -3;

    slot.eMeasToAvg     = AM_HAL_ADC_SLOT_AVG_1;      /* no averaging: every conversion is one audio sample */
    slot.ePrecisionMode = AM_HAL_ADC_SLOT_14BIT;
    slot.eChannel       = AM_HAL_ADC_SLOT_CHSEL_SE1;
    slot.bWindowCompare = false;
    slot.bEnabled       = true;
    if (am_hal_adc_configure_slot(g_adc, 0, &slot) != AM_HAL_STATUS_SUCCESS) return -4;

    if (arm_dma() != 0) return -5;
    am_hal_adc_interrupt_enable(g_adc, AM_HAL_ADC_INT_DERR | AM_HAL_ADC_INT_DCMP);
    if (am_hal_adc_enable(g_adc) != AM_HAL_STATUS_SUCCESS) return -6;
    return 0;
}

static void start_sample_timer(void)
{
    am_hal_ctimer_config_single(3, AM_HAL_CTIMER_TIMERA,
                                AM_HAL_CTIMER_HFRC_12MHZ | AM_HAL_CTIMER_FN_REPEAT | AM_HAL_CTIMER_INT_ENABLE);
    am_hal_ctimer_int_enable(AM_HAL_CTIMER_INT_TIMERA3);
    am_hal_ctimer_period_set(3, AM_HAL_CTIMER_TIMERA, CTIMER_HZ / FS_HZ, (CTIMER_HZ / FS_HZ) / 2);
    am_hal_ctimer_adc_trigger_enable();
    am_hal_ctimer_start(3, AM_HAL_CTIMER_TIMERA);
}

int main(void)
{
    mic_win_t win;
    mic_summary_t sum;
    uint32_t blocks = 0, sec = 0, bad_reads = 0, t_prev;
    int rc;

    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
    am_bsp_low_power_init();
    am_bsp_uart_printf_enable();

    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_XTAL_START, 0);
    am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
    am_hal_stimer_config(AM_HAL_STIMER_XTAL_32KHZ);

    am_util_stdio_printf("# mic_probe v1 fs_target=%u block=%u pin=29 adc=SE1 ref=2p0 avg=1 bits=14\r\n",
                         (unsigned)FS_HZ, (unsigned)BLOCK);

    am_hal_gpio_pinconfig(29, g_pin_mic1);
    start_sample_timer();
    NVIC_EnableIRQ(ADC_IRQn);
    am_hal_interrupt_master_enable();

    rc = adc_init();
    if (rc != 0) {
        am_util_stdio_printf("# error: adc_init failed at step %d\r\n", rc);
        while (1) { }
    }
    if (am_hal_adc_sw_trigger(g_adc) != AM_HAL_STATUS_SUCCESS) {
        am_util_stdio_printf("# error: first ADC trigger failed\r\n");
        while (1) { }
    }

    am_util_stdio_printf("# sec,n,min,max,mean,rms,rate_hz\r\n");
    mic_win_reset(&win);
    t_prev = am_hal_stimer_counter_get();

    while (1) {
        uint32_t n = BLOCK;

        while (!g_dma_done && !g_dma_err) { }
        if (g_dma_err) {
            am_util_stdio_printf("# error: ADC DMA error\r\n");
            while (1) { }
        }

        /* unpack the raw words, then clear the flag and re-arm the DMA at once so no samples are lost while we print */
        if (am_hal_adc_samples_read(g_adc, false, g_dma_words, &n, g_samples) != AM_HAL_STATUS_SUCCESS) {
            bad_reads++;
            n = 0;
        }
        g_dma_done = false;
        if (arm_dma() != 0) {
            am_util_stdio_printf("# error: could not re-arm the DMA\r\n");
            while (1) { }
        }

        for (uint32_t i = 0; i < n; i++) mic_win_add(&win, g_samples[i].ui32Sample);

        if (++blocks == BLOCKS_PER_S) {
            uint32_t t_now = am_hal_stimer_counter_get();
            uint32_t dt = t_now - t_prev;          /* wraps correctly in unsigned arithmetic */
            uint32_t rate = (dt > 0) ? (uint32_t)(((uint64_t)win.n * TICK_HZ) / dt) : 0;
            mic_win_summary(&win, &sum);
            am_util_stdio_printf("%u,%u,%u,%u,%u,%u,%u\r\n", (unsigned)++sec, (unsigned)sum.n, (unsigned)sum.min,
                                 (unsigned)sum.max, (unsigned)sum.mean, (unsigned)sum.rms, (unsigned)rate);
            if (bad_reads) {
                am_util_stdio_printf("# warning: %u block(s) could not be unpacked\r\n", (unsigned)bad_reads);
                bad_reads = 0;
            }
            mic_win_reset(&win);
            blocks = 0;
            t_prev = t_now;
        }
    }
}