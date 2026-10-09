# SparkFun Edge notes (Apollo3 Blue)

Source: SparkFun Edge Hookup Guide (learn.sparkfun.com).

* **3.3 V only.** The board has no onboard 3.3 V regulator and is damaged by 5 V. Power it from a CR2032 coin cell or
  from a 3.3 V USB-serial adapter. Never connect a 5 V pin to it.
* The 6-pin UART header follows the SparkFun serial-breakout pinout (DTR, RXI, TXO, VCC, CTS, GND from the adapter's
  point of view). The silkscreen labels GRN and BLK mark the ends: DTR is by GRN, GND by BLK.
* Programming (TinyML book flow): hold button 14, press RST, then run Ambiq's `uart_wired_update.py`
  (115200 baud first). If it sits at "Sending Hello", check the wiring, try the other baud rate, and check the driver.

Record here what you actually did and what worked.
