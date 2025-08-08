Version 4
SymbolType CELL
LINE Normal -32 -48 -16 -32
LINE Normal -16 32 -32 48
LINE Normal -32 48 -32 -48
LINE Normal -64 -48 -64 -64
LINE Normal -64 64 -64 48
LINE Normal -48 48 -64 48
LINE Normal -48 16 -48 48
LINE Normal -64 16 -48 16
LINE Normal -64 -16 -64 16
LINE Normal -48 -16 -64 -16
LINE Normal -48 -48 -48 -16
LINE Normal -64 -48 -48 -48
LINE Normal -16 64 -16 32
LINE Normal -16 -32 -16 -64
LINE Normal -161 -49 -208 -49
LINE Normal -161 1 -208 1
LINE Normal -161 48 -208 48
WINDOW 0 -33 -80 Bottom 2
SYMATTR SpiceLine chan_width=1u, heater_width=100n, chan_thickness=23.6n, chan_length=14u, sheet_resistance=77.9, heater_resistance=50, critical_temp=12.5, substrate_temp=4.2, eta=3, Jsw_tilde=88.3G, Isupp_tilde=389.7u, Jchanr=100G, tau_on=11.85n, ICh_bias_on=280u, Ih_bias_on=1455u
SYMATTR ModelFile hTron_behavioral.lib
SYMATTR SpiceModel hTron_behav
SYMATTR Prefix X
PIN -64 -64 RIGHT 8
PINATTR PinName heater_p
PINATTR SpiceOrder 1
PIN -64 64 RIGHT 8
PINATTR PinName heater_n
PINATTR SpiceOrder 2
PIN -16 -64 LEFT 8
PINATTR PinName drain
PINATTR SpiceOrder 3
PIN -16 64 LEFT 8
PINATTR PinName source
PINATTR SpiceOrder 4
PIN -208 -48 TOP 8
PINATTR PinName B1
PINATTR SpiceOrder 5
PIN -208 0 TOP 8
PINATTR PinName B2
PINATTR SpiceOrder 6
PIN -208 48 TOP 8
PINATTR PinName B3
PINATTR SpiceOrder 7
