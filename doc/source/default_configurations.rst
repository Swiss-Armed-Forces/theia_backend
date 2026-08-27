IRIS-T (BODLUV MR)
------------------

https://www.vbs.admin.ch/de/newnsb/ZjsNYPpgoETj

Sensor: Hensoldt TRML-4D

Source: https://dam.hensoldt.net/m/23b5fe680f976d6f/original/TRML-4D-English.pdf

* C-Band -> 6 GHz
* Range: 120km fighter aircraft, 60km supersonic missile
* Elevation: -2 - +70°
* Accuracy: 0.2° azimuth, 0.3° elevation, 15m range

Range resolution:

dr = c / (2 B)
-> B = c / (2 * dr) ~= 10 MHz

Angle resolution:

da = c / (f * D)
-> D = c / (f * da) ~= 14.3m (0.2° assumption)

Rotation time: 3s (assumption from Google AI search)
-> coherent integration time = da *  rotation_period / (360°) = 0.2 * 3 / 360 ~= 0.0017 s
