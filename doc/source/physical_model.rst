Physical Model
==============

.. _snr-section:

SNR formula
-----------

The detection models in `theia` are based on the book by Barton :footcite:p:`Barton2004-ru`,
which is an excellent resource to get started in the field of radar technology.

The signal-to-noise ratio (SNR) for the monostatic case is given by Equ. (1.18)

.. math::

   SNR = \frac{P_t \tau n G_t G_r \lambda^2 \sigma F_p^2 F_t^2 F_r^2}{(4 \pi)^3 k_B T_S R^4 L_t L_a}.

This formula is valid for the bistatic case as well by replacing :math:`R^2 = R_t R_r` according to Sec. (1.2.2) in Barton :footcite:p:`Barton2004-ru`.

Quantities
----------

The following table summarises the meaning of each quantity.

.. list-table:: Physical Quantities
   :header-rows: 1

   * - Quantity
     - SI Unit
     - Name
     - Description
   * - :math:`P_t`
     - W
     - Power of the transmitter
     - Power the transmitter would have if it radiated isotropically
   * - :math:`\tau`
     - s
     - Pulse width
     - Duration of a single pulse
   * - :math:`n`
     - dimensionless
     - Number of pulses in coherent integration interval
     - 
   * - :math:`G_t`
     - dimensionless
     - Transmitting antenna power gain
     - Modifies :math:`P_t` to obtain the power in the direction of the beam
   * - :math:`G_r`
     - dimensionless
     - Receiving antenna power gain
     - Takes into account how much of the signal is captured by the receiver antenna
   * - :math:`\lambda`
     - m
     - Signal wavelength
     - 
   * - :math:`\sigma`
     - :math:`\textrm{m}^2`
     - Target radar cross section (RCS)
     - Area of a spherical target equivalent to the actual target in terms of reflected power density
   * - :math:`F_p^2`
     - dimensionless
     - Polarization factor
     - | "A factor :math:`F_p^2` modifies the target RCS to account for
       | the polarizations of the transmitting and receiving antennas,
       | in cases where the received echo is not captured efficiently by
       | the receiving antennas"
   * - :math:`F_t`
     - dimensionless
     - Pattern propagation factor for the transmitting path
     - | "[...] F is defined [...] to account for departures of the one-way
       | field strength from the value applicable to free-space propagation
       | along the antenna beam axis."
   * - :math:`F_r`
     - dimensionless
     - Pattern propagation factor for the receiving path
     - see description for :math:`F_t`
   * - :math:`k_B`
     - J / K
     - Boltzmann constant
     - 1.380649 * 10-23 J / K
   * - :math:`T_S`
     - K
     - Effective noise temperature of the receiver
     - This term summarises all kinds of thermal noise within and outside the radar.
   * - :math:`R`
     - m
     - Range
     - | One-way distance between radar and target.
       | Needs to be adjusted for the bistatic case (see previous section).
   * - :math:`L_t`
     - dimensionless
     - Transmission line loss
     - | "The loss between the transmitter output at which :math:`P_t``
       | is conventionally defined and the transmitting antenna terminal
       | at which :math:`G_t` is defined."
   * - :math:`L_a`
     - dimensionless
     - Atmospheric and precipitation attenuation
     - Accounts for wheather

Relations
---------

.. list-table:: Relations
   :header-rows: 1

   * - Name
     - Formula
     - Source in Barton :footcite:p:`Barton2004-ru`
   * - Effective radiated power
     - :math:`P_{ERP} = P_t G_t`
     - Sec. 1.2.1
   * - Receiver antenna gain
     - :math:`G_r = \frac{\lambda^2}{4 \pi A_r}`, where :math:`A_r` denotes the receiving antenna's aperture area
     - Equ. (1.7)
   * - Number of coherently integrated pulses
     - :math:`n = t_f f_r`, where :math:`t_f` is the coherent integration interval and :math:`f_r` is the pulse repetition rate.
     - Sec. (1.2.6)
   * - | Atmospheric and precipitation attenuation
       | (two-way loss)
     - | :math:`L_a = 10^{0.1 k_a R}`,
       | where :math:`R` is the distance between radar and target (range) and
       | :math:`k_a` is an attenuation coefficient in units of dB / m.
       | :math:`k_a` is taken from Tab. (6.1) in Barton :footcite:p:`Barton2004-ru`. 
     - Sec. (1.2.6)

.. footbibliography::